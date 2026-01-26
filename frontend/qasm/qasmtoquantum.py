"""
#   Frontend generating Quantum dialect code from QASM2 and QASM3 code.
#
# @author  Lars Schütze (lars.schuetze@tu-dresden.de)
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from enum import Enum

from mlir._mlir_libs._mlirDialectsQPU import qpu as qpu_dialect
from mlir._mlir_libs._mlirDialectsQuantum import QuantumMeasurementType, QuantumQubitType
from mlir._mlir_libs._mlirDialectsQuantum import quantum as quantum_dialect
from mlir._mlir_libs._mlirDialectsRVSDG import ControlType, MatchRuleAttr
from mlir._mlir_libs._mlirDialectsRVSDG import rvsdg as rvsdg_dialect
from mlir.dialects import arith, func, qpu, quantum, rvsdg, tensor, vector
from mlir.dialects.arith import CmpIPredicate
from mlir.dialects.builtin import (
    Block,
    BlockArgumentList,
    DenseElementsAttr,
    DenseIntElementsAttr,
    FunctionType,
    IndexType,
    IntegerType,
    RankedTensorType,
)
from mlir.dialects.vector import CombiningKind
from mlir.ir import (
    ArrayAttr,
    Context,
    F64Type,
    InsertionPoint,
    IntegerAttr,
    Location,
    Module,
    StringAttr,
    SymbolRefAttr,
    Type,
    TypeAttr,
    Value,
)
from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    Instruction,
    Operation,
    ParameterExpression,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit import library as lib
from qiskit.circuit.classical.expr import Expr
from qiskit.circuit.controlflow.if_else import IfElseOp
from qiskit.qasm2 import loads as qasm2_loads
from qiskit.qasm2.parse import LEGACY_CUSTOM_INSTRUCTIONS
from qiskit.qasm2.parse import _DefinedGate as QASM2_Gate


class ConversionError(RuntimeError): ...


class ParseError(RuntimeError): ...


# Specify the QASM version the frontend conforms to
class QASMVersion(Enum):
    Unspecified = 0
    QASM_2_0 = 1
    QASM_3_0 = 2


# Types that can be coerced to a valid Qubit specifier in a circuit.
type QubitSpecifier = Qubit | QuantumRegister
# int,
# slice,
# Sequence[Union[Qubit, int]],

# Types that can be coerced to a valid Clbit specifier in a circuit.
type ClbitSpecifier = Clbit | ClassicalRegister
# int,
# slice,
# Sequence[Union[Clbit, int]],


@dataclass(order=True)
class Interval:
    start: int
    end: int
    value: Value = field(compare=False)

    def contains(self, key: int) -> bool:
        return self.start <= key <= self.end

    def __len__(self):
        return self.end - self.start + 1


class IntervalMap:
    def __init__(self):
        self._intervals: list[Interval] = []

    def __len__(self):
        return len(self._intervals)

    def get(self, key: int) -> Value | None:
        starts = [iv.start for iv in self._intervals]
        i = bisect_right(starts, key) - 1
        if i >= 0 and self._intervals[i].contains(key):
            return self._intervals[i].value
        return None

    def interval_containing(self, key: int) -> Interval:
        starts = [iv.start for iv in self._intervals]
        i = bisect_right(starts, key) - 1
        if i < 0 or key > self._intervals[i].end:
            raise KeyError(key)
        return self._intervals[i]

    def add(self, start: int, end: int, value: Value):
        if start > end:
            raise ValueError(f"start {start} must be <= end {end}")

        new = Interval(start, end, value)
        i = bisect_left(self._intervals, new)

        # Check left neighbor
        if i > 0 and self._intervals[i - 1].end >= start:
            raise ValueError("Overlapping or touching interval")

        # Check right neighbor
        if i < len(self._intervals) and self._intervals[i].start <= end:
            raise ValueError("Overlapping or touching interval")

        self._intervals.insert(i, new)

    def remove(self, start: int, end: int) -> Value:
        starts = [iv.start for iv in self._intervals]
        i = bisect_left(starts, start)

        if i < len(self._intervals):
            iv = self._intervals[i]
            if iv.start == start and iv.end == end:
                self._intervals.pop(i)
                return iv.value

        raise KeyError(f"Interval [{start}, {end}] not found")

    def replace_interval(self, old: Interval, new: list[Interval]):
        self.remove(old.start, old.end)
        for iv in new:
            self.add(iv.start, iv.end, iv.value)

    def __repr__(self):
        return f"IntervalMap({self._intervals})"


class Scope:
    def __init__(self):
        # Maps circuit quantum registers to quantum.qubit values
        self._registers: dict[QuantumRegister, IntervalMap] = {}
        # Holds already visited gates that can be reused and have not to be revisited.
        self._visited_gates: dict[str, quantum.GateOp] = {}
        # Maps circuit classical registers of size N to tensor<1xN> values
        self.cregs: dict[ClassicalRegister, Value | None] = {}
        # Maps circuit classical bit to current quantum.qubit value
        self.clbits: dict[Clbit, Value | None] = {}

    @classmethod
    def from_scope(cls, other: Scope) -> Scope:
        new = cls()

        # Copy quantum registers + interval maps
        for qreg, im in other._registers.items():
            new_im = IntervalMap()
            for iv in im._intervals:
                new_im.add(iv.start, iv.end, iv.value)
            new._registers[qreg] = new_im

        new._visited_gates = dict(other._visited_gates)
        new.cregs = dict(other.cregs)
        new.clbits = dict(other.clbits)

        return new

    def intervals(self, reg: QuantumRegister):
        return self._registers.setdefault(reg, IntervalMap())

    def lookup(self, reg: QuantumRegister, index: int) -> Interval:
        iv = self.intervals(reg).interval_containing(index)
        if iv is None:
            raise KeyError(f"No qubit at index {index}")
        return iv

    def replace(self, reg: QuantumRegister, old: Interval, new: list[Interval]):
        imap = self.intervals(reg)
        imap.remove(old.start, old.end)
        for iv in new:
            imap.add(iv.start, iv.end, iv.value)

    def findResult(self, c: ClbitSpecifier) -> Value | None:
        if isinstance(c, ClassicalRegister):
            return self.cregs.get(c)
        if isinstance(c, Clbit):
            return self.clbits.get(c)

    def setResult(self, c: ClbitSpecifier, measurement: Value) -> None:
        if isinstance(c, ClassicalRegister):
            self.cregs[c] = measurement
        if isinstance(c, Clbit):
            self.clbits[c] = measurement

    def findGate(self, gate: QASM2_Gate) -> quantum.GateOp:
        return self._visited_gates.get(str(gate.name))

    def setGate(self, gate: QASM2_Gate, newGate: quantum.GateOp) -> None:
        self._visited_gates[str(gate.name)] = newGate

    def set_qreg(self, q: QuantumRegister, alloc: Value) -> Value:
        if q in self._registers:
            raise ParseError(f"Quantum register {q} already initialized")

        im = IntervalMap()
        im.add(0, len(q) - 1, alloc)
        self._registers[q] = im
        return alloc

    def find_qubit(self, q: Qubit | QuantumRegister) -> Value | None:
        if isinstance(q, QuantumRegister):
            im = self._registers.get(q)
            if im is None:
                return None
            if len(im) != 1:
                raise ParseError(f"Register {q} is fragmented into multiple intervals")
            return im._intervals[0].value

        if isinstance(q, Qubit):
            im = self._registers.get(q._register)
            if im is None:
                return None
            return im.get(q._index)

        raise TypeError(q)

    def bind_qubit(self, qreg: QuantumRegister, start: int, end: int, alloc: Value) -> Value:
        im = self._registers.get(qreg)
        if im is None:
            raise ParseError(f"Register {qreg} not initialized")

        # Ensure no overlapping ownership
        try:
            iv = im.interval_containing(start)
            raise ParseError(f"QuantumRegister {qreg} already owned by interval {iv}")
        except KeyError:
            pass  # good: slot is free

        im.add(start, end, alloc)
        return alloc

    def update_qubit(self, q: Qubit, new_value: Value) -> Value:
        im = self._registers.get(q._register)
        if im is None:
            raise ParseError(f"Register {q._register} not initialized")

        iv = im.interval_containing(q._index)

        # Must be exactly a single qubit
        if iv.start != iv.end:
            raise ParseError(f"Qubit {q} refers to multi-qubit interval {iv}")

        # Replace interval value
        iv.value = new_value
        return new_value


class QASMToMLIRVisitor:
    def __init__(
        self, compat: QASMVersion, context: Context, module: qpu.QPUModuleOp, loc: Location, block: Block, scope: Scope
    ) -> None:
        self.compat: QASMVersion = compat
        self.context: Context = context
        self.module: qpu.QPUModuleOp = module
        self.loc: Location = loc
        self.block: Block = block
        self.scope = scope

    @classmethod
    def fromParent(cls, parent: QASMToMLIRVisitor, *, block: Block | None = None, scope: Scope | None = None):
        return cls(
            parent.compat,
            parent.context,
            parent.module,
            parent.loc,
            parent.block if block is None else block,
            parent.scope if scope is None else scope,
        )

    def visitCircuit(self, circuit: QuantumCircuit, *, emitRegisters: bool = False) -> None:
        if emitRegisters:
            for qreg in circuit.qregs:
                self.visitQuantumRegister(qreg)
            for creg in circuit.cregs:
                self.visitClassicalRegister(creg)

        for instr in circuit.data:
            if isinstance(instr.operation, Expr):
                self.visitClassic(instr)
            elif isinstance(instr.operation, Instruction):
                self.visitInstruction(instr.operation, instr.qubits, instr.clbits)
            elif isinstance(instr.operation, Operation):
                self.visitQuantum(instr.operation)
            else:
                raise ParseError(f"Unknown instruction: {instr} of type {type(instr)}")

    def visitQuantumRegister(self, reg: QuantumRegister) -> Value:
        assert isinstance(reg, QuantumRegister)
        alloc: Value | None = self.scope.find_qubit(reg)
        if alloc is None:
            qubitTy: QuantumQubitType = QuantumQubitType.get(self.context, len(reg))
            qalloc: Value = quantum.alloc(qubitTy, loc=self.loc, ip=InsertionPoint(self.block))
            return self.scope.set_qreg(reg, qalloc)

        return alloc

    def visitQuantumBit(self, q: Qubit) -> Value:
        """
        Return the `!quantum.qubit<1>` value that represents `q`.
        If `q` is currently represented as a multi-qubit<N> value
        it wille be split accordingly. The new value is returned.
        """
        assert isinstance(q, Qubit)

        iv: Interval = self.scope.lookup(q._register, q._index)
        if len(iv) >= 2:
            # split iv and replace intervals
            if q._index == iv.start:
                # For q[0] we split qreg := q, qs
                qubitLTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
                qubitRTy: QuantumQubitType = QuantumQubitType.get(self.context, len(iv) - 1)
                split: quantum.SplitOp = quantum.split(
                    [qubitLTy, qubitRTy], iv.value, loc=self.loc, ip=InsertionPoint(self.block)
                )
                ileft: Interval = Interval(q._index, q._index, split[0])
                iright: Interval = Interval(q._index + 1, iv.end, split[1])
                self.scope.replace(q._register, iv, [ileft, iright])
                return ileft.value
            elif q._index == len(iv) - 1:
                # for q[n], n == len(q) we split qreg := ql, q
                qubitLTy: QuantumQubitType = QuantumQubitType.get(self.context, len(iv) - 1)
                qubitRTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
                split: quantum.SplitOp = quantum.SplitOp(
                    [qubitLTy, qubitRTy], iv.value, loc=self.loc, ip=InsertionPoint(self.block)
                )
                ileft: Interval = Interval(iv.start, iv.end - 1, split[0])
                iright: Interval = Interval(iv.end, iv.end, split[1])
                self.scope.replace(q._register, iv, [ileft, iright])
                return iright.value
            else:
                # For q[n], 0 <= n <= len(q) we split qreg := ql, q, qs
                qubitLTy: QuantumQubitType = QuantumQubitType.get(self.context, q._index - iv.start)
                qubitMidTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
                qubitRTy: QuantumQubitType = QuantumQubitType.get(self.context, iv.end - q._index)
                split: quantum.SplitOp = quantum.SplitOp(
                    [qubitLTy, qubitMidTy, qubitRTy], iv.value, loc=self.loc, ip=InsertionPoint(self.block)
                )
                ileft: Interval = Interval(iv.start, q._index - 1, split[0])
                imid: Interval = Interval(q._index, q._index, split[1])
                iright: Interval = Interval(q._index + 1, iv.end, split[2])
                self.scope.replace(q._register, iv, [ileft, imid, iright])
                return imid.value

        qv: Value | None = self.scope.find_qubit(q)
        if qv is None:
            raise ParseError(f"Error while retrieving SSA value for {q}")
        return qv

    def visitClassicalRegister(self, creg: ClassicalRegister) -> Value:
        assert isinstance(creg, ClassicalRegister)
        calloc: Value | None = self.scope.findResult(creg)
        if calloc is None:
            i1 = IntegerType.get_signless(1)
            tensor_ty = RankedTensorType.get([len(creg)], i1)
            buf = memoryview(bytes([0]))  # 8 zero bits
            attr = DenseIntElementsAttr.get(buf, type=tensor_ty)  # type: ignore
            zero = arith.constant(tensor_ty, attr, loc=self.loc, ip=InsertionPoint(self.block))
            self.scope.setResult(creg, zero)
            return zero

        return calloc

    def visitClassic(self, expr: Expr) -> Value:
        if isinstance(expr, ParameterExpression):
            raise NotImplementedError("Parameter Expression")
        elif isinstance(expr, float):
            return arith.ConstantOp(F64Type.get(self.context), expr, ip=InsertionPoint(self.block)).result
        else:
            raise NotImplementedError(f"Classic expressions are not supported for {expr}")

    # Operation encapsulates virtual instructions that must
    # be synthesized to physical instructions
    def visitQuantum(self, instr: Operation) -> None:
        raise NotImplementedError(f"Virtual quantum expressions are not supported for {instr}")

    # Instruction represents physical quantum instructions
    def visitInstruction(self, instr: Instruction, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        match instr, len(qubits):
            case lib.Barrier(), _:
                with self.loc:
                    args = [self.visitQuantumBit(q) for q in qubits]
                    outTy = [arg.type for arg in args]
                    outs: list[Value] = quantum.BarrierOp(outTy, args, ip=InsertionPoint(self.block)).result
                    for q, r in zip(qubits, outs):
                        self.scope.update_qubit(q, r)
            case QASM2_Gate(), _:
                self._visitDefinedGate(instr, qubits, clbits)
            case IfElseOp(), _:
                self._visitIfElse(instr, qubits, clbits)
            case lib.CCXGate(), 3:
                with self.loc:
                    control1: Value = self.visitQuantumBit(qubits[0])
                    control2: Value = self.visitQuantumBit(qubits[1])
                    target: Value = self.visitQuantumBit(qubits[2])
                    ccx: quantum.CCXOp = quantum.CCXOp(control1, control2, target, ip=InsertionPoint(self.block))
                    self.scope.update_qubit(qubits[0], ccx.control1_out)
                    self.scope.update_qubit(qubits[1], ccx.control2_out)
                    self.scope.update_qubit(qubits[2], ccx.target_out)
            case lib.CSwapGate(), 3:
                with self.loc:
                    control: Value = self.visitQuantumBit(qubits[0])
                    lhs: Value = self.visitQuantumBit(qubits[1])
                    rhs: Value = self.visitQuantumBit(qubits[2])
                    cswap: quantum.CSWAPOp = quantum.CSWAPOp(control, lhs, rhs, ip=InsertionPoint(self.block))
                    self.scope.update_qubit(qubits[0], cswap.control_out)
                    self.scope.update_qubit(qubits[1], cswap.lhs_out)
                    self.scope.update_qubit(qubits[2], cswap.rhs_out)
            case _, 1:
                self._visitUnaryGates(instr, qubits, clbits)
            case _, 2:
                self._visitBinaryGates(instr, qubits, clbits)
            case _, _:
                raise NotImplementedError(f"{instr} of type {type(instr)}")

    def _visitUnaryGates(self, instr: Instruction, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        assert len(qubits) == 1, f"Require unary gate, got: {instr}"
        match qubits[0]:
            case QuantumRegister():
                raise NotImplementedError(f"Visiting instruction {instr} with QuantumRegister {qubits} and {clbits}")
            case Qubit():
                with self.loc:
                    target: Value = self.visitQuantumBit(qubits[0])
                    match instr:
                        case lib.XGate():
                            op: quantum.XOp = quantum.XOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.YGate():
                            op: quantum.YOp = quantum.YOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.ZGate():
                            op: quantum.ZOp = quantum.ZOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.HGate():
                            op: quantum.HOp = quantum.HOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.SGate():
                            op: quantum.SOp = quantum.SOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.SXGate():
                            op: quantum.SXOp = quantum.SXOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.SdgGate():
                            op: quantum.SdgOp = quantum.SdgOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.TGate():
                            op: quantum.TOp = quantum.TOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.TdgGate():
                            op: quantum.TdgOp = quantum.TdgOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.RZGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RzOp = quantum.RzOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.RXGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RxOp = quantum.RxOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.RYGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RyOp = quantum.RyOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.U3Gate():
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U3Op = quantum.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.U2Gate():
                            phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U2Op = quantum.U2Op(target, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.U1Gate():
                            lam = self.visitClassic(instr.params[0])
                            op: quantum.U1Op = quantum.U1Op(target, lam, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.UGate():
                            # TODO: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.UGate
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U3Op = quantum.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.Reset():
                            outTy = QuantumQubitType.get(self.context, 1)
                            op: quantum.ResetOp = quantum.ResetOp(outTy, target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.Measure():
                            assert len(clbits) == 1
                            self._visitMeasure(instr, target, qubits[0], clbits[0])
                        case lib.IGate():
                            op: quantum.IdOp = quantum.IdOp(target, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case lib.PhaseGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.PhaseOp = quantum.PhaseOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result)
                        case _:
                            raise NotImplementedError(f"Unary gate {instr}")

    def _visitBinaryGates(self, instr: Instruction, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        assert len(qubits) == 2, f"Require binary gate, got: {instr}"
        match qubits[0], qubits[1]:
            case QuantumRegister(), QuantumRegister():
                raise NotImplementedError(f"Visiting instruction {instr} with QuantumRegister {qubits} and {clbits}")
            case Qubit(), Qubit():
                with self.loc:
                    lhs: Value = self.visitQuantumBit(qubits[0])
                    rhs: Value = self.visitQuantumBit(qubits[1])
                    match instr:
                        case lib.SwapGate():
                            op: quantum.SWAPOp = quantum.SWAPOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.result_lhs)
                            self.scope.update_qubit(qubits[1], op.result_rhs)
                        case lib.CZGate():
                            op: quantum.CZOp = quantum.CZOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.control_out)
                            self.scope.update_qubit(qubits[1], op.target_out)
                        case lib.CXGate():
                            op: quantum.CNOTOp = quantum.CNOTOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.control_out)
                            self.scope.update_qubit(qubits[1], op.target_out)
                        case lib.CRYGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CRyOp = quantum.CRyOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.control_out)
                            self.scope.update_qubit(qubits[1], op.target_out)
                        case lib.CRZGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CRzOp = quantum.CRzOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.control_out)
                            self.scope.update_qubit(qubits[1], op.target_out)
                        case lib.CU1Gate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CU1Op = quantum.CU1Op(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.update_qubit(qubits[0], op.control_out)
                            self.scope.update_qubit(qubits[1], op.target_out)

    def _visitMeasure(self, instr: lib.Measure, target: Value, qubit: QubitSpecifier, clbit: ClbitSpecifier) -> None:
        assert isinstance(clbit, Clbit)
        assert isinstance(qubit, Qubit)
        # Create the measurement
        measurementTy: QuantumMeasurementType = QuantumMeasurementType.get(self.context, 1)
        qubitTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
        op: quantum.MeasureOp = quantum.MeasureOp(measurementTy, qubitTy, target, ip=InsertionPoint(self.block))
        self.scope.update_qubit(qubit, op.result)
        # Create the tensor holding the result value
        i1 = IntegerType.get_signless(1)
        tensor_ty = RankedTensorType.get([1], i1)
        result_tensor = quantum.to_tensor(tensor_ty, op.measurement, loc=self.loc, ip=InsertionPoint(self.block))
        # Insert the result into the register
        creg = self.visitClassicalRegister(clbit._register)
        offset = index_const(clbit._index, context=self.context, loc=self.loc, ip=InsertionPoint(self.block))

        creg_tensor = tensor.insert_slice(
            source=result_tensor,
            dest=creg,
            offsets=[offset],  # dynamic offset
            sizes=[],  # no dynamic sizes
            strides=[],  # no dynamic strides
            static_offsets=[-1],  # -1 means "use operand"
            static_sizes=[1],  # compile-time constant
            static_strides=[1],  # compile-time constant
            loc=self.loc,
            ip=InsertionPoint(self.block),
        )
        self.scope.setResult(clbit._register, creg_tensor)

    def _visitDefinedGate(self, instr: QASM2_Gate, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        if instr.definition is not None:
            if self.scope.findGate(instr) is None:
                # Construct quantum.GateOp for defined custom gate
                # Insert into module body and recursively visit gate body
                inputTypes: list[QuantumQubitType] = [QuantumQubitType.get(self.context, 1)] * instr.num_qubits
                gty: func.FunctionType = func.FunctionType.get(inputs=inputTypes, results=inputTypes, context=self.context)
                gate: quantum.GateOp = quantum.GateOp(
                    StringAttr.get(str(instr.name)),
                    TypeAttr.get(gty),
                    loc=self.loc,
                    ip=InsertionPoint.at_block_begin(self.module.body),
                )
                gateBody: Block = gate.body.blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))

                self.scope.setGate(instr, gate)
                circuit: QuantumCircuit = instr.definition
                inner_gate_scope: Scope = Scope.from_scope(self.scope)
                for q, v in zip(circuit.qubits, gate.body.blocks[0].arguments):
                    inner_gate_scope.update_qubit(q, v)

                visitor: QASMToMLIRVisitor = QASMToMLIRVisitor.fromParent(self, block=gateBody, scope=inner_gate_scope)
                visitor.visitCircuit(circuit)

                quantum.ReturnOp([visitor.visitQuantumBit(q) for q in circuit.qubits], loc=self.loc, ip=InsertionPoint(gateBody))

            callee: StringAttr = instr.name
            operands: list[Value] = [self.visitQuantumBit(q) for q in qubits]
            outTys: list[Type] = [o.type for o in operands]
            op: quantum.GateCallOp = quantum.GateCallOp(outTys, callee, operands, loc=self.loc, ip=InsertionPoint(self.block))
            for inq, outq in zip(qubits, op.results):
                self.scope.update_qubit(inq, outq)
        else:
            ParseError(f"Expected gate with definition, got: {instr}")

    def _visitIfElse(self, instr: IfElseOp, qubits: list[QubitSpecifier], clbits: list[ClbitSpecifier]) -> None:
        with self.loc:
            condition: Value = self._visitIfElseCondition(instr.condition, qubits, clbits)
            true_body, false_body = instr.params
            hasElse: bool = false_body is not None

            matchTy: ControlType = ControlType.get(self.context, 2)
            mapTrue: MatchRuleAttr = MatchRuleAttr.get(self.context, [1], 0)
            mapFalse: MatchRuleAttr = MatchRuleAttr.get(self.context, [0], 1)
            mappings: ArrayAttr = ArrayAttr.get([mapTrue, mapFalse])
            predicate: rvsdg.MatchOp = rvsdg.MatchOp(matchTy, condition, mappings, ip=InsertionPoint(self.block))

            ifTy = [QuantumQubitType.get(self.context, 1)] * len(qubits)
            inputs = [self.visitQuantumBit(q) for q in qubits]
            ifOp: rvsdg.GammaNode = rvsdg.GammaNode(ifTy, predicate, inputs, 2, ip=InsertionPoint(self.block))
            inputTypes = [inp.type for inp in inputs]

            thenBlock: Block = ifOp.regions[0].blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))
            then_scope: Scope = Scope.from_scope(self.scope)
            for q, v in zip(qubits, iter_block_args(thenBlock.arguments)):
                then_scope.update_qubit(q, v)
            thenVisitor = QASMToMLIRVisitor.fromParent(self, block=thenBlock, scope=then_scope)
            thenVisitor.visitCircuit(true_body)
            rvsdg.YieldOp([thenVisitor.visitQuantumBit(q) for q in qubits], ip=InsertionPoint(thenBlock))

            elseBlock: Block = ifOp.regions[1].blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))
            else_scope: Scope = Scope.from_scope(self.scope)
            for q, v in zip(qubits, iter_block_args(elseBlock.arguments)):
                else_scope.update_qubit(q, v)
            elseVisitor = QASMToMLIRVisitor.fromParent(self, block=elseBlock, scope=else_scope)
            if hasElse:
                # Visit the else circuit only if it exists
                elseVisitor.visitCircuit(false_body)

            rvsdg.YieldOp([elseVisitor.visitQuantumBit(q) for q in qubits], ip=InsertionPoint(elseBlock))

            # Update qubits with returned values
            for inq, outv in zip(qubits, ifOp.outputs):
                self.scope.update_qubit(inq, outv)

    def _visitIfElseCondition(
        self,
        condition: Expr | tuple[ClassicalRegister, int] | tuple[Clbit, int],
        qubits: list[QubitSpecifier],
        clbits: list[ClbitSpecifier],
    ) -> Value:
        match condition:
            case Expr():
                raise NotImplementedError(f" IfElseOp with condition of type {type(condition)}")
            case (bitOrRegister, axiom):  # tuple[ClassicalRegister, int] | tuple[Clbit, int]
                with self.loc:
                    match bitOrRegister:
                        case Clbit():
                            raise NotImplementedError(f"IfElseOp condition on classical bit of type {type(bitOrRegister)}")
                        case ClassicalRegister():
                            # Create a DenseTensor from the axiom to compare against
                            i1 = IntegerType.get_signless(1)
                            tensor_ty = RankedTensorType.get([len(bitOrRegister)], i1)
                            if axiom == 0:
                                elem = IntegerAttr.get(i1, 0)
                                attr = DenseElementsAttr.get_splat(shaped_type=tensor_ty, element_attr=elem)
                            else:
                                bits = int_to_bits(axiom, len(bitOrRegister))
                                # buf = memoryview(bytes(bits))
                                attrs = [IntegerAttr.get(i1, b) for b in bits]
                                attr = DenseIntElementsAttr.get(attrs=attrs, type=tensor_ty, context=self.context)  # type: ignore
                                # attr = DenseElementsAttr.get(buf, type=i1, context=self.context) # type: ignore
                                # , shape=[size])

                            axiom_tensor = arith.constant(tensor_ty, attr, loc=self.loc, ip=InsertionPoint(self.block))
                            # Compare the axiom tensor with the register tensor
                            creg = self.visitClassicalRegister(bitOrRegister)
                            cmp = arith.cmpi(CmpIPredicate.eq, creg, axiom_tensor, loc=self.loc, ip=InsertionPoint(self.block))
                            match = vector.reduction(i1, CombiningKind.AND, cmp, loc=self.loc, ip=InsertionPoint(self.block))
                            return match
                        case _:
                            raise NotImplementedError(f"IfElseOp with condition of type {type(condition)}")


def qasm_version(code: str) -> QASMVersion:
    match = re.search(r"OPENQASM\s+(\d+)\.(\d+);", code)

    if match:
        major = int(match.group(1))
        minor = int(match.group(2))

        if major == 2 and minor == 0:
            return QASMVersion.QASM_2_0
        elif major == 3 and minor == 0:
            return QASMVersion.QASM_3_0

    return QASMVersion.Unspecified


def QASMToMLIR(code: str, emitResults: bool) -> Module:
    compat: QASMVersion = qasm_version(code)
    circuit: QuantumCircuit

    match compat:
        case QASMVersion.QASM_2_0:
            try:
                circuit = qasm2_loads(code, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
            except Exception as e:
                raise ConversionError(f"QASM2 parse failed: {e}")
        case QASMVersion.QASM_3_0:
            raise NotImplementedError("QASM3 not implemented")
            # circuit = qasm3_loads(code)
        case QASMVersion.Unspecified:
            raise ParseError("No version string found")

    context: Context = Context()
    # context.allow_unregistered_dialects = True
    quantum_dialect.register_dialect(context)
    rvsdg_dialect.register_dialect(context)
    qpu_dialect.register_dialect(context)

    with context, Location.unknown() as location:
        # Module representing the compilation unit
        module: Module = Module.create()
        # Create wrapping qpu.module op
        device_name = StringAttr.get("qpu")
        device: qpu.QPUModuleOp = qpu.QPUModuleOp(device_name)
        device.bodyRegion.blocks.append()
        module.body.append(device)

        circuit_name = StringAttr.get("main")
        empty_functy = FunctionType.get([], [])
        qpu_main: qpu.CircuitOp = qpu.CircuitOp(circuit_name, TypeAttr.get(empty_functy))
        qpu_main.body.blocks.append()

        scope: Scope = Scope()
        visitor: QASMToMLIRVisitor = QASMToMLIRVisitor(compat, context, device, location, qpu_main.body.blocks[0], scope)
        visitor.visitCircuit(circuit, emitRegisters=True)

        for qubit in circuit.qubits:
            qubitValue: Value = visitor.visitQuantumBit(qubit)
            quantum.DeallocateOp(qubitValue, loc=visitor.loc, ip=InsertionPoint(visitor.block))

        if not emitResults:
            qpu.ReturnOp([], ip=InsertionPoint(qpu_main.body.blocks[0]))
        else:
            # Create a new main function with the correct type and move the body to it
            m: list[Value] = [r for _, r in scope.cregs.items() if r is not None]
            resType: RankedTensorType = RankedTensorType.get([len(m)], IntegerType.get_signless(1))
            qpu_main.attributes["function_type"] = TypeAttr.get(func.FunctionType.get([], [resType]))
            # Merge all measurements into a tensor and return it
            res: Value = tensor.FromElementsOp(resType, m, ip=InsertionPoint(qpu_main.body.blocks[0])).result
            qpu.ReturnOp([res], ip=InsertionPoint(qpu_main.body.blocks[0]))

        device.bodyRegion.blocks[0].append(qpu_main)

        # Add main function
        res_ty = qpu_main.function_type.value.results
        qasm_main: func.FuncOp = func.FuncOp("qasm_main", ([], res_ty), visibility="public")
        qasm_main.add_entry_block()
        module.body.append(qasm_main)

        # ExecuteOp requires the construction of a default return value
        # Quantum code normally returns tensor<Nxi1> or i1
        exec_res = []
        for ty in res_ty:
            if isinstance(ty, RankedTensorType):
                empty = tensor.EmptyOp(ty.shape, ty.element_type, ip=InsertionPoint(qasm_main.entry_block))
                exec_res.append(empty)
            else:
                raise ParseError("Expected circuit to return RankedTensorType, found %s", ty)
        circuit_ref = SymbolRefAttr.get([device_name.value, circuit_name.value])
        qpu.ExecuteOp(circuit_ref, [], exec_res, ip=InsertionPoint(qasm_main.entry_block))
        func.ReturnOp(exec_res, ip=InsertionPoint(qasm_main.entry_block))

        return module


def bit_at(n: int, i: int) -> int:
    """Return the value (0 or 1) of bit *i* of n."""
    return (n >> i) & 1


def int_to_bits(x: int, n: int) -> list[int]:
    return [(x >> i) & 1 for i in range(n)]


def index_const(v: int, *, context: Context, loc: Location, ip: InsertionPoint) -> Value:
    return arith.ConstantOp(
        IndexType.get(context),
        v,
        loc=loc,
        ip=ip,
    ).result


def iter_block_args(bal: BlockArgumentList):
    for i in range(len(bal)):
        yield bal[i]
