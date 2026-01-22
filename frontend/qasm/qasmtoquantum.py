"""
#   Frontend generating Quantum dialect code from QASM2 and QASM3 code.
#
# @author  Lars Schütze (lars.schuetze@tu-dresden.de)
"""

from __future__ import annotations

import re
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


class Scope:
    def __init__(self, visited: dict[str, quantum.GateOp] | None = None) -> None:
        # Maps circuit quantum registers of size N to quantum.alloc<N> values
        self.qregs: dict[QuantumRegister, Value | None] = {}
        # Maps circuit qubit to current quantum.qubit value
        self.qubits: dict[Qubit, Value | None] = {}
        # Maps circuit classical registers of size N to tensor<1xN> values
        self.cregs: dict[ClassicalRegister, Value | None] = {}
        # Maps circuit classical bit to current quantum.qubit value
        self.clbits: dict[Clbit, Value | None] = {}
        # Holds already visited gates that can be reused and have not to be revisited.
        self.visitedGates: dict[str, quantum.GateOp] = visited if visited is not None else {}

    @classmethod
    def fromList(
        cls,
        qregs: list[QuantumRegister],
        cregs: list[ClassicalRegister],
        visited: dict[str, quantum.GateOp] | None = None,
    ) -> Scope:
        s = cls(visited)
        s.qregs = {qreg: None for qreg in qregs}
        s.cregs = {creg: None for creg in cregs}
        return s

    @classmethod
    def fromMap(
        cls,
        qregs: dict[QuantumRegister, Value | None],
        qubits: dict[Qubit, Value | None],
        cregs: dict[ClassicalRegister, Value | None],
        clbits: dict[Clbit, Value | None],
        visited: dict[str, quantum.GateOp] | None = None,
    ) -> Scope:
        s = cls(visited)
        s.qregs = qregs
        s.qubits = qubits
        s.cregs = cregs
        s.clbits = clbits
        return s

    def findAlloc(self, q: QubitSpecifier) -> Value | None:
        if isinstance(q, QuantumRegister):
            return self.qregs.get(q)
        if isinstance(q, Qubit):
            return self.qubits.get(q)

    def setQubit(self, q: QubitSpecifier, alloc: Value) -> Value:
        assert not isinstance(alloc, list)
        if isinstance(q, QuantumRegister):
            self.qregs[q] = alloc
        if isinstance(q, Qubit):
            self.qubits[q] = alloc

        return alloc

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
        return self.visitedGates.get(str(gate.name))

    def setGate(self, gate: QASM2_Gate, newGate: quantum.GateOp) -> None:
        self.visitedGates[str(gate.name)] = newGate


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
        alloc: Value | None = self.scope.findAlloc(reg)
        if alloc is None:
            size: int = len(reg)
            assert size >= 1
            qubitTy: QuantumQubitType = QuantumQubitType.get(self.context, size)
            qalloc: quantum.AllocOp = quantum.AllocOp(qubitTy, loc=self.loc, ip=InsertionPoint(self.block))
            return self.scope.setQubit(reg, qalloc.result)

        return alloc

    def visitQuantumBit(self, q: Qubit) -> Value:
        assert isinstance(q, Qubit)
        val: Value | None = self.scope.findAlloc(q)
        if val is None:
            # A qubit value has not been allocated yet.
            # Check if the qubit register exists.
            # TODO: Check what happens when we parse a Qiskit circuit (without QASM)
            reg: Value | None = self.scope.findAlloc(q._register)
            if reg is None:
                raise ParseError(f"Register {q._register} for {q} not defined.")

            # If the register has length 1 the qubit and register are identical values
            if len(q._register) == q._index + 1:
                return self.scope.setQubit(q, reg)
            else:
                # We have to split the qubit value from the register.
                match q._index:
                    case 0:
                        qubitLTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
                        qubitRTy: QuantumQubitType = QuantumQubitType.get(self.context, len(q._register) - 1)
                        split: quantum.SplitOp = quantum.SplitOp(
                            [qubitLTy, qubitRTy], reg, loc=self.loc, ip=InsertionPoint(self.block)
                        )
                        return self.scope.setQubit(q, split.result)
                    case _:
                        raise ParseError(f"Index > 0 for {q} of type {type(q)} not implemented yet.")
        else:
            return val

    def visitClassicalRegister(self, creg: ClassicalRegister) -> Value:
        assert isinstance(creg, ClassicalRegister)
        calloc: Value | None = self.scope.findResult(creg)
        if calloc is None:
            i1 = IntegerType.get_signless(1)
            tensor_ty = RankedTensorType.get([4], i1)
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
                        self.scope.setQubit(q, r)
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
                    self.scope.setQubit(qubits[0], ccx.control1_out)
                    self.scope.setQubit(qubits[1], ccx.control2_out)
                    self.scope.setQubit(qubits[2], ccx.target_out)
            case lib.CSwapGate(), 3:
                with self.loc:
                    control: Value = self.visitQuantumBit(qubits[0])
                    lhs: Value = self.visitQuantumBit(qubits[1])
                    rhs: Value = self.visitQuantumBit(qubits[2])
                    cswap: quantum.CSWAPOp = quantum.CSWAPOp(control, lhs, rhs, ip=InsertionPoint(self.block))
                    self.scope.setQubit(qubits[0], cswap.control_out)
                    self.scope.setQubit(qubits[1], cswap.lhs_out)
                    self.scope.setQubit(qubits[2], cswap.rhs_out)
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
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.YGate():
                            op: quantum.YOp = quantum.YOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.ZGate():
                            op: quantum.ZOp = quantum.ZOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.HGate():
                            op: quantum.HOp = quantum.HOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.SGate():
                            op: quantum.SOp = quantum.SOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.SXGate():
                            op: quantum.SXOp = quantum.SXOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.SdgGate():
                            op: quantum.SdgOp = quantum.SdgOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.TGate():
                            op: quantum.TOp = quantum.TOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.TdgGate():
                            op: quantum.TdgOp = quantum.TdgOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.RZGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RzOp = quantum.RzOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.RXGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RxOp = quantum.RxOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.RYGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.RyOp = quantum.RyOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.U3Gate():
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U3Op = quantum.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.U2Gate():
                            phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U2Op = quantum.U2Op(target, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.U1Gate():
                            lam = self.visitClassic(instr.params[0])
                            op: quantum.U1Op = quantum.U1Op(target, lam, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.UGate():
                            # TODO: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.UGate
                            theta, phi, lam = [self.visitClassic(param) for param in instr.params]
                            op: quantum.U3Op = quantum.U3Op(target, theta, phi, lam, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.Reset():
                            outTy = QuantumQubitType.get(self.context, 1)
                            op: quantum.ResetOp = quantum.ResetOp(outTy, target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.Measure():
                            assert len(clbits) == 1
                            self._visitMeasure(instr, target, qubits[0], clbits[0])
                        case lib.IGate():
                            op: quantum.IdOp = quantum.IdOp(target, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
                        case lib.PhaseGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.PhaseOp = quantum.PhaseOp(target, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.result)
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
                            self.scope.setQubit(qubits[0], op.result_lhs)
                            self.scope.setQubit(qubits[1], op.result_rhs)
                        case lib.CZGate():
                            op: quantum.CZOp = quantum.CZOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.control_out)
                            self.scope.setQubit(qubits[1], op.target_out)
                        case lib.CXGate():
                            op: quantum.CNOTOp = quantum.CNOTOp(lhs, rhs, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.control_out)
                            self.scope.setQubit(qubits[1], op.target_out)
                        case lib.CRYGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CRyOp = quantum.CRyOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.control_out)
                            self.scope.setQubit(qubits[1], op.target_out)
                        case lib.CRZGate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CRzOp = quantum.CRzOp(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.control_out)
                            self.scope.setQubit(qubits[1], op.target_out)
                        case lib.CU1Gate():
                            angle = self.visitClassic(instr.params[0])
                            op: quantum.CU1Op = quantum.CU1Op(lhs, rhs, angle, ip=InsertionPoint(self.block))
                            self.scope.setQubit(qubits[0], op.control_out)
                            self.scope.setQubit(qubits[1], op.target_out)

    def _visitMeasure(self, instr: lib.Measure, target: Value, qubit: QubitSpecifier, clbit: ClbitSpecifier) -> None:
        assert isinstance(clbit, Clbit)
        assert isinstance(qubit, Qubit)
        # Create the measurement
        measurementTy: QuantumMeasurementType = QuantumMeasurementType.get(self.context, 1)
        qubitTy: QuantumQubitType = QuantumQubitType.get(self.context, 1)
        op: quantum.MeasureOp = quantum.MeasureOp(measurementTy, qubitTy, target, ip=InsertionPoint(self.block))
        self.scope.setQubit(qubit, op.result)
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
        self.scope.setResult(creg, creg_tensor)

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
                gateQubits = {q: v for q, v in zip(circuit.qubits, gate.body.blocks[0].arguments)}
                innerGateScope: Scope = Scope.fromMap(self.scope.qregs, gateQubits, {}, {}, self.scope.visitedGates)
                visitor: QASMToMLIRVisitor = QASMToMLIRVisitor.fromParent(self, block=gateBody, scope=innerGateScope)
                visitor.visitCircuit(circuit)
                quantum.ReturnOp([visitor.visitQuantumBit(q) for q in gateQubits], loc=self.loc, ip=InsertionPoint(gateBody))
            # Construct quantum.CallOp for defined custom gate
            # TODO: qpu.circuit instead of GateOp
            callee: StringAttr = instr.name
            operands: list[Value] = [self.visitQuantumBit(q) for q in qubits]
            outTys: list[Type] = [o.type for o in operands]
            op: quantum.GateCallOp = quantum.GateCallOp(outTys, callee, operands, loc=self.loc, ip=InsertionPoint(self.block))
            for inq, outq in zip(qubits, op.results):
                self.scope.setQubit(inq, outq)
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
            thenScope: Scope = Scope.fromMap(
                qregs=self.scope.qregs,
                qubits={q: v for q, v in zip(qubits, iter_block_args(thenBlock.arguments))},
                cregs=self.scope.cregs,
                clbits=self.scope.clbits,
                visited=self.scope.visitedGates,
            )
            thenVisitor = QASMToMLIRVisitor.fromParent(self, block=thenBlock, scope=thenScope)
            thenVisitor.visitCircuit(true_body)
            rvsdg.YieldOp([thenVisitor.visitQuantumBit(q) for q in qubits], ip=InsertionPoint(thenBlock))

            elseBlock: Block = ifOp.regions[1].blocks.append(*inputTypes, arg_locs=[self.loc] * len(inputTypes))
            elseScope: Scope = Scope.fromMap(
                qregs=self.scope.qregs,
                qubits={q: v for q, v in zip(qubits, iter_block_args(elseBlock.arguments))},
                cregs=self.scope.cregs,
                clbits=self.scope.clbits,
                visited=self.scope.visitedGates,
            )
            elseVisitor = QASMToMLIRVisitor.fromParent(self, block=elseBlock, scope=elseScope)
            if hasElse:
                # Visit the else circuit only if it exists
                elseVisitor.visitCircuit(false_body)

            rvsdg.YieldOp([elseVisitor.visitQuantumBit(q) for q in qubits], ip=InsertionPoint(elseBlock))

            # Update qubits with returned values
            for inq, outv in zip(qubits, ifOp.outputs):
                self.scope.setQubit(inq, outv)

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
                            buf = memoryview(bytes(int_to_bits(axiom, len(bitOrRegister))))
                            attr = DenseIntElementsAttr.get(buf, type=tensor_ty)  # type: ignore
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

        scope: Scope = Scope.fromList(circuit.qregs, circuit.cregs)
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
