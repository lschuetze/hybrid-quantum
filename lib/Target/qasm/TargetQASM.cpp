//===- TargetQASM.cpp - QILLR to OpenQASM Translation ---------------------===//
//
// Translate QILLR dialect ops into OpenQASM 2.0.
//
/// @file
/// @author     Washim Neupane (washim.neupane@outlook.com)
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Target/qasm/TargetQASM.h"

#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"
#include "quantum-mlir/Dialect/QPU/IR/QPU.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.h"

#include <atomic>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir-c/Diagnostics.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::qillr;

using llvm::formatv;

namespace {
class QASMEmitter {
public:
    QASMEmitter(raw_ostream &o) : os(o), names() {}

    LogicalResult emitOperation(Operation &op);

    /// Return the mapped qubit register name or new name for value.
    std::string getOrCreateQubitRegisterName(Value value);

    /// Return the mapped classical register name or new name for value.
    std::string getOrCreateClassicalRegisterName(Value value);

    /// Returns the output stream.
    raw_ostream &ostream() { return os; };

private:
    raw_ostream &os;
    std::atomic_int nextQubitId{0};
    std::atomic_int nextRegisterId{0};
    llvm::DenseMap<Value, std::string> names;
};
} // namespace

static std::string printOperand(Value v)
{
    auto c = v.getDefiningOp<arith::ConstantOp>();
    auto result = mlir::cast<FloatAttr>(c.getValue());
    if (result)
        return std::to_string(result.getValueAsDouble());
    else
        emitError(v.getLoc(), "Defining op not a constant.");

    return "";
}

/// Emit the QASM header
static void printHeader(QASMEmitter &emitter)
{
    raw_ostream &os = emitter.ostream();
    os << "OPENQASM 2.0;\n"
          "include \"qelib1.inc\";\n\n";
}

static LogicalResult printQubitAlloc(QASMEmitter &emitter, AllocOp op)
{
    std::string name = emitter.getOrCreateQubitRegisterName(op.getResult());
    int64_t size = op.getSize();
    raw_ostream &os = emitter.ostream();
    os << "qreg " << name << "[" << size << "];\n";
    return success();
}

static LogicalResult printResultAlloc(QASMEmitter &emitter, AllocResultOp op)
{
    std::string name = emitter.getOrCreateClassicalRegisterName(op.getResult());
    int64_t size = op.getSize();
    raw_ostream &os = emitter.ostream();
    os << "creg " << name << "[" << size << "];\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printPrimitiveGate(QASMEmitter &emitter, OpTy op, std::string opName)
{
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateQubitRegisterName(op.getInput());
    std::optional<int64_t> index = op.getIndex();
    os << opName << " " << name << "[" << index << "]" << ";\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printRotationGate(QASMEmitter &emitter, OpTy op, std::string opName)
{
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateQubitRegisterName(op.getInput());
    std::optional<int64_t> index = op.getIndex();
    std::string theta = printOperand(op.getAngle());
    os << opName << "(" << theta << ") " << name << "[" << index << "];\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printControledGate(QASMEmitter &emitter, OpTy op, std::string opname)
{
    raw_ostream &os = emitter.ostream();
    std::string control = emitter.getOrCreateQubitRegisterName(op.getControl());
    std::optional<int64_t> ctrlIndex = op.getControlIndex();
    std::string target = emitter.getOrCreateQubitRegisterName(op.getTarget());
    std::optional<int64_t> targetIndex = op.getTargetIndex();
    os << opname << " " << control << "[" << ctrlIndex << "]" << ", " << target
       << "[" << targetIndex << "]" << ";\n";
    return success();
}

static LogicalResult printToffoli(QASMEmitter &emitter, CCXOp op)
{
    raw_ostream &os = emitter.ostream();
    std::string control1 =
        emitter.getOrCreateQubitRegisterName(op.getControl1());
    std::optional<int64_t> ctrl1Index = op.getControl1Index();

    std::string control2 =
        emitter.getOrCreateQubitRegisterName(op.getControl2());
    std::optional<int64_t> ctrl2Index = op.getControl2Index();

    std::string target = emitter.getOrCreateQubitRegisterName(op.getTarget());
    std::optional<int64_t> targetIndex = op.getTargetIndex();

    os << "ccx"
       << " " << control1 << "[" << ctrl1Index << "]" << ", " << control2 << "["
       << ctrl2Index << "]" << ", " << target << "[" << targetIndex << "]"
       << ";\n";
    return success();
}

static LogicalResult printSwap(QASMEmitter &emitter, SwapOp op)
{
    raw_ostream &os = emitter.ostream();
    std::string lhs = emitter.getOrCreateQubitRegisterName(op.getLhs());
    std::optional<int64_t> lhsIndex = op.getLhsIndex();
    std::string rhs = emitter.getOrCreateQubitRegisterName(op.getRhs());
    std::optional<int64_t> rhsIndex = op.getRhsIndex();

    os << "swap"
       << " " << lhs << "[" << lhsIndex << "]" << ", " << rhs << "[" << rhsIndex
       << "]" << ";\n";
    return success();
}

static LogicalResult printMeasure(QASMEmitter &emitter, MeasureOp op)
{
    raw_ostream &os = emitter.ostream();
    std::string input = emitter.getOrCreateQubitRegisterName(op.getInput());
    std::optional<int64_t> inputIndex = op.getInputIndex();

    std::string result =
        emitter.getOrCreateClassicalRegisterName(op.getResult());
    std::optional<int64_t> resultIndex = op.getResultIndex();

    os << "measure"
       << " " << input << "[" << inputIndex << "]" << " -> " << result << "["
       << resultIndex << "]" << ";\n";
    return success();
}

static LogicalResult printBarrier(QASMEmitter &emitter, BarrierOp op)
{
    raw_ostream &os = emitter.ostream();
    bool isFirst = true;

    os << "barrier ";
    for (auto &&[operand, index] :
         llvm::zip_equal(op->getOperands(), op.getIndices()->getValue())) {
        if (isFirst)
            isFirst = false;
        else
            os << ", ";
        int64_t i = llvm::cast<IntegerAttr>(index).getInt();
        os << emitter.getOrCreateQubitRegisterName(operand) << "[" << i << "]";
    }
    os << ";\n";
    return success();
}

template<typename OpTy>
static LogicalResult
printControledRotationGate(QASMEmitter &emitter, OpTy op, std::string opname)
{
    raw_ostream &os = emitter.ostream();
    std::string control = emitter.getOrCreateQubitRegisterName(op.getControl());
    std::optional<int64_t> ctrlIndex = op.getControlIndex();
    std::string target = emitter.getOrCreateQubitRegisterName(op.getTarget());
    std::optional<int64_t> targetIndex = op.getTargetIndex();
    std::string theta = printOperand(op.getAngle());

    os << opname << "(" << theta << ") " << control << "[" << ctrlIndex << "]"
       << ", " << target << "[" << targetIndex << "]" << ";\n";
    return success();
}

static LogicalResult printU3(QASMEmitter &emitter, U3Op op)
{
    std::string t = printOperand(op.getTheta());
    std::string p = printOperand(op.getPhi());
    std::string l = printOperand(op.getLambda());
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateQubitRegisterName(op.getInput());
    std::optional<int64_t> index = op.getIndex();

    os << "u3(" << t << "," << p << "," << l << ") " << name << "[" << index
       << "]" << ";\n";
    return success();
}

static LogicalResult printU2(QASMEmitter &emitter, U2Op op)
{
    std::string p = printOperand(op.getPhi());
    std::string l = printOperand(op.getLambda());
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateQubitRegisterName(op.getInput());
    std::optional<int64_t> index = op.getIndex();

    os << "u2(" << p << "," << l << ") " << name << "[" << index << "];\n";
    return success();
}

static LogicalResult printU1(QASMEmitter &emitter, U1Op op)
{
    std::string l = printOperand(op.getLambda());
    raw_ostream &os = emitter.ostream();
    std::string name = emitter.getOrCreateQubitRegisterName(op.getInput());
    std::optional<int64_t> index = op.getIndex();

    os << "u1(" << l << ") " << name << "[" << index << "];\n";
    return success();
}

static LogicalResult printCU1(QASMEmitter &emitter, CU1Op op)
{
    std::string l = printOperand(op.getAngle());
    raw_ostream &os = emitter.ostream();
    std::string control = emitter.getOrCreateQubitRegisterName(op.getControl());
    std::optional<int64_t> controlIndex = op.getControlIndex();
    std::string target = emitter.getOrCreateQubitRegisterName(op.getTarget());
    std::optional<int64_t> targetIndex = op.getTargetIndex();

    os << "cu1(" << l << ") " << control << "[" << controlIndex << "], "
       << target << "[" << targetIndex << "];\n";
    return success();
}

static LogicalResult printIfOp(QASMEmitter &emitter, scf::IfOp ifOp)
{
    raw_ostream &os = emitter.ostream();
    Operation* conditionOp = ifOp.getCondition().getDefiningOp();

    os << "if(";

    // TODO: Short path if condition is a constant bool
    std::string creg;
    std::string condition;

    // Case 1: tensor.extract on the condition and result tensor
    if (auto extract = llvm::dyn_cast<tensor::ExtractOp>(conditionOp)) {
        auto cmp = extract.getTensor().getDefiningOp();
        if (auto cmpOp = llvm::dyn_cast<arith::CmpIOp>(cmp)) {
            auto mt = llvm::dyn_cast<ReadMeasurementOp>(
                cmpOp.getLhs().getDefiningOp());
            creg = emitter.getOrCreateClassicalRegisterName(mt.getInput());

            os << creg;

            assert(
                cmpOp.getPredicate() == arith::CmpIPredicate::eq
                && "Current support limited to equals");
            os << "==";

            auto cst = llvm::dyn_cast<arith::ConstantOp>(
                cmpOp.getRhs().getDefiningOp());
            auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());

            APInt cond(denseAttr.getNumElements(), 0);
            unsigned bitIndex = 0;
            for (bool value : denseAttr.getValues<bool>())
                cond.setBitVal(bitIndex++, value);

            os << cond.getZExtValue() << ") ";
        }
    }
    return success();
}

std::string QASMEmitter::getOrCreateQubitRegisterName(Value value)
{
    if (names.contains(value)) return names[value];

    std::string name = "q" + std::to_string(nextQubitId++);
    names.insert({value, name});
    return name;
}

std::string QASMEmitter::getOrCreateClassicalRegisterName(Value value)
{
    if (names.contains(value)) return names[value];

    std::string name = "c" + std::to_string(nextRegisterId++);
    names.insert({value, name});
    return name;
}

LogicalResult QASMEmitter::emitOperation(Operation &op)
{
    LogicalResult result =
        TypeSwitch<Operation*, LogicalResult>(&op)
            // Memory allocations
            .Case<AllocOp>([&](AllocOp a) { return printQubitAlloc(*this, a); })
            .Case<AllocResultOp>(
                [&](AllocResultOp r) { return printResultAlloc(*this, r); })
            // Single-qubit gates
            .Case<HOp>(
                [&](HOp h) { return printPrimitiveGate<HOp>(*this, h, "h"); })
            .Case<XOp>(
                [&](XOp x) { return printPrimitiveGate<XOp>(*this, x, "x"); })
            .Case<YOp>(
                [&](YOp y) { return printPrimitiveGate<YOp>(*this, y, "y"); })
            .Case<ZOp>(
                [&](ZOp z) { return printPrimitiveGate<ZOp>(*this, z, "z"); })
            .Case<SOp>(
                [&](SOp s) { return printPrimitiveGate<SOp>(*this, s, "s"); })
            .Case<SdgOp>([&](SdgOp sdg) {
                return printPrimitiveGate<SdgOp>(*this, sdg, "sdg");
            })
            .Case<TOp>(
                [&](TOp t) { return printPrimitiveGate<TOp>(*this, t, "t"); })
            .Case<TdgOp>([&](TdgOp tdg) {
                return printPrimitiveGate<TdgOp>(*this, tdg, "tdg");
            })
            // controlled gates
            .Case<CNOTOp>([&](CNOTOp cx) {
                return printControledGate<CNOTOp>(*this, cx, "cx");
            })
            .Case<CZOp>([&](CZOp cz) {
                return printControledGate<CZOp>(*this, cz, "cz");
            })
            .Case<CCXOp>([&](CCXOp ccx) { return printToffoli(*this, ccx); })
            // Controled rotation gates
            .Case<CRzOp>([&](CRzOp crz) {
                return printControledRotationGate<CRzOp>(*this, crz, "crz");
            })
            .Case<CRyOp>([&](CRyOp cry) {
                return printControledRotationGate<CRyOp>(*this, cry, "cry");
            })
            // U1/U2/U3 gates
            .Case<U3Op>([&](U3Op u3) { return printU3(*this, u3); })
            .Case<U2Op>([&](U2Op u2) { return printU2(*this, u2); })
            .Case<U1Op>([&](U1Op u1) { return printU1(*this, u1); })
            // CU1 gate
            .Case<CU1Op>([&](CU1Op cu1) { return printCU1(*this, cu1); })
            // Rx/Ry/Rz gates
            .Case<RxOp>([&](RxOp rx) {
                return printRotationGate<RxOp>(*this, rx, "rx");
            })
            .Case<RyOp>([&](RyOp ry) {
                return printRotationGate<RyOp>(*this, ry, "ry");
            })
            .Case<RzOp>([&](RzOp rz) {
                return printRotationGate<RzOp>(*this, rz, "rz");
            })
            // Others
            .Case<BarrierOp>(
                [&](BarrierOp barrier) { return printBarrier(*this, barrier); })
            .Case<SwapOp>([&](SwapOp swap) { return printSwap(*this, swap); })
            .Case<MeasureOp>(
                [&](MeasureOp measure) { return printMeasure(*this, measure); })
            .Case<ResetOp>([&](ResetOp reset) {
                return printPrimitiveGate<ResetOp>(*this, reset, "reset");
            })
            // Structured Control Flow
            .Case<scf::IfOp>(
                [&](scf::IfOp ifOp) { return printIfOp(*this, ifOp); })
            // Ignored ops
            .Case<
                // QPU dialect
                qpu::CircuitOp,
                qpu::ReturnOp,
                // QILLR
                qillr::ReadMeasurementOp,
                qillr::DeallocateOp,
                // MLIR
                scf::YieldOp,
                arith::ConstantOp,
                arith::CmpIOp,
                tensor::ExtractOp>([](Operation*) { return success(); })
            // Default = error case
            .Default([](Operation* op) {
                emitError(
                    op->getLoc(),
                    llvm::formatv(
                        "No codegen for operation {}.",
                        op->getName()));
                return failure();
            });

    if (failed(result)) return failure();
    return result;
}

static LogicalResult walk(QASMEmitter &emitter, Operation* op)
{
    auto walk = op->walk<WalkOrder::PreOrder>([&](Operation* child) {
        auto status = emitter.emitOperation(*child);

        if (status.failed())
            emitError(
                child->getLoc(),
                "Interrupt of QILLR to QASM translation.");

        return WalkResult(status);
    });
    return failure(walk.wasInterrupted());
}

LogicalResult qillr::QILLRTranslateToQASM(Operation* op, raw_ostream &os)
{
    QASMEmitter emitter(os);
    printHeader(emitter);

    auto result = op->walk<WalkOrder::PreOrder>([&](qpu::CircuitOp circuit) {
        return WalkResult(walk(emitter, circuit));
    });
    return failure(result.wasInterrupted());
}
