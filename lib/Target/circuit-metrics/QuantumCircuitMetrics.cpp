//===- QuantumCircuitMetrics.cpp - Quantum to Circuit Metrics -------------===//
//
// Translate Quantum dialect ops circuit metrics.
//
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Target/circuit-metrics/QuantumCircuitMetrics.h"

#include "quantum-mlir/Dialect/QPU/IR/QPU.h"
#include "quantum-mlir/Dialect/QPU/IR/QPUOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include <cstddef>
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
using namespace mlir::quantum;

using llvm::formatv;

namespace {

struct CircuitMetrics {
    uint single = 0;
    uint two = 0;
    uint other = 0;
};

static LogicalResult walk(raw_ostream &os, qpu::CircuitOp op)
{
    CircuitMetrics metrics;

    auto walk = op->walk<WalkOrder::PreOrder>([&](Operation* child) {
        // quantum::unitaries (includes hermitian) represent gates on qubits
        if (child->hasTrait<Hermitian>() || child->hasTrait<Unitary>()) {
            size_t qubitArgs = llvm::count_if(
                child->getOperandTypes(),
                [](Type ty) { return llvm::isa<QubitType>(ty); });
            auto opTyIt = llvm::find_if(child->getOperandTypes(), [](Type ty) {
                return llvm::isa<QubitType>(ty);
            });
            if (opTyIt != child->getOperandTypes().end()) {
                auto qty = llvm::dyn_cast_or_null<QubitType>(*opTyIt);
                if (qty && qubitArgs == 1) metrics.single += qty.getSize();
                if (qty && qubitArgs == 2) metrics.two += qty.getSize();
            }
        } else {
            metrics.other++;
        }
        return WalkResult::advance();
    });
    // os.indent(2);
    // os << formatv("{0} single qubit ops\n", metrics.single);
    // os.indent(2);
    // os << formatv("{0} double qubit ops\n", metrics.two);
    // os.indent(2);
    // os << formatv("{0} other ops\n", metrics.other);
    os << formatv(
        "{0};{1};{2};{3}\r\n",
        op.getSymName(),
        metrics.single,
        metrics.two,
        metrics.other);
    return failure(walk.wasInterrupted());
}

} // namespace

LogicalResult quantum::QuantumCircuitMetrics(Operation* op, raw_ostream &os)
{
    // os << formatv("{0};{1};{2};{3}\r\n", "circuit", "single", "two",
    // "other");
    auto result = op->walk<WalkOrder::PreOrder>(
        [&](qpu::CircuitOp circuit) -> WalkResult {
            // os << formatv(
            //     "Print metrics for qpu.circuit: {0}\n",
            //     circuit.getSymName());
            return walk(os, circuit);
            // return WalkResult::advance();
        });
    if (result.wasInterrupted()) os << "Error";
    return failure(result.wasInterrupted());
}
