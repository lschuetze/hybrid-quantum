//===- QuantumCircuitMetricsRegistration.cpp - Quantum circuit metrics ----===//
//
// Registers the Quantum circuit metrics translation
//
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Dialect/QPU/IR/QPUBase.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Target/circuit-metrics/QuantumCircuitMetrics.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/Translation.h>

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Quantum to Circuit Metrics registration
//===----------------------------------------------------------------------===//

void mlir::quantum::registerQuantumCircuitMetrics()
{
    TranslateFromMLIRRegistration registration(
        "print-metrics",
        "Translate Quantum dialect to Circuit Metrics",
        [](Operation* op, raw_ostream &os) -> LogicalResult {
            return quantum::QuantumCircuitMetrics(op, os);
        },
        [](DialectRegistry &registry) {
            registry.insert<quantum::QuantumDialect>();
            registry.insert<arith::ArithDialect>();
            registry.insert<tensor::TensorDialect>();
            registry.insert<scf::SCFDialect>();
            registry.insert<rvsdg::RVSDGDialect>();
            registry.insert<qpu::QPUDialect>();
            registry.insert<func::FuncDialect>();
        });
}
