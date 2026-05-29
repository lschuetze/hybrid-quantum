//===- QuantumCircuitMetrics.h - Quantum Metrics --------------------------===//
//
// A translator that counts Quantum metrics.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Value.h"

namespace mlir {
namespace quantum {

LogicalResult QuantumCircuitMetrics(Operation* op, raw_ostream &os);

void registerQuantumCircuitMetrics();
} // namespace quantum
} // namespace mlir
