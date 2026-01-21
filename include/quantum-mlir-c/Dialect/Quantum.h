//=== quantum-mlir-c/Dialect/Quantum.h - C API for Quantum dialect -*- C-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_C_DIALECT_QUANTUM_H
#define QUANTUM_MLIR_C_DIALECT_QUANTUM_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Quantum, quantum);

//===---------------------------------------------------------------------===//
// QubitType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a quantum::QubitType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAQuantumQubitType(MlirType type);

/// Creates an quantum.QubitType type.
MLIR_CAPI_EXPORTED MlirType
mlirQuantumQubitTypeGet(MlirContext ctx, int64_t length);

//===---------------------------------------------------------------------===//
// MeasurementType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a quantum::MeasurementType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAQuantumMeasurementType(MlirType type);

/// Creates an quantum.MeasurementType type.
MLIR_CAPI_EXPORTED MlirType
mlirQuantumMeasurementTypeGet(MlirContext ctx, int64_t length);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_MLIR_C_DIALECT_QUANTUM_H
