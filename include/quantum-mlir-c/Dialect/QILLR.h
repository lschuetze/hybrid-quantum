//===-- quantum-mlir-c/Dialect/QILLR.h - C API for QILLR dialect ---*- C-*-===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_C_DIALECT_QILLR_H
#define QUANTUM_MLIR_C_DIALECT_QILLR_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QILLR, qillr);

//===---------------------------------------------------------------------===//
// QubitType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a qillr::QubitType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAQubitType(MlirType type);

/// Creates an qillr.QubitType type.
MLIR_CAPI_EXPORTED MlirType mlirQubitTypeGet(MlirContext ctx, int64_t size);

//===---------------------------------------------------------------------===//
// ResultType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a qillr::ResultType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAResultType(MlirType type);

/// Creates an qillr.QubitType type.
MLIR_CAPI_EXPORTED MlirType mlirResultTypeGet(MlirContext ctx, int64_t size);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_MLIR_C_DIALECT_QILLR_H
