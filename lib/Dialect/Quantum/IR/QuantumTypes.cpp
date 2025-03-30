/// Implements the Quantum dialect types.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "quantum-types"

using namespace mlir;
using namespace mlir::quantum;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"

//===----------------------------------------------------------------------===//

bool QubitType::isSingleQubit() const { return getSize() == 1; }

LogicalResult QubitType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    int64_t size)
{
    if (size < 1)
        return emitError() << "expected integer value greater equals 1";

    return success();
}

//===----------------------------------------------------------------------===//
// QuantumDialect
//===----------------------------------------------------------------------===//

void QuantumDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.cpp.inc"
        >();
}
