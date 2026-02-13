#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Dialect/Quantum/Analysis/RegisterRangesAnalysis.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTQUANTUMTOQILLR
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace quantum {

void populateConvertQuantumToQILLRPatterns(
    mlir::DataFlowSolver &solver,
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

} // namespace quantum

std::unique_ptr<Pass> createConvertQuantumToQILLRPass();

} // namespace mlir
