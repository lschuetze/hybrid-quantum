#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTRVSDGTOSCF
#include "quantum-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace rvsdg {

void populateConvertRVSDGToScfPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);
} // namespace rvsdg

std::unique_ptr<Pass> createConvertRVSDGToScfPass();

} // namespace mlir
