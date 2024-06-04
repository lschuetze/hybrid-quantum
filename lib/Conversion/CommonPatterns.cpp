#include "cinm-mlir/Conversion/CommonPatterns.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace mlir {

SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<int64_t> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgsInit,
                                              BodyBuilderCallback bodyBuilder) {
  assert(loopSizes.size() == loopSteps.size());

  SmallVector<affine::AffineForOp> loops;
  SmallVector<Value> indices;
  ValueRange iterArgs = iterArgsInit;

  for (auto [size, step] : llvm::zip(loopSizes, loopSteps)) {
    affine::AffineForOp current =
        builder.create<affine::AffineForOp>(loc, 0, size, step, iterArgs);
    if (!loops.empty()) {
      builder.create<affine::AffineYieldOp>(loc, current.getResults());
    }
    loops.push_back(current);
    indices.push_back(current.getRegion().front().getArguments().front());
    iterArgs = current.getRegion().front().getArguments().drop_front();
    builder.setInsertionPointToStart(&current.getRegion().front());
  }

  builder.create<affine::AffineYieldOp>(
      loc, bodyBuilder(builder, loc, indices, iterArgs));

  builder.setInsertionPointAfter(loops.front());
  return loops.front().getResults();
}

ConvertCnmSetZeroToAffine::ConvertCnmSetZeroToAffine(MLIRContext *context,
                                                     PatternBenefit benefit)
    : OpConversionPattern<cnm::SetZeroOp>(context, benefit) {}

LogicalResult ConvertCnmSetZeroToAffine::matchAndRewrite(
    cnm::SetZeroOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const {
  const Value dst = rewriter.getRemappedValue(op.getOperand());

  const MemRefType type = dst.getType().cast<MemRefType>();
  const SmallVector<int64_t> loopSizes{type.getShape()};
  const SmallVector<int64_t> loopSteps(loopSizes.size(), 1);

  createNestedAffineForLoops(
      rewriter, op.getLoc(), loopSizes, loopSteps, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange) -> SmallVector<Value> {
        const Value zero = builder.create<arith::ConstantOp>(
            loc, builder.getZeroAttr(op.getType().getElementType()));
        rewriter.create<memref::StoreOp>(loc, zero, dst, indices);
        return {};
      });

  rewriter.replaceOp(op, {dst});
  return success();
}

} // namespace mlir
