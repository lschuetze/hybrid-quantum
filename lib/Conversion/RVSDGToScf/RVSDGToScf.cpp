/// Implements the RVSDG transformation to SCF dialect.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/RVSDGToScf/RVSDGToScf.h"

#include "mlir/Pass/Pass.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGAttributes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGTypes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace mlir::rvsdg;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTRVSDGTOSCF
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertRVSDGToScfPass
        : mlir::impl::ConvertRVSDGToScfBase<ConvertRVSDGToScfPass> {
    using ConvertRVSDGToScfBase::ConvertRVSDGToScfBase;

    void runOnOperation() override;
};

struct ConvertMatch : public OpConversionPattern<rvsdg::MatchOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::MatchOp op,
        rvsdg::MatchOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Support more than boolean 0 -> 1, 1 -> 0 mapping
        if (adaptor.getInput().getType() != rewriter.getI1Type())
            return failure();

        rewriter.replaceAllUsesWith(op.getResult(), adaptor.getInput());
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertGamma : public OpConversionPattern<rvsdg::GammaNode> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::GammaNode op,
        rvsdg::GammaNodeAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // TODO: We can have more than 2 regions (switch statement)
        if (op->getNumRegions() != 2) return failure();

        auto newIf = rewriter.create<scf::IfOp>(
            op->getLoc(),
            op->getResultTypes(),
            adaptor.getPredicate(),
            /* addThenBlock */ true,
            /* addElseBlock */ true);

        for (auto &&[oldRegion, newRegion] :
             llvm::zip(op->getRegions(), newIf->getRegions())) {
            Block* oldBlock = &oldRegion.front();
            Block* newBlock = &newRegion.front();
            rewriter.inlineBlockBefore(
                oldBlock,
                newBlock,
                newBlock->end(),
                adaptor.getInputs());
        }

        rewriter.replaceOp(op, newIf);
        return success();
    }
};

struct ConvertYield : public OpConversionPattern<rvsdg::YieldOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        rvsdg::YieldOp op,
        rvsdg::YieldOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
        return success();
    }
};

} // namespace

void ConvertRVSDGToScfPass::runOnOperation()
{
    auto context = &getContext();
    TypeConverter converter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    converter.addConversion([](Type type) { return type; });

    target.addIllegalDialect<rvsdg::RVSDGDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    populateConvertRVSDGToScfPatterns(converter, patterns);

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        signalPassFailure();
}

void mlir::rvsdg::populateConvertRVSDGToScfPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertGamma, ConvertYield, ConvertMatch>(
        typeConverter,
        patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertRVSDGToScfPass()
{
    return std::make_unique<ConvertRVSDGToScfPass>();
}
