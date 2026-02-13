/// Implements the ConvertQuantumToQILLRPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "quantum-mlir/Conversion/QuantumToQILLR/QuantumToQILLR.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "quantum-mlir/Conversion/RVSDGConversion/RVSDGConversion.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLR.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLROps.h"
#include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.h"
#include "quantum-mlir/Dialect/Quantum/Analysis/RegisterRangesAnalysis.h"
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/OneToNFuncConversions.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
#include <utility>

using namespace mlir;
using namespace mlir::quantum;
using namespace quantum::dataflow;

//===- Generated includes
//-------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQILLR
#include "quantum-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

namespace {

struct ConvertQuantumToQILLRPass
        : mlir::impl::ConvertQuantumToQILLRBase<ConvertQuantumToQILLRPass> {
    using ConvertQuantumToQILLRBase::ConvertQuantumToQILLRBase;

    void runOnOperation() override;
};

template<typename OpT>
struct IndexTrackingOpConversionPattern : public OpConversionPattern<OpT> {
public:
    IndexTrackingOpConversionPattern(
        mlir::DataFlowSolver &solver,
        const TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit = 1)
            : OpConversionPattern<OpT>(typeConverter, context, benefit),
              solver(solver)
    {}

    LogicalResult lookupSingle(
        Value value,
        std::optional<uint64_t> &result,
        ConversionPatternRewriter &rewriter) const;

protected:
    mlir::DataFlowSolver &solver;

}; // struct IndexTrackingOpConversionPattern

template<typename OpT>
LogicalResult IndexTrackingOpConversionPattern<OpT>::lookupSingle(
    Value value,
    std::optional<uint64_t> &result,
    ConversionPatternRewriter &rewriter) const
{
    const auto* lattice =
        this->solver.lookupState<RegisterRangesLattice>(value);
    if (!lattice)
        return rewriter.notifyMatchFailure(
            value.getDefiningOp(),
            "Lattice value not found");

    const auto indices = lattice->getValue().getValue();
    if (indices.size() > 1)
        return rewriter.notifyMatchFailure(
            value.getDefiningOp(),
            "Expected singe inferred value, got multiple.");

    auto index = indices.front().getRanges().getConstantValue();
    result =
        index ? std::optional<uint64_t>(index->getSExtValue()) : std::nullopt;
    return success();
}

struct ConvertAlloc
        : public IndexTrackingOpConversionPattern<quantum::AllocOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto opType = op.getResult().getType();
        auto resultSize = opType.getSize();

        rewriter.replaceOpWithNewOp<qillr::AllocOp>(
            op,
            qillr::QubitType::get(getContext(), resultSize));
        return success();
    }
}; // struct ConvertAlloc

struct ConvertSplit
        : public IndexTrackingOpConversionPattern<quantum::SplitOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        SplitOp op,
        SplitOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Value, 4> replace(op->getNumResults(), adaptor.getInput());
        rewriter.replaceOpWithMultiple(op, ValueRange{replace});
        return success();
    }
}; // struct ConvertSplit

struct ConvertMerge
        : public IndexTrackingOpConversionPattern<quantum::MergeOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        MergeOp op,
        MergeOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithMultiple(op, adaptor.getInput());
        return success();
    }
}; // struct ConvertMerge

struct ConvertMeasure
        : public IndexTrackingOpConversionPattern<quantum::MeasureOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto opType = op.getResult().getType();
        auto loc = op.getLoc();

        // auto resultAlloc = rewriter.create<qillr::AllocResultOp>(
        //     loc,
        //     qillr::ResultType::get(op.getContext(), opType.getSize()));
        // rewriter.create<qillr::MeasureOp>(loc, adaptor.getInput(),
        // resultAlloc); auto readMeasurement =
        // rewriter.create<qillr::ReadMeasurementOp>(
        //     loc,
        //     resultAlloc.getResult());

        // auto i1Type = rewriter.getI1Type();
        // auto genTensorType = mlir::RankedTensorType::get({1}, i1Type);
        // auto tensor = rewriter.create<tensor::FromElementsOp>(
        //     loc,
        //     genTensorType,
        //     readMeasurement.getResult());

        // rewriter.replaceOpWithMultiple(
        //     op,
        //     {tensor.getResult(), adaptor.getInput()});
        return success();
    }
}; // struct ConvertMeasure

struct ConvertDealloc
        : public IndexTrackingOpConversionPattern<quantum::DeallocateOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        DeallocateOp op,
        DeallocateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        std::optional<uint64_t> index;
        auto result = this->lookupSingle(op.getInput(), index, rewriter);
        if (failed(result)) return result;

        rewriter.replaceOpWithNewOp<qillr::ResetOp>(
            op,
            adaptor.getInput(),
            index);
        return success();
    }
}; // struct ConvertDealloc

struct ConvertFunc : public IndexTrackingOpConversionPattern<func::FuncOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op,
        func::FuncOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ftype = op.getFunctionType();

        auto genFuncTy = typeConverter->convertType(ftype);
        auto genFunc = rewriter.create<func::FuncOp>(
            op->getLoc(),
            op.getSymName(),
            llvm::dyn_cast<FunctionType>(genFuncTy));

        if (!op.isExternal()) {
            rewriter.inlineRegionBefore(
                adaptor.getBody(),
                genFunc.getBody(),
                genFunc.end());
        }
        rewriter.replaceOp(op, genFunc);

        return success();
    }
}; // struct ConvertFunc

template<typename SourceOp, typename TargetOp>
struct ConvertUnaryOp : public IndexTrackingOpConversionPattern<SourceOp> {
    using IndexTrackingOpConversionPattern<
        SourceOp>::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        std::optional<uint64_t> index;
        auto result = this->lookupSingle(op.getInput(), index, rewriter);
        if (failed(result)) return result;

        rewriter.create<TargetOp>(op.getLoc(), adaptor.getInput(), index);
        if (op->getNumResults() > 0)
            rewriter.replaceOp(op, adaptor.getInput());
        else
            op->erase();
        return success();
    }
}; // struct ConvertUnaryOp

template<typename SourceOp, typename TargetOp>
struct ConvertRotationOp : public IndexTrackingOpConversionPattern<SourceOp> {
    using IndexTrackingOpConversionPattern<
        SourceOp>::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        std::optional<uint64_t> index;
        auto result = this->lookupSingle(op.getInput(), index, rewriter);
        if (failed(result)) return result;

        rewriter.create<TargetOp>(
            op.getLoc(),
            adaptor.getInput(),
            adaptor.getTheta(),
            index);
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
}; // struct ConvertRotationOp

struct ConvertSwap : public IndexTrackingOpConversionPattern<quantum::SWAPOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        SWAPOp op,
        SWAPOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        std::optional<uint64_t> indexLhs;
        std::optional<uint64_t> indexRhs;
        LogicalResult result = LogicalResult::success();
        result = this->lookupSingle(op.getLhs(), indexLhs, rewriter);
        if (failed(result)) return result;
        result = this->lookupSingle(op.getRhs(), indexRhs, rewriter);
        if (failed(result)) return result;

        // Retrieve the two input qubits from the adaptor.
        Value qubit1 = adaptor.getLhs();
        Value qubit2 = adaptor.getRhs();
        rewriter.create<qillr::SwapOp>(
            op.getLoc(),
            qubit1,
            qubit2,
            indexLhs,
            indexRhs);
        rewriter.replaceOp(op, {qubit1, qubit2});
        return success();
    }
};

struct ConvertCSwap
        : public IndexTrackingOpConversionPattern<quantum::CSWAPOp> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        CSWAPOp op,
        CSWAPOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        LogicalResult result = LogicalResult::success();
        std::optional<uint64_t> indexCtrl;
        std::optional<uint64_t> indexLhs;
        std::optional<uint64_t> indexRhs;
        result = this->lookupSingle(op.getLhs(), indexCtrl, rewriter);
        if (failed(result)) return result;
        result = this->lookupSingle(op.getLhs(), indexLhs, rewriter);
        if (failed(result)) return result;
        result = this->lookupSingle(op.getLhs(), indexRhs, rewriter);
        if (failed(result)) return result;

        Value control = adaptor.getControl();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();
        rewriter.create<qillr::CSwapOp>(
            op.getLoc(),
            control,
            lhs,
            rhs,
            indexCtrl,
            indexLhs,
            indexRhs);
        rewriter.replaceOp(op, {control, lhs, rhs});
        return success();
    }
};

struct ConvertCU1 : public IndexTrackingOpConversionPattern<quantum::CU1Op> {
    using IndexTrackingOpConversionPattern::IndexTrackingOpConversionPattern;

    LogicalResult matchAndRewrite(
        CU1Op op,
        CU1OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        LogicalResult result = LogicalResult::success();
        std::optional<uint64_t> indexCtrl;
        std::optional<uint64_t> indexTarget;
        result = this->lookupSingle(op.getControl(), indexCtrl, rewriter);
        if (failed(result)) return result;
        result = this->lookupSingle(op.getTarget(), indexTarget, rewriter);
        if (failed(result)) return result;

        // Retrieve the two input qubits from the adaptor.
        Value control = adaptor.getControl();
        Value target = adaptor.getTarget();
        Value angle = adaptor.getAngle();
        rewriter.create<qillr::CU1Op>(
            op.getLoc(),
            control,
            target,
            angle,
            indexCtrl,
            indexTarget);

        rewriter.replaceOp(op, {control, target});
        return success();
    }
};

} // namespace

void ConvertQuantumToQILLRPass::runOnOperation()
{
    TypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](quantum::QubitType ty) {
        return qillr::QubitType::get(ty.getContext(), ty.getSize());
    });

    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<quantum::dataflow::RegisterRangesAnalysis>();
    if (failed(solver.initializeAndRun(getOperation())))
        return signalPassFailure();

    quantum::populateConvertQuantumToQILLRPatterns(
        solver,
        typeConverter,
        patterns);
    populateFuncTypeConversionPatterns(typeConverter, patterns);
    rvsdg::populateConvertRVSDGPatterns(typeConverter, patterns);

    target.addIllegalDialect<quantum::QuantumDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<qillr::QILLRDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<rvsdg::GammaNode>(
        [&](rvsdg::GammaNode op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<rvsdg::YieldOp>([&](rvsdg::YieldOp op) {
        return typeConverter.isLegal(op->getOperandTypes());
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        return signalPassFailure();
}

void mlir::quantum::populateConvertQuantumToQILLRPatterns(
    mlir::DataFlowSolver &solver,
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertAlloc,
        ConvertMeasure,
        ConvertUnaryOp<quantum::ResetOp, qillr::ResetOp>,
        ConvertUnaryOp<quantum::DeallocateOp, qillr::DeallocateOp>,
        ConvertUnaryOp<quantum::HOp, qillr::HOp>,
        ConvertUnaryOp<quantum::XOp, qillr::XOp>,
        ConvertUnaryOp<quantum::YOp, qillr::YOp>,
        ConvertUnaryOp<quantum::ZOp, qillr::ZOp>,
        ConvertUnaryOp<quantum::IdOp, qillr::IdOp>,
        ConvertUnaryOp<quantum::SXOp, qillr::SXOp>,
        ConvertRotationOp<quantum::RxOp, qillr::RxOp>,
        ConvertRotationOp<quantum::RyOp, qillr::RyOp>,
        ConvertRotationOp<quantum::RzOp, qillr::RzOp>,
        ConvertRotationOp<quantum::PhaseOp, qillr::PhaseOp>,
        ConvertCSwap,
        ConvertSwap>(
        solver,
        typeConverter,
        patterns.getContext(),
        /* benefit*/ 1);
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQILLRPass()
{
    return std::make_unique<ConvertQuantumToQILLRPass>();
}
