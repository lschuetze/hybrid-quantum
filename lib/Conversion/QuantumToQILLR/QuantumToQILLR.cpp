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
#include "quantum-mlir/Dialect/Quantum/IR/Quantum.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGBase.h"
#include "quantum-mlir/Dialect/RVSDG/IR/RVSDGOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/OneToNFuncConversions.h>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

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

struct ConvertAlloc : public OpConversionPattern<quantum::AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto opType = op.getResult().getType();
        if (!opType.isSingleQubit())
            return rewriter.notifyMatchFailure(
                op,
                "Please run --quantum-multi-qubit-legalize to transform "
                "multi-qubit into single-qubits.");

        rewriter.replaceOpWithNewOp<qillr::AllocOp>(
            op,
            qillr::QubitType::get(getContext()));
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertMeasure : public OpConversionPattern<quantum::MeasureOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto opType = op.getResult().getType();
        if (!opType.isSingleQubit())
            return rewriter.notifyMatchFailure(
                op,
                "Please run --quantum-multi-qubit-legalize to transform "
                "multi-qubit into single-qubits.");

        auto loc = op.getLoc();

        auto resultAlloc = rewriter.create<qillr::AllocResultOp>(
            loc,
            qillr::ResultType::get(op.getContext()));
        rewriter.create<qillr::MeasureOp>(loc, adaptor.getInput(), resultAlloc);
        auto readMeasurement = rewriter.create<qillr::ReadMeasurementOp>(
            loc,
            resultAlloc.getResult());

        auto i1Type = rewriter.getI1Type();
        auto genTensorType = mlir::RankedTensorType::get({1}, i1Type);
        auto tensor = rewriter.create<tensor::FromElementsOp>(
            loc,
            genTensorType,
            readMeasurement.getResult());

        rewriter.replaceOpWithMultiple(
            op,
            {tensor.getResult(), adaptor.getInput()});
        return success();
    }
}; // struct ConvertMeasure

struct ConvertSingleMeasure
        : public OpConversionPattern<quantum::MeasureSingleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MeasureSingleOp op,
        MeasureSingleOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();

        auto resultAlloc = rewriter.create<qillr::AllocResultOp>(
            loc,
            qillr::ResultType::get(op.getContext()));
        rewriter.create<qillr::MeasureOp>(loc, adaptor.getInput(), resultAlloc);
        auto readMeasurement = rewriter.create<qillr::ReadMeasurementOp>(
            loc,
            resultAlloc.getResult());

        rewriter.replaceOp(
            op,
            {readMeasurement.getResult(), adaptor.getInput()});
        return success();
    }
}; // struct ConvertSingleMeasure

struct ConvertDealloc : public OpConversionPattern<quantum::DeallocateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        DeallocateOp op,
        DeallocateOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<qillr::ResetOp>(op, adaptor.getInput());
        return success();
    }
}; // struct ConvertDealloc

struct ConvertFunc : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern::OpConversionPattern;

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
struct ConvertUnaryOp : public OpConversionPattern<SourceOp> {
    using OpConversionPattern<SourceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.create<TargetOp>(op.getLoc(), adaptor.getInput());
        if (op->getNumResults() > 0)
            rewriter.replaceOp(op, adaptor.getInput());
        else
            op->erase();
        return success();
    }
}; // struct ConvertUnaryOp

template<typename SourceOp, typename TargetOp>
struct ConvertRotationOp : public OpConversionPattern<SourceOp> {
    using OpConversionPattern<SourceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        SourceOp op,
        OpConversionPattern<SourceOp>::OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.create<TargetOp>(
            op.getLoc(),
            adaptor.getInput(),
            adaptor.getTheta());
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
}; // struct ConvertRotationOp

struct ConvertSwap : public OpConversionPattern<quantum::SWAPOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        SWAPOp op,
        SWAPOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Retrieve the two input qubits from the adaptor.
        Value qubit1 = adaptor.getLhs();
        Value qubit2 = adaptor.getRhs();
        rewriter.create<qillr::SwapOp>(op.getLoc(), qubit1, qubit2);
        rewriter.replaceOp(op, {qubit1, qubit2});
        return success();
    }
};

struct ConvertCSwap : public OpConversionPattern<quantum::CSWAPOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        CSWAPOp op,
        CSWAPOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Value control = adaptor.getControl();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();
        rewriter.create<qillr::CSwapOp>(op.getLoc(), control, lhs, rhs);
        rewriter.replaceOp(op, {control, lhs, rhs});
        return success();
    }
};

struct ConvertCU1 : public OpConversionPattern<quantum::CU1Op> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        CU1Op op,
        CU1OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Retrieve the two input qubits from the adaptor.
        Value control = adaptor.getControl();
        Value target = adaptor.getTarget();
        Value angle = adaptor.getAngle();
        rewriter.create<qillr::CU1Op>(op.getLoc(), control, target, angle);
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
        return qillr::QubitType::get(ty.getContext());
    });

    quantum::populateConvertQuantumToQILLRPatterns(typeConverter, patterns);
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
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<
        ConvertAlloc,
        ConvertMeasure,
        ConvertSingleMeasure,
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
        typeConverter,
        patterns.getContext(),
        /* benefit*/ 1);
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQILLRPass()
{
    return std::make_unique<ConvertQuantumToQILLRPass>();
}
