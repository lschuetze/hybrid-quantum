/// Implements the ConvertQuantumToQIRPass.
///
/// @file
/// @author     Lars Schütze (lars.schuetze@tu-dresden.de)

#include "cinm-mlir/Conversion/QuantumToQIR/QuantumToQIR.h"

#include "cinm-mlir/Dialect/QIR/IR/QIR.h"
#include "cinm-mlir/Dialect/QIR/IR/QIRTypes.h"
#include "cinm-mlir/Dialect/Quantum/IR/Quantum.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumOps.h"
#include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::quantum;

//===- Generated includes -------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_CONVERTQUANTUMTOQIR
#include "cinm-mlir/Conversion/Passes.h.inc"

} // namespace mlir

//===----------------------------------------------------------------------===//

struct mlir::quantum::QuantumToQirQubitTypeMapping {
public:
    QuantumToQirQubitTypeMapping() : map() {}

    // Map a `quantum.Qubit` to (possible many) `qir.qubit`
    void allocate(Value quantumQubit, ValueRange qirQubits)
    {
        for (auto qirQubit : qirQubits) map[quantumQubit].push_back(qirQubit);
    }

    llvm::ArrayRef<Value> find(Value quantum) { return map[quantum]; }

private:
    llvm::DenseMap<Value, llvm::SmallVector<Value>> map;
};

namespace {

struct ConvertQuantumToQIRPass
        : mlir::impl::ConvertQuantumToQIRBase<ConvertQuantumToQIRPass> {
    using ConvertQuantumToQIRBase::ConvertQuantumToQIRBase;

    void runOnOperation() override;
};

template<typename Op>
struct QuantumToQIROpConversion : OpConversionPattern<Op> {
    explicit QuantumToQIROpConversion(
        TypeConverter* typeConverter,
        MLIRContext* context,
        QuantumToQirQubitTypeMapping* mapping)
            : OpConversionPattern<Op>(context, /* benefit */ 1),
              mapping(mapping),
              typeConverter(typeConverter)
    {}

    QuantumToQirQubitTypeMapping* mapping;
    TypeConverter* typeConverter;
};

struct ConvertAlloc : public QuantumToQIROpConversion<quantum::AllocOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

    LogicalResult matchAndRewrite(
        AllocOp op,
        AllocOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        int64_t size = op.getType().getSize();
        llvm::SmallVector<Value> qubits;
        for (int64_t i = 0; i < size; i++) {
            auto qubit = rewriter.create<qir::AllocOp>(
                op.getLoc(),
                qir::QubitType::get(getContext()));
            qubits.push_back(qubit.getResult());
        }
        mapping->allocate(op.getResult(), qubits);
        rewriter.eraseOp(op);
        return success();
    }
}; // struct ConvertAllocOp

struct ConvertMeasure : public QuantumToQIROpConversion<quantum::MeasureOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

    LogicalResult matchAndRewrite(
        MeasureOp op,
        MeasureOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto inArg = op.getInput();
        auto dim = inArg.getType().getSize();
        auto inResult = op.getResult();
        auto inMeasurement = op.getMeasurement();
        auto loc = op.getLoc();

        // Create ralloc, measurement, read_measurement
        // for each qubit in the qubit register
        auto mapped = mapping->find(inArg);
        // Map resulting qubit to qubit memory reference
        mapping->allocate(inResult, mapped);
        auto tensorType =
            mlir::RankedTensorType::get({dim}, rewriter.getI1Type());
        ArrayRef<int64_t> shape = {dim};
        auto resultTensor =
            rewriter
                .create<mlir::tensor::EmptyOp>(loc, shape, rewriter.getI1Type())
                .getResult();
        int idx = 0;
        for (auto &genInput : mapped) {
            // Create new result type holding measurement value
            auto genResultDef = rewriter
                                    .create<qir::AllocResultOp>(
                                        loc,
                                        qir::ResultType::get(op.getContext()))
                                    .getResult();
            auto in =
                llvm::dyn_cast<::mlir::TypedValue<::mlir::qir::QubitType>>(
                    genInput);
            rewriter.create<qir::MeasureOp>(loc, in, genResultDef);

            auto genMeasurement = rewriter
                                      .create<qir::ReadMeasurementOp>(
                                          loc,
                                          rewriter.getI1Type(),
                                          genResultDef)
                                      .getResult();

            auto genIndex =
                rewriter.create<mlir::index::ConstantOp>(loc, idx).getResult();

            rewriter.create<mlir::tensor::InsertOp>(
                loc,
                tensorType,
                genMeasurement,
                resultTensor,
                genIndex);

            // Replace direct uses of the measurement value with QIR
            // values rewriter.replaceAllUsesWith(inResult, genInput);
            // rewriter.replaceAllUsesWith(inMeasurement, genResult);
            rewriter.replaceOp(op, {resultTensor, genInput});
        }
        return success();
    }
}; // struct ConvertMeasure

struct ConvertFunc : public QuantumToQIROpConversion<func::FuncOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

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

struct ConvertH : public QuantumToQIROpConversion<quantum::HOp> {
    using QuantumToQIROpConversion::QuantumToQIROpConversion;

    LogicalResult matchAndRewrite(
        HOp op,
        HOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto qirQubit = mapping->find(op.getInput())[0];
        mapping->allocate(op.getResult(), qirQubit);

        rewriter.create<qir::HOp>(op.getLoc(), qirQubit);
        rewriter.eraseOp(op);

        return success();
    }
}; // struct ConvertAllocOp

} // namespace

void ConvertQuantumToQIRPass::runOnOperation()
{
    OneToNTypeConverter typeConverter;
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](quantum::QubitType ty) {
        return qir::QubitType::get(ty.getContext());
    });
    typeConverter.addConversion([&](FunctionType fty) {
        llvm::SmallVector<Type> argTypes, resTypes;

        for (auto ins : fty.getInputs())
            argTypes.push_back(typeConverter.convertType(ins));

        for (auto res : fty.getResults())
            resTypes.push_back(typeConverter.convertType(res));

        return FunctionType::get(fty.getContext(), argTypes, resTypes);
    });

    QuantumToQirQubitTypeMapping mapping;

    quantum::populateConvertQuantumToQIRPatterns(
        typeConverter,
        mapping,
        patterns);

    target.addIllegalDialect<quantum::QuantumDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation* op) { return true; });
    target.addLegalDialect<qir::QIRDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isLegal(op.getFunctionType());
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns))))
        return signalPassFailure();
}

void mlir::quantum::populateConvertQuantumToQIRPatterns(
    TypeConverter &typeConverter,
    QuantumToQirQubitTypeMapping &mapping,
    RewritePatternSet &patterns)
{
    patterns.add<ConvertAlloc, ConvertMeasure, ConvertH, ConvertFunc>(
        &typeConverter,
        patterns.getContext(),
        &mapping);
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQIRPass()
{
    return std::make_unique<ConvertQuantumToQIRPass>();
}
