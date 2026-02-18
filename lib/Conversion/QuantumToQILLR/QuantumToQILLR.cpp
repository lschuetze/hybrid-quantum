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

#include "llvm/Support/Debug.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/OneToNFuncConversions.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
#include <utility>

#define DEBUG_TYPE "quantum-qillr-conversion"

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
        IRMapping &cregs,
        const TypeConverter &typeConverter,
        MLIRContext* context,
        PatternBenefit benefit = 1)
            : OpConversionPattern<OpT>(typeConverter, context, benefit),
              solver(solver),
              cregs(cregs)
    {}

    LogicalResult lookupSingle(
        Value value,
        std::optional<uint64_t> &result,
        ConversionPatternRewriter &rewriter) const;

protected:
    mlir::DataFlowSolver &solver;
    IRMapping &cregs;

}; // struct IndexTrackingOpConversionPattern

template<typename OpT>
LogicalResult IndexTrackingOpConversionPattern<OpT>::lookupSingle(
    Value value,
    std::optional<uint64_t> &result,
    ConversionPatternRewriter &rewriter) const
{
    const RegisterRangesLattice* lattice =
        this->solver.lookupState<RegisterRangesLattice>(value);
    if (!lattice)
        return rewriter.notifyMatchFailure(
            value.getDefiningOp(),
            "Lattice value not found");

    const auto latticeValue = lattice->getValue();
    const auto indices = latticeValue.getValue();

    ConstantIntRanges range(indices[0].getRanges());
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].getRegisterValue() != indices[0].getRegisterValue()) {
            return rewriter.notifyMatchFailure(
                value.getDefiningOp(),
                "Lattice value depends on unequal register values");
        }
        range = range.rangeUnion(indices[i].getRanges());
    }
    bool isConstant = (range.umax() - 1 == range.umin());
    if (isConstant) {
        result = range.umin().trySExtValue();
        return success();
    }

    if (auto valTy = llvm::dyn_cast<quantum::QubitType>(
            indices[0].getRegisterValue().getType()))
        if (valTy.getSize() == range.umax() - range.umin()) return success();
    return failure();
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
        auto size = opType.getSize();

        LLVM_DEBUG(llvm::dbgs() << "Create alloc with size " << size << "\n");

        rewriter.replaceOpWithNewOp<qillr::AllocOp>(
            op,
            qillr::QubitType::get(getContext()),
            size);
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
        LLVM_DEBUG(llvm::dbgs() << "Convert op " << op << "\n");
        SmallVector<ValueRange, 4> replace;
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
            LLVM_DEBUG(
                llvm::dbgs() << "Replace result " << i << "with "
                             << adaptor.getInput() << "\n");
            replace.push_back(adaptor.getInput());
        }
        rewriter.replaceOpWithMultiple(op, replace);
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
        Value reg = adaptor.getInput()[0];
        LLVM_DEBUG(llvm::dbgs() << "Replace result with " << reg << "\n");

        for (size_t i = 0; i < adaptor.getInput().size(); ++i)
            assert(
                reg == adaptor.getInput()[i]
                && "Must merge access to same register.");

        rewriter.replaceOp(op, reg);
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
        // The quantum measurement looks like follows:
        // %c = ... tensor<1xi1>
        // %m, %q1 = measure(%q) : qubit<1> -> measurement<1>, qubit<1>
        // %t = to_tensor(%m) -> tensor<1xi1>
        // %cnew = insert_slice(%t, %c) {0} -> tensor<1xi1>
        auto measurement = op.getMeasurement();
        auto toTensorOp = llvm::filter_to_vector(
            measurement.getUsers(),
            [](Operation* op) { return llvm::isa<ToTensorOp>(op); });
        assert(toTensorOp.size() == 1 && "Only one use implemented.");
        auto insertSliceOpV = llvm::filter_to_vector(
            toTensorOp[0]->getResult(0).getUsers(),
            [](Operation* op) { return llvm::isa<tensor::InsertSliceOp>(op); });
        assert(insertSliceOpV.size() == 1 && "Only one use implemented.");
        auto insertSliceOp =
            llvm::cast<tensor::InsertSliceOp>(insertSliceOpV[0]);
        auto creg = insertSliceOp.getDest();

        if (!cregs.contains(creg)) {
            // Create qillr.ralloc
            auto resultAlloc = rewriter.create<qillr::AllocResultOp>(
                op->getLoc(),
                qillr::ResultType::get(op.getContext()),
                creg.getType().getRank());
            cregs.map(creg, resultAlloc.getResult());
        }
        auto resultAlloc = cregs.lookup(creg);

        std::optional<uint64_t> qubitIndex;
        auto result = this->lookupSingle(op.getInput(), qubitIndex, rewriter);
        if (failed(result)) return result;

        std::optional<uint64_t> resultIndex = insertSliceOp.getStaticOffset(0);
        rewriter.create<qillr::MeasureOp>(
            op->getLoc(),
            adaptor.getInput(),
            resultAlloc,
            qubitIndex,
            resultIndex);

        auto readMeasurement = rewriter.create<qillr::ReadMeasurementOp>(
            op->getLoc(),
            creg.getType(),
            resultAlloc);

        rewriter.replaceOp(insertSliceOp, readMeasurement);
        rewriter.replaceOp(op, {readMeasurement, adaptor.getInput()});
        rewriter.eraseOp(toTensorOp[0]);
        return success();
    }
}; // struct ConvertMeasure

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
        LLVM_DEBUG(llvm::dbgs() << "Convert " << op << "\n");
        std::optional<uint64_t> index;
        auto result = this->lookupSingle(op.getInput(), index, rewriter);
        if (failed(result)) return result;
        LLVM_DEBUG(llvm::dbgs() << "Inferred index " << index << "\n");

        LLVM_DEBUG(
            llvm::dbgs()
            << "Create new op with argument: " << adaptor.getInput() << "\n");
        rewriter.create<TargetOp>(op.getLoc(), adaptor.getInput(), index);
        if (op->getNumResults() > 0)
            rewriter.replaceOp(op, adaptor.getInput());
        else
            rewriter.eraseOp(op);
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
    IRMapping mapping;

    typeConverter.addConversion([](Type ty) { return ty; });
    typeConverter.addConversion([](quantum::QubitType ty) {
        return qillr::QubitType::get(ty.getContext());
    });

    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<quantum::dataflow::RegisterRangesAnalysis>();
    if (failed(solver.initializeAndRun(getOperation())))
        return signalPassFailure();

    quantum::populateConvertQuantumToQILLRPatterns(
        solver,
        mapping,
        typeConverter,
        patterns);
    populateFuncTypeConversionPatterns(typeConverter, patterns);
    rvsdg::populateConvertRVSDGPatterns(typeConverter, patterns);

    target.addIllegalDialect<quantum::QuantumDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<qillr::QILLRDialect>();
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
    IRMapping &mapping,
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
        ConvertSwap,
        ConvertSplit,
        ConvertMerge>(
        solver,
        mapping,
        typeConverter,
        patterns.getContext(),
        /* benefit*/ 1);
}

std::unique_ptr<Pass> mlir::createConvertQuantumToQILLRPass()
{
    return std::make_unique<ConvertQuantumToQILLRPass>();
}
