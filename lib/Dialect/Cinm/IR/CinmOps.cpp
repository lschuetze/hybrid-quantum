/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmEnums.cpp.inc"

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"
      >();
}

namespace mlir {
namespace cinm {

::mlir::ParseResult MinOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  return MaxOp::parse(parser, result);
}

SmallVector<Value> MinOp::convertToTiledOps(OpBuilder builder,
                                            ArrayRef<int64_t>,
                                            int64_t reduceClusterSize) {
  return {createVectorReduceMin(builder, getLoc(), getOperand(),
                                reduceClusterSize)};
}

::mlir::ParseResult MaxOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType;
  if (parser.parseOperand(input) || parser.parseColonType(inputType))
    return failure();

  SmallVector<Value> typedInput;
  if (parser.resolveOperand(input, inputType, typedInput))
    return failure();

  result.addOperands(typedInput);

  if (auto shaped = dyn_cast<ShapedType>(inputType)) {
    result.addTypes({shaped.getElementType()});
    return success();
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a tensor type as operand");
  }
}

SmallVector<Value> MaxOp::convertToTiledOps(OpBuilder builder,
                                            ArrayRef<int64_t>,
                                            int64_t reduceClusterSize) {
  return {createVectorReduceMax(builder, getLoc(), getOperand(),
                                reduceClusterSize)};
}

void printMinMaxOp(Value v, ::mlir::OpAsmPrinter &p) {
  p << " ";
  p.printOperand(v);
  p << " : ";
  p.printType(v.getType());
}

void MaxOp::print(::mlir::OpAsmPrinter &p) { printMinMaxOp(getOperand(), p); }

void MinOp::print(::mlir::OpAsmPrinter &p) { printMinMaxOp(getOperand(), p); }

SmallVector<Value> GemmOp::convertToTiledOps(OpBuilder builder,
                                             ArrayRef<int64_t> tileCounts,
                                             int64_t reduceClusterSize) {
  const Value lhs = getOperand(0);
  const Value rhs = getOperand(1);

  if (tileCounts.empty()) {
    return {createMatmul(builder, getLoc(), lhs, rhs, reduceClusterSize)};
  }

  assert(tileCounts.size() == 2);

  const RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
  const RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

  const RankedTensorType resultType =
      getResult().getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> resultShape = resultType.getShape();

  const Value resultInit = builder.create<tensor::EmptyOp>(
      getLoc(), resultShape, resultType.getElementType());

  const SmallVector<int64_t, 2> tileSizes =
      getTileSizes(tileCounts, resultShape);

  return createNestedAffineForLoops(
      builder, getLoc(), resultShape, tileSizes, ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> lhsOffsets{ShapedType::kDynamic, 0};
        const SmallVector<int64_t, 2> lhsSizes{tileSizes[0],
                                               lhsType.getDimSize(1)};
        const SmallVector<int64_t, 2> unitStrides{1, 1};
        const SmallVector<int64_t, 2> &lhsStrides = unitStrides;
        const SmallVector<int64_t, 2> rhsOffsets{0, ShapedType::kDynamic};
        const SmallVector<int64_t, 2> rhsSizes{rhsType.getDimSize(0),
                                               tileSizes[1]};
        const SmallVector<int64_t, 2> &rhsStrides = unitStrides;
        const SmallVector<int64_t, 2> resultOffsets{ShapedType::kDynamic,
                                                    ShapedType::kDynamic};
        const SmallVector<int64_t, 2> &resultSizes = tileSizes;
        const SmallVector<int64_t, 2> resultStrides = unitStrides;

        const SmallVector<Value> lhsDynamicOffsets{indices[0]};
        const RankedTensorType lhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                1, lhsType, lhsOffsets, lhsSizes, lhsStrides);
        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
            ValueRange{}, lhsOffsets, lhsSizes, lhsStrides);

        const SmallVector<Value> rhsDynamicOffsets{indices[1]};
        const Type rhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                1, rhsType, rhsOffsets, rhsSizes, rhsStrides);
        const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, rhsSliceType, rhs, rhsDynamicOffsets, ValueRange{},
            ValueRange{}, rhsOffsets, rhsSizes, rhsStrides);

        const Value resultSlice =
            createMatmul(builder, loc, lhsSlice, rhsSlice, reduceClusterSize);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, resultSlice, iterArgs[0], indices, ValueRange{}, ValueRange{},
            resultOffsets, resultSizes, resultStrides);

        return {result};
      });
}

::mlir::LogicalResult GemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLeft().getType());
  ShapeAdaptor rhsShape(adaptor.getRight().getType());

  SmallVector<int64_t, 2> outShape;
  outShape.push_back(lhsShape.getDimSize(0));
  outShape.push_back(rhsShape.getDimSize(1));

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

SmallVector<Value> GemvOp::convertToTiledOps(OpBuilder builder,
                                             ArrayRef<int64_t> tileCounts,
                                             int64_t reduceClusterSize) {
  const Value lhs = getOperand(0);
  const Value rhs = getOperand(1);

  if (tileCounts.empty()) {
    return {createMatmul(builder, getLoc(), lhs, rhs, reduceClusterSize)};
  }

  const RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();

  const RankedTensorType resultType =
      getResult().getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> resultShape = resultType.getShape();

  const Value resultInit = builder.create<tensor::EmptyOp>(
      getLoc(), resultShape, resultType.getElementType());

  tileCounts = tileCounts.slice(resultShape.size());
  const SmallVector<int64_t, 2> tileSizes =
      getTileSizes(tileCounts, resultShape);

  return createNestedAffineForLoops(
      builder, getLoc(), resultShape, tileSizes, ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> lhsOffsets{ShapedType::kDynamic, 0};
        const SmallVector<int64_t, 2> lhsSizes{tileSizes[0],
                                               lhsType.getDimSize(1)};
        const SmallVector<int64_t, 2> lhsStrides{1, 1};

        const SmallVector<int64_t, 2> resultOffsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 2> resultSizes = tileSizes;
        const SmallVector<int64_t, 2> resultStrides{1};

        const RankedTensorType lhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                1, lhsType, lhsOffsets, lhsSizes, lhsStrides);
        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsSliceType, lhs, indices, ValueRange{}, ValueRange{},
            lhsOffsets, lhsSizes, lhsStrides);

        const Value resultSlice =
            createMatmul(builder, loc, lhsSlice, rhs, reduceClusterSize);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, resultSlice, iterArgs[0], indices, ValueRange{}, ValueRange{},
            resultOffsets, resultSizes, resultStrides);

        return {result};
      });
}

template <typename OP>
SmallVector<Value> tileElementWiseBinaryOp(ImplicitLocOpBuilder builder, OP op,
                                           ArrayRef<int64_t> tileCounts,
                                           int64_t reduceClusterSize) {
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  RankedTensorType tensorTy = lhs.getType().cast<RankedTensorType>();
  auto shape = tensorTy.getShape();
  const RankedTensorType originalType = tensorTy;
  Value originalShapeValue;
  if (shape.size() > 1) {
    // then flatten the tensors first
    Value flatShapeValue = builder.create<arith::ConstantOp>(
        RankedTensorType::get({1}, builder.getI64Type()),
        builder.getI64TensorAttr({tensorTy.getNumElements()}));
    originalShapeValue = builder.create<arith::ConstantOp>(
        RankedTensorType::get({static_cast<int64_t>(shape.size())},
                              builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    RankedTensorType flatTy = RankedTensorType::get({tensorTy.getNumElements()},
                                                    tensorTy.getElementType());
    lhs = builder.create<tensor::ReshapeOp>(flatTy, lhs, flatShapeValue);
    rhs = builder.create<tensor::ReshapeOp>(flatTy, rhs, flatShapeValue);
    tensorTy = flatTy;
    shape = tensorTy.getShape();
  }

  const Value resultInit =
      builder.create<tensor::EmptyOp>(tensorTy, ValueRange{});

  const SmallVector<int64_t, 2> tileSizes = {
      tensorTy.getNumElements() / reduceClusterSize, reduceClusterSize};

  SmallVector<Value, 1> result = createNestedAffineForLoops(
      builder, op.getLoc(), {tensorTy.getNumElements()}, {reduceClusterSize},
      ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> offsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 2> sizes{reduceClusterSize};
        const SmallVector<int64_t, 2> strides{1};

        const RankedTensorType sliceTy = RankedTensorType::get(
            {reduceClusterSize}, tensorTy.getElementType());

        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, lhs, indices, ValueRange{},
            ValueRange{}, offsets, sizes, strides);
        const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, rhs, indices, ValueRange{},
            ValueRange{}, offsets, sizes, strides);

        // here we recreate the same op with smaller dimensions
        const Value resultSlice = builder.create<OP>(loc, lhsSlice, rhsSlice);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, resultSlice, iterArgs[0], indices, ValueRange{},
            ValueRange{}, offsets, sizes, strides);

        return {result};
      });
  if (originalType.getShape().size() > 1) {
    result[0] = builder.create<tensor::ReshapeOp>(originalType, result[0],
                                                  originalShapeValue);
  }
  return result;
}

SmallVector<Value> AddOp::convertToTiledOps(OpBuilder builder,
                                            ArrayRef<int64_t> tileCounts,
                                            int64_t reduceClusterSize) {
  ImplicitLocOpBuilder ibuilder(getLoc(), builder);
  return tileElementWiseBinaryOp<AddOp>(ibuilder, *this, tileCounts,
                                        reduceClusterSize);
}

::mlir::LogicalResult GemvOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  auto result = ShapedTypeComponents(adaptor.getRight().getType());
  inferredReturnShapes.emplace_back(std::move(result));
  return success();
}

::mlir::LogicalResult SimSearchOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getLeft().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

::mlir::LogicalResult TopKOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

// Copied from the TOSA codebase.
::mlir::LogicalResult TransposeOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  ShapeAdaptor permsShape(adaptor.getPerms().getType());

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of the
  // input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(permsShape.getDimSize(0), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputShape.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputShape.getRank(), inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
  // If the permuations are a constant we can directly determine the output
  // shape.
  DenseIntElementsAttr attr;
  if (matchPattern(adaptor.getPerms(), m_Constant(&attr)) &&
      attr.getType().getRank() == 1) {
    ShapeAdaptor permShape = attr;
    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

SmallVector<Value> ReduceOp::convertToTiledOps(OpBuilder builder,
                                               ArrayRef<int64_t>,
                                               int64_t reduceClusterSize) {
  if (getMethod() == ReduceMethod::ADD) {
    return {createVectorReduceAdd(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else if (getMethod() == ReduceMethod::MUL) {
    return {createVectorReduceMul(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else {
    abort();
  }
}

ParseResult SimSearchOp::parse(::mlir::OpAsmParser &parser,
                               ::mlir::OperationState &result) {

  //  let assemblyFormat= "$metric `,` $k `(` $left `,` $right `)` attr-dict `:`
  //  type($left)";
  std::string opname;
  if (parser.parseKeywordOrString(&opname))
    return failure();

  SimilarityMetric metric;
  if (opname == "cos") {
    metric = SimilarityMetric::COS;
  } else if (opname == "dot") {
    metric = SimilarityMetric::DOT;
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a string \"cos\" or \"dot\"");
  }

  result.addAttribute(getMetricAttrName(result.name),
                      SimilarityMetricAttr::get(result.getContext(), metric));

  int64_t numK;
  if (parser.parseInteger(numK))
    return failure();

  auto i64Ty = IntegerType::get(result.getContext(), 64);
  result.addAttribute(getKAttrName(result.name), IntegerAttr::get(i64Ty, numK));

  OpAsmParser::UnresolvedOperand left, right;
  NamedAttrList attrDict;
  Type opTy;

  if (parser.parseLParen() || parser.parseOperand(left, false) ||
      parser.parseComma() || parser.parseOperand(right, false) ||
      parser.parseRParen() || parser.parseColonType(opTy))
    return failure();

  result.addAttributes(attrDict);
  if (parser.resolveOperand(left, opTy, result.operands) ||
      parser.resolveOperand(right, opTy, result.operands))
    return failure();

  // finally add result types

  auto eltTy = opTy.cast<RankedTensorType>().getElementType();
  auto resultValuesTy = RankedTensorType::get({ShapedType::kDynamic}, eltTy);
  auto resultIndicesTy = RankedTensorType::get(
      {ShapedType::kDynamic}, IndexType::get(result.getContext()));
  result.addTypes({resultValuesTy, resultIndicesTy});

  return success();
}

void SimSearchOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printKeywordOrString(stringifySimilarityMetric(getMetric()));
  p << ' ' << getK().getInt();
  p << " (";
  p.printOperand(getLeft());
  p << ", ";
  p.printOperand(getRight());
  p << ") : ";
  p.printType(getLeft().getType());
}

} // namespace cinm
} // namespace mlir

// parsers/printers
