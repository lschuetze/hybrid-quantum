//===- InferRegisterRangesInterface.cpp -  Register inference interface ---===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h"

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.cpp.inc"

#include <llvm/ADT/SmallVector.h>

using namespace mlir;
using namespace mlir::quantum;

void mlir::quantum::registerrange::detail::defaultInferResultRanges(
    InferRegisterRangesInterface interface,
    ArrayRef<RegisterRanges> argRanges,
    SetRangeFn setResultRanges)
{
    // Standard implementation passes for each input operand its analysis result
    // to the corresponding result value
    for (auto &&[result, range] :
         llvm::zip(interface->getResults(), argRanges)) {
        if (range.isUninitialized()) return;
        setResultRanges(result, range);
    }
}

bool ConstantRegisterRanges::operator==(
    const ConstantRegisterRanges &other) const
{
    return getRegisterValue() == other.getRegisterValue()
           && getRanges() == other.getRanges();
}

const Value ConstantRegisterRanges::getRegisterValue() const { return value; }

const ConstantIntRanges ConstantRegisterRanges::getRanges() const
{
    return ranges;
}

raw_ostream &
mlir::quantum::operator<<(raw_ostream &os, const ConstantRegisterRanges &range)
{
    return os << range.getRegisterValue() << " -> " << range.getRanges();
}
