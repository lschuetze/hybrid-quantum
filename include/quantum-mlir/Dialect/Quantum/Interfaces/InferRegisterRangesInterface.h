//===- InterRegisterRangesInterface.h - Register Range Inference -*- C++-*-===//
//
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the register range inference interface
// defined in `InterRegisterRangesInterface.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTUM_INTERFACES_INFERREGISTERRANGESINTERFACE_H
#define MLIR_QUANTUM_INTERFACES_INFERREGISTERRANGESINTERFACE_H

#include "mlir/IR/OpDefinition.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>

namespace mlir {
namespace quantum {

class ConstantRegisterRanges {
public:
    ConstantRegisterRanges(Value value, ConstantIntRanges ranges)
            : value(value),
              ranges(ranges)
    {}

    bool operator==(const ConstantRegisterRanges &other) const;

    const Value getRegisterValue() const;
    const ConstantIntRanges getRanges() const;

    friend raw_ostream &
    operator<<(raw_ostream &os, const ConstantRegisterRanges &range);

private:
    Value value;
    ConstantIntRanges ranges;
};

raw_ostream &operator<<(raw_ostream &, const ConstantRegisterRanges &);

class RegisterRanges {
public:
    RegisterRanges() {}

    RegisterRanges(ConstantRegisterRanges range) { ranges.push_back(range); }

    RegisterRanges(llvm::SmallVector<ConstantRegisterRanges> ranges)
            : ranges(std::move(ranges))
    {}

    /// Whether the range is uninitialized. This happens when the state hasn't
    /// been set during the analysis.
    bool isUninitialized() const { return ranges.empty(); }

    /// Compare two ranges.
    bool operator==(const RegisterRanges &rhs) const
    {
        if (ranges.size() != rhs.ranges.size()) return false;
        bool cmp = true;
        for (unsigned int i = 0; i < ranges.size(); ++i)
            cmp &= ranges[i] == rhs.ranges[i];
        return cmp;
    }

    /// Compute the least upper bound of two ranges.
    static RegisterRanges
    join(const RegisterRanges &lhs, const RegisterRanges &rhs)
    {
        if (lhs.isUninitialized()) return rhs;
        if (rhs.isUninitialized()) return lhs;

        llvm::SmallVector<ConstantRegisterRanges> result;
        for (unsigned int i = 0; i < lhs.ranges.size(); ++i) {
            for (unsigned int j = 0; j < rhs.ranges.size(); ++j)
                if (lhs.ranges == rhs.ranges) return lhs;
        }
        return RegisterRanges();
    }

    const llvm::ArrayRef<ConstantRegisterRanges> getValue() const
    {
        assert(!isUninitialized());
        return ranges;
    }

    /// Print the integer value range.
    void print(raw_ostream &os) const
    {
        for (const auto &elem : ranges) os << elem;
    }

private:
    llvm::SmallVector<ConstantRegisterRanges> ranges;
};

/// The type of the `setResultRanges` callback provided to ops implementing
/// InferRegisterRangesAnalysis. It should be called once for each quantum
/// register result value and be passed the RegisterRanges corresponding to
/// that value.
using SetRangeFn = llvm::function_ref<void(Value, const RegisterRanges &)>;

class InferRegisterRangesInterface;

namespace registerrange::detail {

/// Default implementation of `inferResultRanges`.
void defaultInferResultRanges(
    InferRegisterRangesInterface interface,
    ArrayRef<RegisterRanges> argRanges,
    SetRangeFn setResultRanges);

} // namespace registerrange::detail
} // namespace quantum
} // namespace mlir

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h.inc"

#endif // MLIR_QUANTUM_INTERFACES_INFERREGISTERRANGESINTERFACE_H
