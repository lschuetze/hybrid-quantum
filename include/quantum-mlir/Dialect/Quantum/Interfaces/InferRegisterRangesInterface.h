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
#include <llvm/ADT/STLExtras.h>
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
    RegisterRanges() = default;

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
        for (unsigned int i = 0; i < ranges.size(); ++i)
            if (ranges[i] != rhs.ranges[i]) return false;
        // Everything matches
        return true;
    }

    /// Compute the least upper bound of two ranges.
    /// Must be monolitic
    static RegisterRanges
    join(const RegisterRanges &lhs, const RegisterRanges &rhs)
    {
        if (lhs.isUninitialized()) return rhs;
        if (rhs.isUninitialized()) return lhs;

        if (lhs == rhs) return lhs;

        llvm::SmallVector<ConstantRegisterRanges> result(lhs.ranges);
        for (unsigned int i = 0; i < rhs.ranges.size(); ++i)
            if (!lhs.contains(rhs.ranges[i]))
                result.emplace_back(rhs.ranges[i]);
        return RegisterRanges(result);
    }

    const llvm::ArrayRef<ConstantRegisterRanges> getValue() const
    {
        assert(!isUninitialized());
        return ranges;
    }

    /// Print the integer value range.
    void print(raw_ostream &os) const
    {
        for (const auto &elem : ranges) os << elem << ", ";
    }

    bool contains(ConstantRegisterRanges entry) const
    {
        for (auto ranges : ranges)
            if (ranges == entry) return true;
        return false;
    }

private:
    llvm::SmallVector<ConstantRegisterRanges> ranges;
};

raw_ostream &operator<<(raw_ostream &, const RegisterRanges &);

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
