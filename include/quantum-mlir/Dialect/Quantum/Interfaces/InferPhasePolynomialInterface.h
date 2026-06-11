//===- InferPhasePolynomialInterface.h - Phase Poly. Inference ---*- C++-*-===//
//
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the phase polynomial inference interface
// defined in `InferPhasePolynomialInterface.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE_H
#define MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE_H

#include "mlir/IR/OpDefinition.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace quantum {

class ConstantPhasePolynomial {
public:
    ConstantPhasePolynomial(const unsigned qubitCount, const unsigned epoch = 0)
            : parityVal(qubitCount, false),
              epochVal(epoch)
    {}

    ConstantPhasePolynomial(const ConstantPhasePolynomial &other)
            : parityVal(other.parityVal),
              epochVal(other.epochVal)
    {}

    bool operator==(const ConstantPhasePolynomial &other) const;

    ConstantPhasePolynomial
    parityOr(const ConstantPhasePolynomial &other) const;

    ConstantPhasePolynomial
    parityAnd(const ConstantPhasePolynomial &other) const;

    ConstantPhasePolynomial
    parityXor(const ConstantPhasePolynomial &other) const;

    const llvm::BitVector &getParity() const;

    void setBit(size_t index) { parityVal.set(index); }

    unsigned getEpoch() const;

    void setEpoch(unsigned epoch) { epochVal = epoch; }

    friend llvm::raw_ostream &operator<<(
        llvm::raw_ostream &os,
        const ConstantPhasePolynomial &polynomial);

private:
    llvm::BitVector parityVal;
    unsigned epochVal;
};

llvm::raw_ostream &
operator<<(llvm::raw_ostream &, const ConstantPhasePolynomial &);

/// This lattice value represents the phase polynomial of an SSA value.
class PhasePolynomial {
public:
    PhasePolynomial() = default;

    /// Create a phase polynomial lattice value
    PhasePolynomial(ConstantPhasePolynomial value) : value(std::move(value)) {}

    /// Check whether the state is uninitialized
    bool isUninitialized() const { return !value.has_value(); }

    /// Get the known phase polynomial.
    const ConstantPhasePolynomial &getValue() const
    {
        assert(!isUninitialized());
        return *value;
    }

    /// Compare two phase polynomials.
    bool operator==(const PhasePolynomial &rhs) const
    {
        return value == rhs.value;
    }

    /// Print the phase polynomial
    void print(llvm::raw_ostream &os) const { os << value; }

    /// Compute the combination of two phase polynomials
    static PhasePolynomial
    join(const PhasePolynomial &lhs, const PhasePolynomial &rhs)
    {
        if (lhs.isUninitialized()) return rhs;
        if (rhs.isUninitialized()) return lhs;
        return PhasePolynomial{lhs.getValue().parityOr(rhs.getValue())};
    }

    /// Compute the symmetric difference of two phase polynomials
    static PhasePolynomial
    meet(const PhasePolynomial &lhs, const PhasePolynomial &rhs)
    {
        if (lhs.isUninitialized()) return rhs;
        if (rhs.isUninitialized()) return lhs;
        return PhasePolynomial{lhs.getValue().parityXor(rhs.getValue())};
    }

    static PhasePolynomial nextEpoche(const PhasePolynomial &other) {}

private:
    /// The known phase polynomial.
    std::optional<ConstantPhasePolynomial> value;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const PhasePolynomial &);

using SetPolynomialFn =
    llvm::function_ref<void(Value, const PhasePolynomial &)>;

class InferPhasePolynomialInterface;

namespace phasepolynomial::detail {

void defaultInferResultPolynomial(
    InferPhasePolynomialInterface interface,
    ArrayRef<PhasePolynomial> argPolynomials,
    SetPolynomialFn setResultPolynomials);

} // namespace phasepolynomial::detail

} // namespace quantum
} // namespace mlir

namespace llvm {

template<>
struct DenseMapInfo<mlir::quantum::ConstantPhasePolynomial> {
    static mlir::quantum::ConstantPhasePolynomial getEmptyKey()
    {
        return mlir::quantum::ConstantPhasePolynomial(0, ~0u);
    }

    static mlir::quantum::ConstantPhasePolynomial getTombstoneKey()
    {
        return mlir::quantum::ConstantPhasePolynomial(0, ~0u - 1);
    }

    static unsigned
    getHashValue(const mlir::quantum::ConstantPhasePolynomial &v)
    {
        llvm::hash_code h = llvm::hash_value(v.getEpoch());

        for (auto word : v.getParity().getData())
            h = llvm::hash_combine(h, word);

        return static_cast<unsigned>(h);
    }

    static bool isEqual(
        const mlir::quantum::ConstantPhasePolynomial &lhs,
        const mlir::quantum::ConstantPhasePolynomial &rhs)
    {
        return lhs == rhs;
    }
};

} // namespace llvm

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.h.inc"

#endif // MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE_H
