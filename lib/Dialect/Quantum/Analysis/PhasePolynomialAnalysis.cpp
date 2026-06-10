//===- RegisterRangesAnalysis.cpp - Quantum Register Interval Anaysis ---===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Dialect/Quantum/Analysis/PhasePolynomialAnalysis.h"

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferPhasePolynomialInterface.h"

#include "llvm/Support/Debug.h"

#include <llvm/Support/Casting.h>

#define DEBUG_TYPE "phase-polynomial-analysis"

using namespace mlir;
using namespace mlir::quantum;
using namespace mlir::quantum::dataflow;

LogicalResult PhasePolynomialAnalysis::visitOperation(
    Operation* op,
    ArrayRef<const PhasePolynomialLattice*> operands,
    ArrayRef<PhasePolynomialLattice*> results)
{
    auto inferrable = llvm::dyn_cast<InferPhasePolynomialInterface>(op);
    if (!inferrable) {
        setAllToEntryStates(results);
        return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    auto argRanges = llvm::map_to_vector(
        operands,
        [](const PhasePolynomialLattice* lattice) {
            return lattice->getValue();
        });

    auto joinCallback = [&](Value v, const PhasePolynomial &attrs) {
        auto result = dyn_cast<OpResult>(v);
        if (!result) return;
        assert(llvm::is_contained(op->getResults(), result));

        LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
        PhasePolynomialLattice* lattice = results[result.getResultNumber()];
        PhasePolynomial oldRange = lattice->getValue();

        ChangeResult changed = lattice->join(attrs);

        // Catch loop results with loop variant bounds and conservatively make
        // them [-inf, inf] so we don't circle around infinitely often (because
        // the dataflow analysis in MLIR doesn't attempt to work out trip counts
        // and often can't).
        bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation* op) {
            return op->hasTrait<OpTrait::IsTerminator>();
        });
        if (isYieldedResult && !oldRange.isUninitialized()
            && !(lattice->getValue() == oldRange)) {
            LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
            changed |= lattice->join(PhasePolynomial());
        }
        propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultPolynomial(argRanges, joinCallback);
    return success();
}
