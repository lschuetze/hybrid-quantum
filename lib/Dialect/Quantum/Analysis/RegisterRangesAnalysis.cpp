//===- RegisterRangesAnalysis.cpp - Quantum Register Interval Anaysis ---===//
//
// @author  Lars Schütze (lars.schuetze@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "quantum-mlir/Dialect/Quantum/Analysis/RegisterRangesAnalysis.h"

#include "quantum-mlir/Dialect/Quantum/Interfaces/InferRegisterRangesInterface.h"

#include "llvm/Support/Debug.h"

#include <llvm/Support/Casting.h>

#define DEBUG_TYPE "register-range-analysis"

using namespace mlir;
using namespace mlir::quantum;
using namespace mlir::quantum::dataflow;

LogicalResult RegisterRangesAnalysis::visitOperation(
    Operation* op,
    ArrayRef<const RegisterRangesLattice*> operands,
    ArrayRef<RegisterRangesLattice*> results)
{
    auto inferrable = llvm::dyn_cast<InferRegisterRangesInterface>(op);
    if (!inferrable) {
        setAllToEntryStates(results);
        return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    auto argRanges =
        llvm::map_to_vector(operands, [](const RegisterRangesLattice* lattice) {
            return lattice->getValue();
        });

    auto joinCallback = [&](Value v, const RegisterRanges &attrs) {
        auto result = dyn_cast<OpResult>(v);
        if (!result) return;
        assert(llvm::is_contained(op->getResults(), result));

        LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
        RegisterRangesLattice* lattice = results[result.getResultNumber()];
        RegisterRanges oldRange = lattice->getValue();

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
            changed |= lattice->join(RegisterRanges());
        }
        propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
    return success();
}
