//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_PASSES
#define QUANTUM_PASSES

include "mlir/Pass/PassBase.td"

def QuantumOptimise : Pass<"quantum-optimise", "ModuleOp"> {
  let summary = "Optimise the `quantum` dialect";

  let description = [{
  }];

  let constructor = "mlir::quantum::createQuantumOptimisePass()";
}

def HermitianCancel : Pass<"hermitian-cancel", "ModuleOp"> {
  let summary = "Hermitian Cancellation on the `quantum` dialect";

  let description = [{
  }];

  let constructor = "mlir::quantum::createHermitianCancelPass()";
}

#endif // QUANTUM_PASSES
