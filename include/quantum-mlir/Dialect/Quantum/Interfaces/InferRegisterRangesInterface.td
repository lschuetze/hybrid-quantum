//===- InferRegisterRangesInterface.td - Register Range Inference --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
//
// Defines the interface for range analysis on quantum registers
//
//===-----------------------------------------------------===//

#ifndef MLIR_QUANTUM_INTERFACES_INFERREGISTERRANGESINTERFACE
#define MLIR_QUANTUM_INTERFACES_INFERREGISTERRANGESINTERFACE

include "mlir/IR/OpBase.td"

def InferRegisterRangesInterface : OpInterface<"InferRegisterRangesInterface"> {
  let description = [{
    Allows operations to participate in range analysis for quantum register values by
    providing a methods that allows them to specify lower and upper bounds on their
    result(s) given lower and upper bounds on their input(s) if known.
  }];
  let cppNamespace = "::mlir::quantum";
  
  let methods = [
    InterfaceMethod<[{
      Infer the bounds on the results of this op given the lattice representation
      of the bounds for its arguments. For each result value or block argument
      (that isn't a branch argument, since the dataflow analysis handles
      those case), the method should call `setValueRange` with that `Value`
      as an argument. When implemented, `setValueRange` should be called on
      all result values for the operation.

      This method allows for more precise implementations when operations
      want to reason about inputs which may be undefined during the analysis.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferResultRanges",
    /*args=*/(ins "::llvm::ArrayRef<::mlir::quantum::RegisterRanges>":$argRanges,
                  "::mlir::quantum::SetRangeFn":$setResultRanges),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      ::mlir::quantum::registerrange::detail::defaultInferResultRanges($_op,
                                                         argRanges,
                                                         setResultRanges);
    }]>
  ];
}

#endif // MLIR_QUANTUM_INTERFACES_INFERREGISTERRANGESINTERFACE
