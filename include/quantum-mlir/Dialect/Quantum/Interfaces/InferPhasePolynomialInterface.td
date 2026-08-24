//===- InferPhasePolynomialInterface.td - Register Range Inference --*- tablegen -*-===//
//
//===-----------------------------------------------------===//
//
// Defines the interface for phase polynomial analysis on quantum registers
//
//===-----------------------------------------------------===//

#ifndef MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE
#define MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE

include "mlir/IR/OpBase.td"

def InferPhasePolynomialInterface : OpInterface<"InferPhasePolynomialInterface"> {
    let description = [{ }];
    let cppNamespace = "::mlir::quantum";
    
    let methods = [
        InterfaceMethod<[{
        }],
        /*retTy=*/"void",
        /*methodName=*/"inferResultPolynomial",
        /*args=*/(ins "::llvm::ArrayRef<::mlir::quantum::PhasePolynomial>":$argPolynomials,
            "::mlir::quantum::SetPolynomialFn":$setResultPolynomials),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            ::mlir::quantum::phasepolynomial::detail::defaultInferResultPolynomial($_op,
                                                         argPolynomials,
                                                         setResultPolynomials);
        }]>
    ];
}

#endif // MLIR_QUANTUM_INTERFACES_INFERPHASEPOLYNOMIALINTERFACE
