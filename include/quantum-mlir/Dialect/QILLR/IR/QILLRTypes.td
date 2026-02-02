//===- Types.td - QILLR dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the QILLR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QILLR_TYPES
#define QILLR_TYPES

include "quantum-mlir/Dialect/QILLR/IR/QILLRAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def QILLR_QubitType : QILLR_Type<"Qubit"> {
  let mnemonic = "qubit";
  let summary = "qubit type";
  let parameters = (ins "int64_t":$size);
  let assemblyFormat = "`<` $size `>`";
}

def QILLR_ResultType : QILLR_Type<"Result"> {
  let mnemonic = "result";
  let summary = "A quantum measurement result.";
  let parameters = (ins "int64_t":$size);
  let assemblyFormat = "`<` $size `>`";
}

def QILLR_MeasurementType : 1DTensorOf<[I1]>;

#endif
