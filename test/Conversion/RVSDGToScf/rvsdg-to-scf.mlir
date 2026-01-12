// RUN: quantum-opt %s --debug --mlir-print-ir-after-all -split-input-file -convert-rvsdg-to-scf | FileCheck %s

// CHECK-LABEL: func.func @rvsdg_gamma_to_if(
// CHECK: %[[B:.*]]: i1)
func.func @rvsdg_gamma_to_if(%b : i1) -> i1 {
  // CHECK-NOT: rvsdg.match
  %predicate = rvsdg.match(%b : i1) [
          #rvsdg.matchRule<0 -> 1>,
          #rvsdg.matchRule<1 -> 0>
  ] -> !rvsdg.ctrl<2>
  // CHECK: %[[BRES:.*]] = scf.if %[[B]] -> (i1) {
  // CHECK: %[[T:.*]] = arith.constant true
  // CHECK: %[[NOTB:.*]] = arith.xori %[[B]], %[[T]] : i1
  // CHECK: scf.yield %[[NOTB]] : i1
  // CHECK: } else {
  // CHECK: scf.yield %[[B]] : i1
  // CHECK: }
  %b2 = rvsdg.gamma (%predicate : !rvsdg.ctrl<2>) (%b : i1):[
      (%arg1 : i1):{
          %1 = arith.constant 1 : i1
          %notb = arith.xori %arg1, %1 : i1
          rvsdg.yield (%notb : i1)
      },
      (%arg2 : i1):{
          rvsdg.yield (%arg2 : i1)
      }
  ] -> i1
  // CHECK: return %[[BRES]] : i1
  return %b2 : i1
}

// -----

// CHECK-LABEL: func.func @rvsdg_gamma_to_if_qillr(
// CHECK: %[[B:.*]]: i1)
func.func @rvsdg_gamma_to_if_qillr(%b : i1) {
  // CHECK-NOT: rvsdg.match
  %predicate = rvsdg.match(%b : i1) [
          #rvsdg.matchRule<0 -> 1>,
          #rvsdg.matchRule<1 -> 0>
  ] -> !rvsdg.ctrl<2>
  // CHECK: %[[Q:.*]] = "qillr.alloc"() : () -> !qillr.qubit
  %q = "qillr.alloc"() : () -> !qillr.qubit
  // CHECK: scf.if %[[B]] {
  // CHECK: "qillr.X"(%[[Q]]) : (!qillr.qubit) -> ()
  // CHECK: }
  %q2 = rvsdg.gamma (%predicate : !rvsdg.ctrl<2>) (%q : !qillr.qubit):[
      (%arg1 : !qillr.qubit):{
          "qillr.X"(%arg1) : (!qillr.qubit) -> ()
          rvsdg.yield (%arg1 : !qillr.qubit)
      },
      (%arg2 : !qillr.qubit):{
          rvsdg.yield (%arg2 : !qillr.qubit)
      }
  ] -> !qillr.qubit
  // CHECK: "qillr.deallocate"(%[[Q]]) : (!qillr.qubit) -> ()
  "qillr.deallocate" (%q2) : (!qillr.qubit) -> ()
  // CHECK: return
  return
}

// -----
