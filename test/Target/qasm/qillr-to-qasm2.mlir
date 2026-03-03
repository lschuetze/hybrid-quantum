// RUN: quantum-translate -split-input-file --mlir-to-openqasm %s | FileCheck %s

// CHECK: OPENQASM 2.0;
// CHECK-NEXT: include "qelib1.inc";
// CHECK-DAG: qreg [[q0:q.+]][3]; 
%q0 = "qillr.alloc"() <{size = 3 : i64}> : () -> (!qillr.qubit)
// CHECK-NEXT: creg [[c0:c.+]][2];
%r0 = "qillr.ralloc"() <{size = 2 : i64}> : () -> (!qillr.result)
// Basic gates and operations
// CHECK-NEXT: h [[q0]][0];
"qillr.H"(%q0) <{index = [0]}> : (!qillr.qubit) -> ()
// CHECK-NEXT: x [[q0]][1];
"qillr.X"(%q0) <{index = [1]}> : (!qillr.qubit) -> ()
// CHECK-NOT: arith.constant
%c1 = arith.constant 0.1 : f64
%c2 = arith.constant 0.2 : f64
%c3 = arith.constant 0.3 : f64
// CHECK-NEXT: u3({{.+}},{{.+}},{{.+}}) [[q0]][2];
"qillr.U3"(%q0, %c1, %c2, %c3) <{index = [2]}> : (!qillr.qubit, f64, f64, f64) -> ()
// CHECK-NEXT: cx [[q0]][0], [[q0]][1];
"qillr.CNOT"(%q0, %q0) <{controlIndex = [0], targetIndex = [1]}> : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-NEXT: z [[q0]][0];
"qillr.Z"(%q0) <{index = [0]}>: (!qillr.qubit) -> ()
// CHECK-NEXT: y [[q0]][1];
"qillr.Y"(%q0) <{index = [1]}>: (!qillr.qubit) -> ()
// CHECK-NEXT: rx({{.*}}) [[q0]][2];
"qillr.Rx"(%q0, %c1) <{index = [2]}> : (!qillr.qubit, f64) -> ()
// CHECK-NEXT: ry({{.*}}) [[q0]][0];
"qillr.Ry"(%q0, %c2) <{index = [0]}> : (!qillr.qubit, f64) -> ()
// CHECK-NEXT: rz({{.*}}) [[q0]][1];
"qillr.Rz"(%q0, %c3) <{index = [1]}> : (!qillr.qubit, f64) -> ()
// CHECK-NEXT: u2({{.*}},{{.*}}) [[q0]][2];
"qillr.U2"(%q0, %c1, %c2) <{index = [2]}> : (!qillr.qubit, f64, f64) -> ()
// CHECK-NEXT: u1({{.*}}) [[q0]][0];
"qillr.U1"(%q0, %c3) <{index = [0]}> : (!qillr.qubit, f64) -> ()
// CHECK-NEXT: s [[q0]][1];
"qillr.S"(%q0) <{index = [1]}> : (!qillr.qubit) -> ()
// CHECK-NEXT: sdg [[q0]][2];
"qillr.Sdg"(%q0) <{index = [2]}>: (!qillr.qubit) -> ()
// CHECK-NEXT: t [[q0]][0];
"qillr.T"(%q0) <{index = [0]}>: (!qillr.qubit) -> ()
// CHECK-NEXT: tdg [[q0]][1];
"qillr.Tdg"(%q0) <{index = [1]}>: (!qillr.qubit) -> ()
// Extended multi-qubit gates
// CHECK-NEXT: cz [[q0]][0], [[q0]][2];
"qillr.Cz"(%q0, %q0) <{controlIndex = [0], targetIndex = [2]}>: (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-NEXT: crz({{.*}}) [[q0]][1], [[q0]][0];
"qillr.CRz"(%q0, %q0, %c3) <{controlIndex = [1], targetIndex = [0]}> : (!qillr.qubit, !qillr.qubit, f64) -> ()
// CHECK-NEXT: cry({{.*}}) [[q0]][2], [[q0]][1];
"qillr.CRy"(%q0, %q0, %c2) <{controlIndex = [2], targetIndex = [1]}> : (!qillr.qubit, !qillr.qubit, f64) -> ()
// CHECK-NEXT: ccx [[q0]][0], [[q0]][1], [[q0]][2];
"qillr.CCX"(%q0, %q0, %q0) <{control1Index = [0], control2Index = [1], targetIndex = [2]}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK-NEXT: barrier [[q0]][0], [[q0]][1], [[q0]][2];
"qillr.barrier"(%q0, %q0, %q0) <{indices = [0, 1, 2]}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK-NEXT: swap [[q0]][1], [[q0]][2];
"qillr.swap"(%q0, %q0) <{lhsIndex = [1], rhsIndex = [2]}> : (!qillr.qubit, !qillr.qubit) -> ()
// Measurement utilities
// CHECK-NEXT: measure [[q0]][2] -> [[c0]][1];
"qillr.measure"(%q0, %r0) <{inputIndex = [2], resultIndex = [1]}> : (!qillr.qubit, !qillr.result) -> ()
// CHECK-NOT: "qillr.read_measurement"
%mread = "qillr.read_measurement"(%r0) <{index = [1]}> : (!qillr.result) -> tensor<2xi1>
// CHECK-NEXT: reset [[q0]][2];
"qillr.reset"(%q0) <{index = [2]}> : (!qillr.qubit) -> ()

// -----

// CHECK: qreg [[q:q.+]][1]; 
%q = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
// CHECK-NEXT: creg [[c:c.+]][1];
%r = "qillr.ralloc"() <{size = 1 : i64}> : () -> !qillr.result
 // CHECK-NEXT: measure [[q]][0] -> [[c]][0];
"qillr.measure"(%q, %r) <{inputIndex = [0], resultIndex = [0]}> : (!qillr.qubit, !qillr.result) -> ()
// CHECK-NOT: "qillr.read_measurement"
%mt = "qillr.read_measurement"(%r) <{index = [0]}> : (!qillr.result) -> tensor<1xi1>
%cst = arith.constant dense<true> : tensor<1xi1>
%cmp = arith.cmpi eq, %mt, %cst : tensor<1xi1>
%c0 = arith.constant 0 : index
%b = tensor.extract %cmp[%c0] : tensor<1xi1>
// CHECK: if([[c]]==1)
scf.if %b {
  // CHECK: x [[q]][0];
  "qillr.X"(%q) <{index = [0]}> : (!qillr.qubit) -> ()
}
// CHECK-NOT: "qillr.deallocate"
"qillr.deallocate"(%q) <{index = [0]}> : (!qillr.qubit) -> ()
