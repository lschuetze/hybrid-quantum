// RUN: quantum-opt --phase-poly-merge -split-input-file %s | FileCheck %s

//                                          ┌───────────┐          
// q_0: ───────────────■─────────────────■──┤ Rz(0.987) ├──■───|0>─
//      ┌───────────┐┌─┴─┐┌───────────┐┌─┴─┐├───────────┤┌─┴─┐     
// q_1: ┤ Rz(0.321) ├┤ X ├┤ Rz(0.123) ├┤ X ├┤ Rz(0.567) ├┤ X ├─|0>─
//      └───────────┘└───┘└───────────┘└───┘└───────────┘└───┘     

// CHECK: module {
module {
// CHECK: qpu.module @test
qpu.module @test [#qpu.target<qubits = 3, coupling = [[0, 1], [1, 2]]>] {
// CHECK: "qpu.circuit"() <{function_type = (f64, f64, f64, f64) -> (), sym_name = "simple_rz_cancelling"}> ({
"qpu.circuit"() <{function_type = (f64, f64, f64, f64) -> (), sym_name = "simple_rz_cancelling"}>({
  // CHECK: ^bb0(%[[T1:.+]]: {{.*}}, %[[T2:.+]]: {{.*}}, %[[T3:.+]]: {{.*}}, %[[T4:.+]]: {{.*}}):
  ^bb0(%theta1: f64, %theta2: f64, %theta3: f64, %theta4: f64):
  // CHECK: %[[X0:.+]] = "quantum.alloc"() <{pos = 0 : i32}> : () -> !quantum.qubit<1>
  %x0 = "quantum.alloc"() <{pos = 0 : i32}> : () -> (!quantum.qubit<1>)
  // CHECK: %[[Y0:.+]] = "quantum.alloc"() <{pos = 1 : i32}> : () -> !quantum.qubit<1>
  %y0 = "quantum.alloc"() <{pos = 1 : i32}> : () -> (!quantum.qubit<1>)
  // CHECK: %[[P:.+]] = arith.addf %[[T1]], %[[T4]] : f64
  // CHECK: %[[Y1:.+]] = "quantum.Rz"(%[[Y0]], %[[P]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  %y1 = "quantum.Rz"(%y0, %theta1) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
  // CHECK: %[[X2:.+]], %[[Y2:.+]] = "quantum.CNOT"(%[[X0]], %[[Y1]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  %x2, %y2 = "quantum.CNOT" (%x0, %y1) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  // CHECK: %[[Y3:.+]] = "quantum.Rz"(%[[Y2]], %[[T2]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  %y3 = "quantum.Rz"(%y2, %theta2) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
  // CHECK: %[[X4:.+]], %[[Y4:.+]] = "quantum.CNOT"(%[[X2]], %[[Y3]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  %x4, %y4 = "quantum.CNOT" (%x2, %y3) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  // CHECK: %[[X5:.+]] = "quantum.Rz"(%[[X4]], %[[T3]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  %x5 = "quantum.Rz"(%x4, %theta3) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
  // CHECK-NOT: %[[Y5:.+]] = "quantum.Rz"(%[[Y4]], %[[T4]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  %y5 = "quantum.Rz"(%y4, %theta4) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
  // CHECK: %[[X6:.+]], %[[Y6:.+]] = "quantum.CNOT"(%[[X5]], %[[Y4]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)  
  %x6, %y6 = "quantum.CNOT" (%x5, %y5) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  // CHECK: "quantum.deallocate"(%[[X6]])
  "quantum.deallocate"(%x6) : (!quantum.qubit<1>) -> ()
  // CHECK: "quantum.deallocate"(%[[Y6]])
  "quantum.deallocate"(%y6) : (!quantum.qubit<1>) -> ()
  "qpu.return"() : () -> ()
}) : () -> ()
}
}

// -----
