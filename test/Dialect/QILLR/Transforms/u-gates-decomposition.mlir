// RUN: quantum-opt --qillr-decompose-ugates %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @decompose_u3_to_zyz(
  // CHECK-SAME: %[[T:.+]]:{{.*}}, %[[P:.+]]:{{.*}}, %[[L:.+]]:{{.*}}) -> !qillr.qubit {
  func.func @decompose_u3_to_zyz(%theta : f64, %phi : f64, %lambda : f64) -> !qillr.qubit {
    // CHECK: %[[Q:.+]] = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
    %q = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
    // CHECK-DAG: "qillr.Rz"(%[[Q]], %[[P]]) <{index = [0]}> : (!qillr.qubit, f64) -> ()
    // CHECK-DAG: "qillr.Ry"(%[[Q]], %[[T]]) <{index = [0]}> : (!qillr.qubit, f64) -> ()
    // CHECK-DAG: "qillr.Rz"(%[[Q]], %[[L]]) <{index = [0]}> : (!qillr.qubit, f64) -> ()
    // CHECK-NOT: "qillr.U3"
    "qillr.U3"(%q, %theta, %phi, %lambda) <{index = [0]}> : (!qillr.qubit, f64, f64, f64) -> ()

    // CHECK-DAG: return %[[Q]] : !qillr.qubit
    return %q : !qillr.qubit
  }
}
