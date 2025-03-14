// RUN: quantum-opt %s --convert-quantum-to-qir --debug --debug-only=greedy-rewriter,dialect-conversion | FileCheck %s
// --debug-only=dialect-conversion

module {

    // CHECK-LABEL: func.func @return_single_qubit(
    // CHECK: ) {
    func.func @return_single_qubit() -> () {
        // CHECK-DAG: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-DAG: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-DAG: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> tensor<1xi1>
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
        // CHECK-DAG: "qir.reset"(%[[Q]]) : (!qir.qubit) -> ()
        "quantum.deallocate" (%q_m) : (!quantum.qubit<1>) -> ()
        // CHECK-NEXT: return
        func.return
    }

    // CHECK-LABEL: func.func @return_single_measurement_result(
    // CHECK: ) -> tensor<1xi1> {
    func.func @return_single_measurement_result() -> (tensor<1xi1>) {
        // CHECK-DAG: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-DAG: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-DAG: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> tensor<1xi1>
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
        // CHECK-DAG: "qir.reset"(%[[Q]]) : (!qir.qubit) -> ()
        "quantum.deallocate" (%q_m) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: return %[[M]]
        func.return %m : tensor<1xi1>
    }

    // CHECK-LABEL: func.func @convertHOp(
    func.func @convertHOp() -> () {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: "qir.H"(%[[Q]]) : (!qir.qubit) -> ()
        %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: "qir.reset"(%[[Q]]) : (!qir.qubit) -> ()
        "quantum.deallocate" (%q1) : (!quantum.qubit<1>) -> ()
        // CHECK-NEXT: return
        return
    }
}