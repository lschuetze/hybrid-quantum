// RUN: quantum-opt %s --debug --mlir-print-ir-after-all --convert-quantum-to-qillr -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @rvdsg_gamma(
func.func @rvdsg_gamma(%b : i1) {
    // CHECK: %[[PRED:.*]] = rvsdg.match
    %predicate = rvsdg.match(%b : i1) [
            #rvsdg.matchRule<0 -> 1>,
            #rvsdg.matchRule<1 -> 0>
    ] -> !rvsdg.ctrl<2>
    // CHECK: %[[Q:.*]] = "qillr.alloc"() : () -> !qillr.qubit
    %q = "quantum.alloc"() : () -> !quantum.qubit<1>
    // CHECK: rvsdg.gamma(%[[PRED]] : <2>) (%[[Q]]: !qillr.qubit)
    %q2 = rvsdg.gamma (%predicate : !rvsdg.ctrl<2>) (%q : !quantum.qubit<1>):[
        // CHECK: (%[[Q1:.*]]: !qillr.qubit): {
        (%q: !quantum.qubit<1>):{
            // CHECK: "qillr.X"(%[[Q1]]) : (!qillr.qubit) -> ()
            %qX = "quantum.X"(%q): (!quantum.qubit<1>) -> !quantum.qubit<1>
            // CHECK: rvsdg.yield (%[[Q1]]: !qillr.qubit)
            rvsdg.yield (%qX:!quantum.qubit<1>)
        },
        // CHECK: (%[[Q2:.*]]: !qillr.qubit): {
        (%q: !quantum.qubit<1>):{
            // CHECK: "qillr.Z"(%[[Q2]]) : (!qillr.qubit) -> ()
            %qZ = "quantum.Z"(%q): (!quantum.qubit<1>) -> !quantum.qubit<1>
            // CHECK: rvsdg.yield (%[[Q2]]: !qillr.qubit)
            rvsdg.yield (%qZ:!quantum.qubit<1>)
        }
    // CHECK: ] -> !qillr.qubit
    ] -> !quantum.qubit<1>

    return
}

 //-------
