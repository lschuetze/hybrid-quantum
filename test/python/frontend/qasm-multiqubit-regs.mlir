// RUN: %PYTHON qasm-import -i %s -r | FileCheck %s

// CHECK: "builtin.module"() ({
// CHECK:   "qpu.module"() <{sym_name = "qpu"}> ({
// CHECK:     "qpu.circuit"() <{function_type = () -> tensor<2xi1>, sym_name = "main"}> ({
// CHECK:       %[[q1:.*]] = "quantum.alloc"() : () -> !quantum.qubit<2>
// CHECK:       %[[q2:.*]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:       %[[q3:.*]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:       %[[q4:.*]]:2 = "quantum.split"(%[[q1]]) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
// CHECK:       %[[q5:.*]]:2 = "quantum.measure"(%[[q4]]#0) : (!quantum.qubit<1>) -> (!quantum.measurement<1>, !quantum.qubit<1>)
// CHECK:       %[[q6:.*]] = "quantum.to_tensor"(%[[q5]]#0) : (!quantum.measurement<1>) -> tensor<1xi1>
// CHECK:       %[[q8:.*]] = "tensor.insert_slice"(%[[q6]], %[[q2]]) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:       %[[q9:.*]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:       %[[q10:.*]] = "arith.cmpi"(%[[q8]], %[[q9]]) <{predicate = 0 : i64}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:       %[[q100:.+]] = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK:       %[[q11:.+]] = "tensor.extract"(%[[q10]], %[[q100]]) : (tensor<1xi1>, index) -> i1 
// CHECK:       %[[q12:.*]] = "rvsdg.match"(%[[q11]]) <{mapping = [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>]}> : (i1) -> !rvsdg.ctrl<2>
// CHECK:       %[[q13:.*]] = "rvsdg.gamma"(%[[q12]], %[[q5]]#1) ({
// CHECK:       ^bb0(%[[arg3:.+]]: !quantum.qubit<1>):
// CHECK:         %[[q27:.*]] = "quantum.X"(%[[arg3]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:         "rvsdg.yield"(%[[q27]]) : (!quantum.qubit<1>) -> ()
// CHECK:       }, {
// CHECK:       ^bb0(%[[arg2:.+]]: !quantum.qubit<1>):
// CHECK:         "rvsdg.yield"(%[[arg2]]) : (!quantum.qubit<1>) -> ()
// CHECK:       }) : (!rvsdg.ctrl<2>, !quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:       %[[q14:.*]]:2 = "quantum.measure"(%[[q4]]#1) : (!quantum.qubit<1>) -> (!quantum.measurement<1>, !quantum.qubit<1>)
// CHECK:       %[[q15:.*]] = "quantum.to_tensor"(%[[q14]]#0) : (!quantum.measurement<1>) -> tensor<1xi1>
// CHECK:       %[[q17:.*]] = "tensor.insert_slice"(%[[q15]], %[[q3]]) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:       %[[q18:.*]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:       %[[q19:.*]] = "arith.cmpi"(%[[q17]], %[[q18]]) <{predicate = 0 : i64}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:       %[[q200:.+]] = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK:       %[[q20:.+]] = "tensor.extract"(%[[q19]], %[[q200]]) : (tensor<1xi1>, index) -> i1 
// CHECK:       %[[q21:.*]] = "rvsdg.match"(%[[q20]]) <{mapping = [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>]}> : (i1) -> !rvsdg.ctrl<2>
// CHECK:       %[[q22:.*]] = "rvsdg.gamma"(%[[q21]], %[[q14]]#1) ({
// CHECK:       ^bb0(%[[arg1:.+]]: !quantum.qubit<1>):
// CHECK:         %[[q26:.*]] = "quantum.X"(%[[arg1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:         "rvsdg.yield"(%[[q26]]) : (!quantum.qubit<1>) -> ()
// CHECK:       }, {
// CHECK:       ^bb0(%[[arg0:.+]]: !quantum.qubit<1>):
// CHECK:         "rvsdg.yield"(%[[arg0]]) : (!quantum.qubit<1>) -> ()
// CHECK:       }) : (!rvsdg.ctrl<2>, !quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:       %[[q23:.*]] = "quantum.reset"(%[[q13]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:       %[[q24:.*]] = "quantum.reset"(%[[q22]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:       "quantum.deallocate"(%[[q23]]) : (!quantum.qubit<1>) -> ()
// CHECK:       "quantum.deallocate"(%[[q24]]) : (!quantum.qubit<1>) -> ()
// CHECK:       %[[q25:.*]] = "tensor.from_elements"(%[[q8]], %[[q17]]) : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
// CHECK:       "qpu.return"(%[[q25]]) : (tensor<2xi1>) -> ()
// CHECK:     }) : () -> ()
// CHECK:   }) : () -> ()
// CHECK:   "func.func"() <{function_type = () -> tensor<2xi1>, sym_name = "qasm_main", sym_visibility = "public"}> ({
// CHECK:     %[[q0:.*]] = "tensor.empty"() : () -> tensor<2xi1>
// CHECK:     "qpu.execute"(%[[q0]]) <{circuit = @qpu::@main, operandSegmentSizes = array<i32: 0, 1>}> : (tensor<2xi1>) -> ()
// CHECK:     "func.return"(%[[q0]]) : (tensor<2xi1>) -> ()
// CHECK:   }) : () -> ()
// CHECK: }) : () -> ()


OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c1[1];
creg c2[1];
measure q[0] -> c1[0];
if(c1==0) x q[0];
measure q[1] -> c2[0];
if(c2==0) x q[1];
reset q[0];
reset q[1];
