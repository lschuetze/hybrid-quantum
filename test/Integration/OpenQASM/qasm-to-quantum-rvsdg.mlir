// RUN: %PYTHON qasm-import -i %s -r | FileCheck %s

// CHECK: "builtin.module"() ({
// CHECK:   "qpu.module"() <{sym_name = "qpu"}> ({
// CHECK:     "qpu.circuit"() <{function_type = () -> tensor<1xi1>, sym_name = "main"}> ({
// CHECK:      %[[v1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK:      %[[v2:.+]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:      %[[v3:.+]]:2 = "quantum.measure"(%[[v1]]) : (!quantum.qubit<1>) -> (!quantum.measurement<1>, !quantum.qubit<1>)
// CHECK:      %[[v4:.+]] = "quantum.to_tensor"(%[[v3]]#0) : (!quantum.measurement<1>) -> tensor<1xi1>
// CHECK:      %[[v6:.+]] = "tensor.insert_slice"(%[[v4]], %[[v2]]) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:      %[[v7:.+]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:      %[[v8:.+]] = "arith.cmpi"(%[[v6]], %[[v7]]) <{predicate = 0 : i64}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:      %[[v30:.+]] = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK:      %[[v31:.+]] = "tensor.extract"(%[[v8]], %[[v30]]) : (tensor<1xi1>, index) -> i1
// CHECK:      %[[v10:.+]] = "rvsdg.match"(%[[v31]]) <{mapping = [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>]}> : (i1) -> !rvsdg.ctrl<2>
// CHECK:      %[[v11:.+]] = "rvsdg.gamma"(%[[v10]], %[[v3]]#1) ({
// CHECK:      ^bb0(%[[arg3:.+]]: !quantum.qubit<1>):
// CHECK:        %[[v24:.+]] = "quantum.X"(%[[arg3]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:        "rvsdg.yield"(%[[v24]]) : (!quantum.qubit<1>) -> ()
// CHECK:      }, {
// CHECK:      ^bb0(%[[arg2:.+]]: !quantum.qubit<1>):
// CHECK:        "rvsdg.yield"(%[[arg2]]) : (!quantum.qubit<1>) -> ()
// CHECK:      }) : (!rvsdg.ctrl<2>, !quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:      %[[v12:.+]] = "arith.constant"() <{value = dense<false> : tensor<1xi1>}> : () -> tensor<1xi1>
// CHECK:      %[[v13:.+]] = "arith.cmpi"(%[[v6]], %[[v12]]) <{predicate = 0 : i64}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:      %[[v32:.+]] = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK:      %[[v33:.+]] = "tensor.extract"(%[[v13]], %[[v32]]) : (tensor<1xi1>, index) -> i1
// CHECK:      %[[v15:.+]] = "rvsdg.match"(%[[v33]]) <{mapping = [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>]}> : (i1) -> !rvsdg.ctrl<2>
// CHECK:      %[[v16:.+]] = "rvsdg.gamma"(%[[v15]], %[[v11]]) ({
// CHECK:      ^bb0(%[[arg1:.+]]: !quantum.qubit<1>):
// CHECK:        %[[v23:.+]] = "quantum.X"(%[[arg1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:        "rvsdg.yield"(%[[v23]]) : (!quantum.qubit<1>) -> ()
// CHECK:      }, {
// CHECK:      ^bb0(%[[arg0:.+]]: !quantum.qubit<1>):
// CHECK:        "rvsdg.yield"(%[[arg0]]) : (!quantum.qubit<1>) -> ()
// CHECK       }) : (!rvsdg.ctrl<2>, !quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:      %[[v17:.+]]:2 = "quantum.measure"(%[[v16]]) : (!quantum.qubit<1>) -> (!quantum.measurement<1>, !quantum.qubit<1>)
// CHECK:      %[[v18:.+]] = "quantum.to_tensor"(%[[v17]]#0) : (!quantum.measurement<1>) -> tensor<1xi1>
// CHECK:      %[[v20:.+]] = "tensor.insert_slice"(%[[v18]], %[[v6]]) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
// CHECK:      %[[v21:.+]] = "quantum.reset"(%[[v17]]#1) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK:      "quantum.deallocate"(%[[v21]]) : (!quantum.qubit<1>) -> ()
// CHECK:      %[[v22:.+]] = "tensor.from_elements"(%[[v20]]) : (tensor<1xi1>) -> tensor<1xi1>
// CHECK:      "qpu.return"(%[[v22]]) : (tensor<1xi1>) -> ()
// CHECK:    }) : () -> ()
// CHECK:  }) : () -> ()
// CHECK:  "func.func"() <{function_type = () -> tensor<1xi1>, sym_name = "qasm_main", sym_visibility = "public"}> ({
// CHECK:    %[[res:.+]] = "tensor.empty"() : () -> tensor<1xi1>
// CHECK:    "qpu.execute"(%[[res]]) <{circuit = @qpu::@main, operandSegmentSizes = array<i32: 0, 1>}> : (tensor<1xi1>) -> ()
// CHECK:    "func.return"(%[[res]]) : (tensor<1xi1>) -> ()
// CHECK:  }) : () -> ()
// CHECK: }) : () -> ()

OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
if(c==0) x q[0];
if(c==0) x q[0];
measure q[0] -> c[0];
reset q[0];
