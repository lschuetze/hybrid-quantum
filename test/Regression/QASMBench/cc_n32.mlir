// RUN: quantum-opt %s --debug --debug-only=dialect-conversion --convert-rvsdg-to-scf | FileCheck %s

module {
  qpu.module @qpu {
    // CHECK-LABEL: @main
    "qpu.circuit"() <{function_type = () -> tensor<32xi1>, sym_name = "main"}> ({
      %cst = arith.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]> : tensor<32xi1>
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %true = arith.constant true
      %cst_0 = arith.constant dense<false> : tensor<32xi1>
      %0 = "qillr.alloc"() <{size = 32 : i64}> : () -> !qillr.qubit
      "qillr.H"(%0) <{index = [0]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [1]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [2]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [3]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [4]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [5]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [6]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [7]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [8]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [9]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [10]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [11]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [12]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [13]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [14]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [15]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [16]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [17]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [18]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [19]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [20]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [21]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [22]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [23]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [24]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [25]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [26]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [27]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [28]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [29]}> : (!qillr.qubit) -> ()
      "qillr.H"(%0) <{index = [30]}> : (!qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [0], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [1], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [2], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [3], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [4], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [5], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [6], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [7], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [8], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [9], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [10], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [11], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [12], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [13], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [14], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [15], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [16], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [17], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [18], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [19], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [20], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [21], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [22], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [23], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [24], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [25], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [26], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [27], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [28], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [29], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      "qillr.CNOT"(%0, %0) <{controlIndex = [30], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
      %1 = "qillr.ralloc"() <{size = 32 : i64}> : () -> !qillr.result
      "qillr.measure"(%0, %1) <{inputIndex = [31], resultIndex = [31]}> : (!qillr.qubit, !qillr.result) -> ()
      %2 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      %3 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %4 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %3[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %5 = rvsdg.match(%4 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %6 = rvsdg.gamma(%5 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.X"(%arg0) <{index = [31]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %7 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %8 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %7[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %9 = rvsdg.match(%8 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %10 = rvsdg.gamma(%9 : <2>) (%6: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [31]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %11 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %12 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %11[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %13 = rvsdg.match(%12 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %14 = rvsdg.gamma(%13 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [0]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %15 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %16 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %15[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %17 = rvsdg.match(%16 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %18 = rvsdg.gamma(%17 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [1]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %19 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %20 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %19[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %21 = rvsdg.match(%20 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %22 = rvsdg.gamma(%21 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [2]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %23 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %24 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %23[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %25 = rvsdg.match(%24 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %26 = rvsdg.gamma(%25 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [3]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %27 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %28 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %27[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %29 = rvsdg.match(%28 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %30 = rvsdg.gamma(%29 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [4]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %31 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %32 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %31[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %33 = rvsdg.match(%32 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %34 = rvsdg.gamma(%33 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [5]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %35 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %36 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %35[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %37 = rvsdg.match(%36 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %38 = rvsdg.gamma(%37 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [6]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %39 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %40 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %39[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %41 = rvsdg.match(%40 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %42 = rvsdg.gamma(%41 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [7]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %43 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %44 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %43[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %45 = rvsdg.match(%44 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %46 = rvsdg.gamma(%45 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [8]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %47 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %48 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %47[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %49 = rvsdg.match(%48 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %50 = rvsdg.gamma(%49 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [9]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %51 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %52 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %51[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %53 = rvsdg.match(%52 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %54 = rvsdg.gamma(%53 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [10]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %55 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %56 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %55[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %57 = rvsdg.match(%56 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %58 = rvsdg.gamma(%57 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [11]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %59 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %60 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %59[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %61 = rvsdg.match(%60 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %62 = rvsdg.gamma(%61 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [12]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %63 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %64 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %63[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %65 = rvsdg.match(%64 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %66 = rvsdg.gamma(%65 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [13]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %67 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %68 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %67[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %69 = rvsdg.match(%68 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %70 = rvsdg.gamma(%69 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [14]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %71 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %72 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %71[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %73 = rvsdg.match(%72 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %74 = rvsdg.gamma(%73 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [15]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %75 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %76 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %75[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %77 = rvsdg.match(%76 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %78 = rvsdg.gamma(%77 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [16]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %79 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %80 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %79[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %81 = rvsdg.match(%80 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %82 = rvsdg.gamma(%81 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [17]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %83 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %84 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %83[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %85 = rvsdg.match(%84 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %86 = rvsdg.gamma(%85 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [18]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %87 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %88 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %87[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %89 = rvsdg.match(%88 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %90 = rvsdg.gamma(%89 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [19]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %91 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %92 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %91[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %93 = rvsdg.match(%92 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %94 = rvsdg.gamma(%93 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [20]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %95 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %96 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %95[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %97 = rvsdg.match(%96 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %98 = rvsdg.gamma(%97 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [21]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %99 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %100 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %99[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %101 = rvsdg.match(%100 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %102 = rvsdg.gamma(%101 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [22]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %103 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %104 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %103[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %105 = rvsdg.match(%104 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %106 = rvsdg.gamma(%105 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [23]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %107 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %108 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %107[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %109 = rvsdg.match(%108 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %110 = rvsdg.gamma(%109 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [24]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %111 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %112 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %111[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %113 = rvsdg.match(%112 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %114 = rvsdg.gamma(%113 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [25]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %115 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %116 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %115[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %117 = rvsdg.match(%116 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %118 = rvsdg.gamma(%117 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [26]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %119 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %120 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %119[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %121 = rvsdg.match(%120 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %122 = rvsdg.gamma(%121 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [27]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %123 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %124 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %123[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %125 = rvsdg.match(%124 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %126 = rvsdg.gamma(%125 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [28]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %127 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %128 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %127[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %129 = rvsdg.match(%128 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %130 = rvsdg.gamma(%129 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [29]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %131 = arith.cmpi eq, %2, %cst : tensor<32xi1>
      %132 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %131[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %133 = rvsdg.match(%132 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %134 = rvsdg.gamma(%133 : <2>) (%0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [30]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      "qillr.barrier"(%14, %18, %22, %26, %30, %34, %38, %42, %46, %50, %54, %58, %62, %66, %70, %74, %78, %82, %86, %90, %94, %98, %102, %106, %110, %114, %118, %122, %126, %130, %134, %10) <{indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
      %135 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %136 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %135[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %137 = rvsdg.match(%136 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %138:2 = rvsdg.gamma(%137 : <2>) (%38: !qillr.qubit, %10: !qillr.qubit) : [
        (%arg0: !qillr.qubit, %arg1: !qillr.qubit): {
          "qillr.CNOT"(%arg0, %arg1) <{controlIndex = [6], targetIndex = [31]}> : (!qillr.qubit, !qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit, %arg1: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit, %arg1: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit, %arg1: !qillr.qubit)
        }
      ] -> !qillr.qubit, !qillr.qubit
      "qillr.barrier"(%14, %18, %22, %26, %30, %34, %138#0, %42, %46, %50, %54, %58, %62, %66, %70, %74, %78, %82, %86, %90, %94, %98, %102, %106, %110, %114, %118, %122, %126, %130, %134, %138#1) <{indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}> : (!qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
      %139 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %140 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %139[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %141 = rvsdg.match(%140 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %142 = rvsdg.gamma(%141 : <2>) (%14: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [0]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %143 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %144 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %143[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %145 = rvsdg.match(%144 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %146 = rvsdg.gamma(%145 : <2>) (%18: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [1]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %147 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %148 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %147[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %149 = rvsdg.match(%148 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %150 = rvsdg.gamma(%149 : <2>) (%22: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [2]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %151 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %152 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %151[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %153 = rvsdg.match(%152 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %154 = rvsdg.gamma(%153 : <2>) (%26: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [3]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %155 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %156 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %155[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %157 = rvsdg.match(%156 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %158 = rvsdg.gamma(%157 : <2>) (%30: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [4]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %159 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %160 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %159[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %161 = rvsdg.match(%160 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %162 = rvsdg.gamma(%161 : <2>) (%34: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [5]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %163 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %164 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %163[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %165 = rvsdg.match(%164 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %166 = rvsdg.gamma(%165 : <2>) (%138#0: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [6]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %167 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %168 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %167[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %169 = rvsdg.match(%168 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %170 = rvsdg.gamma(%169 : <2>) (%42: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [7]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %171 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %172 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %171[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %173 = rvsdg.match(%172 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %174 = rvsdg.gamma(%173 : <2>) (%46: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [8]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %175 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %176 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %175[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %177 = rvsdg.match(%176 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %178 = rvsdg.gamma(%177 : <2>) (%50: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [9]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %179 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %180 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %179[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %181 = rvsdg.match(%180 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %182 = rvsdg.gamma(%181 : <2>) (%54: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [10]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %183 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %184 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %183[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %185 = rvsdg.match(%184 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %186 = rvsdg.gamma(%185 : <2>) (%58: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [11]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %187 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %188 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %187[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %189 = rvsdg.match(%188 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %190 = rvsdg.gamma(%189 : <2>) (%62: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [12]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %191 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %192 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %191[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %193 = rvsdg.match(%192 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %194 = rvsdg.gamma(%193 : <2>) (%66: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [13]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %195 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %196 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %195[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %197 = rvsdg.match(%196 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %198 = rvsdg.gamma(%197 : <2>) (%70: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [14]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %199 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %200 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %199[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %201 = rvsdg.match(%200 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %202 = rvsdg.gamma(%201 : <2>) (%74: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [15]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %203 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %204 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %203[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %205 = rvsdg.match(%204 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %206 = rvsdg.gamma(%205 : <2>) (%78: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [16]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %207 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %208 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %207[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %209 = rvsdg.match(%208 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %210 = rvsdg.gamma(%209 : <2>) (%82: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [17]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %211 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %212 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %211[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %213 = rvsdg.match(%212 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %214 = rvsdg.gamma(%213 : <2>) (%86: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [18]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %215 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %216 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %215[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %217 = rvsdg.match(%216 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %218 = rvsdg.gamma(%217 : <2>) (%90: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [19]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %219 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %220 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %219[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %221 = rvsdg.match(%220 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %222 = rvsdg.gamma(%221 : <2>) (%94: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [20]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %223 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %224 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %223[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %225 = rvsdg.match(%224 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %226 = rvsdg.gamma(%225 : <2>) (%98: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [21]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %227 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %228 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %227[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %229 = rvsdg.match(%228 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %230 = rvsdg.gamma(%229 : <2>) (%102: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [22]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %231 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %232 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %231[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %233 = rvsdg.match(%232 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %234 = rvsdg.gamma(%233 : <2>) (%106: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [23]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %235 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %236 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %235[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %237 = rvsdg.match(%236 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %238 = rvsdg.gamma(%237 : <2>) (%110: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [24]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %239 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %240 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %239[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %241 = rvsdg.match(%240 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %242 = rvsdg.gamma(%241 : <2>) (%114: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [25]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %243 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %244 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %243[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %245 = rvsdg.match(%244 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %246 = rvsdg.gamma(%245 : <2>) (%118: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [26]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %247 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %248 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %247[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %249 = rvsdg.match(%248 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %250 = rvsdg.gamma(%249 : <2>) (%122: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [27]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %251 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %252 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %251[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %253 = rvsdg.match(%252 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %254 = rvsdg.gamma(%253 : <2>) (%126: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [28]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %255 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %256 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %255[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %257 = rvsdg.match(%256 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %258 = rvsdg.gamma(%257 : <2>) (%130: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [29]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      %259 = arith.cmpi eq, %2, %cst_0 : tensor<32xi1>
      %260 = scf.parallel (%arg0) = (%c0) to (%c32) step (%c1) init (%true) -> i1 {
        %extracted = tensor.extract %259[%arg0] : tensor<32xi1>
        scf.reduce(%extracted, %true : i1, i1) {
        ^bb0(%arg1: i1, %arg2: i1):
          %294 = arith.andi %arg1, %arg2 : i1
          scf.reduce.return %294 : i1
        }
      }
      %261 = rvsdg.match(%260 : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
      %262 = rvsdg.gamma(%261 : <2>) (%134: !qillr.qubit) : [
        (%arg0: !qillr.qubit): {
          "qillr.H"(%arg0) <{index = [30]}> : (!qillr.qubit) -> ()
          rvsdg.yield (%arg0: !qillr.qubit)
        }, 
        (%arg0: !qillr.qubit): {
          rvsdg.yield (%arg0: !qillr.qubit)
        }
      ] -> !qillr.qubit
      "qillr.measure"(%142, %1) <{inputIndex = [0], resultIndex = [0]}> : (!qillr.qubit, !qillr.result) -> ()
      %263 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%146, %1) <{inputIndex = [1], resultIndex = [1]}> : (!qillr.qubit, !qillr.result) -> ()
      %264 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%150, %1) <{inputIndex = [2], resultIndex = [2]}> : (!qillr.qubit, !qillr.result) -> ()
      %265 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%154, %1) <{inputIndex = [3], resultIndex = [3]}> : (!qillr.qubit, !qillr.result) -> ()
      %266 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%158, %1) <{inputIndex = [4], resultIndex = [4]}> : (!qillr.qubit, !qillr.result) -> ()
      %267 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%162, %1) <{inputIndex = [5], resultIndex = [5]}> : (!qillr.qubit, !qillr.result) -> ()
      %268 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%166, %1) <{inputIndex = [6], resultIndex = [6]}> : (!qillr.qubit, !qillr.result) -> ()
      %269 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%170, %1) <{inputIndex = [7], resultIndex = [7]}> : (!qillr.qubit, !qillr.result) -> ()
      %270 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%174, %1) <{inputIndex = [8], resultIndex = [8]}> : (!qillr.qubit, !qillr.result) -> ()
      %271 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%178, %1) <{inputIndex = [9], resultIndex = [9]}> : (!qillr.qubit, !qillr.result) -> ()
      %272 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%182, %1) <{inputIndex = [10], resultIndex = [10]}> : (!qillr.qubit, !qillr.result) -> ()
      %273 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%186, %1) <{inputIndex = [11], resultIndex = [11]}> : (!qillr.qubit, !qillr.result) -> ()
      %274 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%190, %1) <{inputIndex = [12], resultIndex = [12]}> : (!qillr.qubit, !qillr.result) -> ()
      %275 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%194, %1) <{inputIndex = [13], resultIndex = [13]}> : (!qillr.qubit, !qillr.result) -> ()
      %276 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%198, %1) <{inputIndex = [14], resultIndex = [14]}> : (!qillr.qubit, !qillr.result) -> ()
      %277 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%202, %1) <{inputIndex = [15], resultIndex = [15]}> : (!qillr.qubit, !qillr.result) -> ()
      %278 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%206, %1) <{inputIndex = [16], resultIndex = [16]}> : (!qillr.qubit, !qillr.result) -> ()
      %279 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%210, %1) <{inputIndex = [17], resultIndex = [17]}> : (!qillr.qubit, !qillr.result) -> ()
      %280 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%214, %1) <{inputIndex = [18], resultIndex = [18]}> : (!qillr.qubit, !qillr.result) -> ()
      %281 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%218, %1) <{inputIndex = [19], resultIndex = [19]}> : (!qillr.qubit, !qillr.result) -> ()
      %282 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%222, %1) <{inputIndex = [20], resultIndex = [20]}> : (!qillr.qubit, !qillr.result) -> ()
      %283 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%226, %1) <{inputIndex = [21], resultIndex = [21]}> : (!qillr.qubit, !qillr.result) -> ()
      %284 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%230, %1) <{inputIndex = [22], resultIndex = [22]}> : (!qillr.qubit, !qillr.result) -> ()
      %285 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%234, %1) <{inputIndex = [23], resultIndex = [23]}> : (!qillr.qubit, !qillr.result) -> ()
      %286 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%238, %1) <{inputIndex = [24], resultIndex = [24]}> : (!qillr.qubit, !qillr.result) -> ()
      %287 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%242, %1) <{inputIndex = [25], resultIndex = [25]}> : (!qillr.qubit, !qillr.result) -> ()
      %288 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%246, %1) <{inputIndex = [26], resultIndex = [26]}> : (!qillr.qubit, !qillr.result) -> ()
      %289 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%250, %1) <{inputIndex = [27], resultIndex = [27]}> : (!qillr.qubit, !qillr.result) -> ()
      %290 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%254, %1) <{inputIndex = [28], resultIndex = [28]}> : (!qillr.qubit, !qillr.result) -> ()
      %291 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%258, %1) <{inputIndex = [29], resultIndex = [29]}> : (!qillr.qubit, !qillr.result) -> ()
      %292 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.measure"(%262, %1) <{inputIndex = [30], resultIndex = [30]}> : (!qillr.qubit, !qillr.result) -> ()
      %293 = "qillr.read_measurement"(%1) <{index = []}> : (!qillr.result) -> tensor<32xi1>
      "qillr.deallocate"(%142) <{index = [0]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%146) <{index = [1]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%150) <{index = [2]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%154) <{index = [3]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%158) <{index = [4]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%162) <{index = [5]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%166) <{index = [6]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%170) <{index = [7]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%174) <{index = [8]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%178) <{index = [9]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%182) <{index = [10]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%186) <{index = [11]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%190) <{index = [12]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%194) <{index = [13]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%198) <{index = [14]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%202) <{index = [15]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%206) <{index = [16]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%210) <{index = [17]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%214) <{index = [18]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%218) <{index = [19]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%222) <{index = [20]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%226) <{index = [21]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%230) <{index = [22]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%234) <{index = [23]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%238) <{index = [24]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%242) <{index = [25]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%246) <{index = [26]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%250) <{index = [27]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%254) <{index = [28]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%258) <{index = [29]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%262) <{index = [30]}> : (!qillr.qubit) -> ()
      "qillr.deallocate"(%138#1) <{index = [31]}> : (!qillr.qubit) -> ()
      "qpu.return"(%293) : (tensor<32xi1>) -> ()
    }) : () -> ()
  }
  func.func public @qasm_main() -> tensor<32xi1> {
    %0 = tensor.empty() : tensor<32xi1>
    qpu.execute @qpu::@main  outs(%0 : tensor<32xi1>)
    return %0 : tensor<32xi1>
  }
}
