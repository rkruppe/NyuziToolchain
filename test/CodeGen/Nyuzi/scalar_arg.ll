; RUN: llc %s -o - | FileCheck %s
;
; Test passing scalar arguments to a called function.
;

target triple = "nyuzi-elf-none"

define i32 @somefunc(i32 %arg1, i32 %arg2, i32 %arg3,
  i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8,
  i32 %arg9, i32 %arg10) { ; CHECK-LABEL: somefunc:

  %result = alloca i32

  ; Technically this stack allocation is not necessary since there are no spills.
  ; This might break if the backend implementation changes. The parameter offsets
  ; for the loads depend on this.

  ; CHECK: add_i sp, sp, -64

  %1 = add i32 %arg1, %arg2

  ; CHECK-DAG: add_i [[RES1:s[0-9]+]], s0, s1

  %2 = add i32 %1, %arg3

  ; CHECK-DAG: add_i [[RES2:s[0-9]+]], [[RES1]], s2

  %3 = add i32 %2, %arg4

  ; CHECK-DAG: add_i [[RES3:s[0-9]+]], [[RES2]], s3

  %4 = add i32 %3, %arg5

  ; CHECK-DAG: add_i [[RES4:s[0-9]+]], [[RES3]], s4

  %5 = add i32 %4, %arg6

  ; CHECK-DAG: add_i [[RES5:s[0-9]+]], [[RES4]], s5

  %6 = add i32 %5, %arg7

  ; CHECK-DAG: add_i [[RES6:s[0-9]+]], [[RES5]], s6

  %7 = add i32 %6, %arg8

  ; CHECK-DAG: add_i [[RES7:s[0-9]+]], [[RES6]], s7

  %8 = add i32 %7, %arg9

  ; Check that these are loaded from the stack
  ; CHECK-DAG: load_32 [[TMPREG1:s[0-9]+]], 64(sp)
  ; CHECK-DAG: add_i [[RES7:s[0-9]+]], [[RES6]], [[TMPREG1]]

  %9 = add i32 %8, %arg10

  ; CHECK-DAG: load_32 [[TMPREG2:s[0-9]+]], 68(sp)
  ; CHECK-DAG: add_i s{{[0-9]+}}, [[RES7]], [[TMPREG2]]

  ret i32 %9
}

define i32 @main() {  ; CHECK-LABEL: main:

  %result = call i32 @somefunc(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10)

  ; The first eight arguments will be passed in registers. The remainders will
  ; be copied onto the stack.

  ; CHECK-DAG: move [[PARAMTMP1:s[0-9]+]], 10
  ; CHECK-DAG: store_32 [[PARAMTMP1]], 4(sp)
  ; CHECK-DAG: move [[PARAMTMP2:s[0-9]+]], 9
  ; CHECK-DAG: store_32 [[PARAMTMP2]], (sp)
  ; CHECK-DAG: move s0, 1
  ; CHECK-DAG: move s1, 2
  ; CHECK-DAG: move s2, 3
  ; CHECK-DAG: move s3, 4
  ; CHECK-DAG: move s4, 5
  ; CHECK-DAG: move s5, 6
  ; CHECK-DAG: move s6, 7
  ; CHECK-DAG: move s7, 8
  ; CHECK: call somefunc

  ret i32 %result
}
