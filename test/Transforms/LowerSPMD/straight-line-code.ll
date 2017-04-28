; RUN: opt < %s  -lowerspmd -S | FileCheck %s
target triple = "nyuzi-elf-none"

; TODO make this unnecessary by vectorizing all functions
define void @wrapper() {
  tail call void @llvm.spmd.call(i8* bitcast (void (i8*)* @kernel to i8*), i8* null)
  ret void
}

declare void @llvm.spmd.call(i8*, i8*)

define void @kernel(i8*) {
  ; CHECK-LABEL: define void @kernel.vector(<16 x i1>, <16 x i8*>)
  ; CHECK: bitcast <16 x i8*> %1 to <16 x i32*>
  ; CHECK: call <16 x i32> @llvm.masked.gather.v16i32
  ; CHECK: add <16 x i32> %old, <i32 1, i32 1,
  ; CHECK: call void @llvm.masked.scatter.v16i32
  ; CHECK: ret void
  %ptr = bitcast i8* %0 to i32*
  %old = load i32, i32* %ptr, align 4
  %new = add i32 %old, 1
  store i32 %new, i32* %ptr, align 4
  ret void
}
