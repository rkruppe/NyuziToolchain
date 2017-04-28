target triple = "nyuzi-elf-none"

define void @fact.wrapper() {
  tail call void @llvm.spmd.call(i8* bitcast (i32 (i8*)* @fact to i8*), i8* null)
  ret void
}

declare void @llvm.spmd.call(i8*, i8*) #0

attributes #0 = { nounwind }

define i32 @fact(i8* %n.ptr) {
entry:
  %n.ptr32 = bitcast i8* %n.ptr to i32*
  %n = load i32, i32* %n.ptr32
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %loop]
  %i.next = add i32 %i, 1
  %cond = icmp ult i32 %i, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %i
}
