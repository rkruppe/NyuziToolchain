target triple = "nyuzi-elf-none"

define void @fact.wrapper() {
  tail call void @llvm.spmd.call(i8* bitcast (void (i8*)* @kernel to i8*), i8* null)
  ret void
}

declare void @llvm.spmd.call(i8*, i8*) #0

attributes #0 = { nounwind }

define void @kernel(i8*) {
  %ptr = bitcast i8* %0 to i32*
  %old = load i32, i32* %ptr, align 4
  %new = tail call i32 @transform(i32 %old)
  store i32 %new, i32* %ptr, align 4
  ret void
}

define internal i32 @transform(i32) {
bb1:
	%is1 = icmp eq i32 %0, 1
	%is2 = icmp eq i32 %0, 2
	br i1 %is1, label %return, label %bb2
bb2:
	br i1 %is2, label %return, label %other
other:
	br label %return
return:
	%result = phi i32 [42, %bb1], [-42, %bb2], [0, %other]
	ret i32 %result
}
