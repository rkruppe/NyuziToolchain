add_llvm_tool(FileCheck
  FileCheck.cpp
  )

target_link_libraries(FileCheck LLVMSupport)
