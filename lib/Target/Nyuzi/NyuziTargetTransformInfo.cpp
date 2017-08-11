//===-- NyuziTargetTransformInfo.cpp - Nyuzi specific TTI -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NyuziTargetTransformInfo.h"

namespace {
bool isLegalScatterGatherType(llvm::Type *Ty) {
  auto EltTy = Ty->getVectorElementType();
  // We don't have a DataLayout, so we need to explictly recognize pointers.
  // FIXME are there other types that are 32 bits but not covered by this?
  return EltTy->getPrimitiveSizeInBits() == 32 || EltTy->isPointerTy();
}
}

namespace llvm {
bool NyuziTTIImpl::isLegalMaskedScatter(Type *DataType) const {
  return isLegalScatterGatherType(DataType);
}
bool NyuziTTIImpl::isLegalMaskedGather(Type *DataType) const {
  return isLegalScatterGatherType(DataType);
}
} // namespace llvm
