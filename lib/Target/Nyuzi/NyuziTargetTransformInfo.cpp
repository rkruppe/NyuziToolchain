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
  return Ty->isVectorTy() && Ty->getScalarSizeInBits() == 32;
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
