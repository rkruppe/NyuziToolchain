//===-- NyuziTargetTransformInfo.h - Nyuzi specific TTI ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NYUZI_NYUZITARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_NYUZI_NYUZITARGETTRANSFORMINFO_H

#include "NyuziTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class NyuziTTIImpl : public BasicTTIImplBase<NyuziTTIImpl> {
  typedef BasicTTIImplBase<NyuziTTIImpl> BaseT;
  friend BaseT;

  const NyuziSubtarget *ST;
  const NyuziTargetLowering *TLI;

  const NyuziSubtarget *getST() const { return ST; }
  const NyuziTargetLowering *getTLI() const { return TLI; }

public:
  explicit NyuziTTIImpl(const NyuziTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool isLegalMaskedScatter(Type *DataType) const;
  bool isLegalMaskedGather(Type *DataType) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NYUZI_NYUZITARGETTRANSFORMINFO_H
