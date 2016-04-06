//===-- MipsAsmPrinter.h - Mips LLVM Assembly Printer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Nyuzi Assembly printer class.
//
// The name is a bit misleading.  Because we use the MC layer for code
// generation, the job of this class is now mostly to convert MachineInstrs
// into MCInsts. Most of the work is done by a helper class MCInstLowering
// (which in turn uses code generated by TableGen).  This also performs
// replacement of inline assembly parameters.
//
//===----------------------------------------------------------------------===//

#ifndef NYUZIASMPRINTER_H
#define NYUZIASMPRINTER_H

#include "NyuziMCInstLower.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class MCStreamer;
class MachineInstr;
class MachineBasicBlock;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY NyuziAsmPrinter : public AsmPrinter {
  NyuziMCInstLower MCInstLowering;

public:
  explicit NyuziAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), MCInstLowering(*this) {}

  const char *getPassName() const override { return "Nyuzi Assembly Printer"; }

  void EmitInstruction(const MachineInstr *MI) override;
  void EmitFunctionBodyStart() override;
  void EmitConstantPool() override;

  // Print operand for inline assembly
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O) override;

private:
  MCSymbol *GetJumpTableLabel(unsigned uid) const;
  void EmitInlineJumpTable(const MachineInstr *MI);
};
}

#endif
