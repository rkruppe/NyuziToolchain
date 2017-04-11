#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

#define DEBUG_TYPE "lowerspmd"

namespace {

const unsigned SIMD_WIDTH = 16;
const char *CONDITION_METADATA_ID = "spmd-conditional";

template <typename T> using BBMap = DenseMap<BasicBlock *, T>;

struct Conditions {
  Function &F;
  BBMap<Value *> ForBlock;
  BBMap<BBMap<Value *>> ForEdge;

  Conditions(Function &F) : F(F) {}

  void addEdge(BasicBlock *From, BasicBlock *To, Value *Cond) {
    assert(ForEdge[From].count(To) == 0 && "Added edge twice");
    ForEdge[From][To] = Cond;
  }

  Value *&forBlock(BasicBlock *BB) {
    assert(BB->getParent() == &F && "query for BB from wrong function");
    return ForBlock[BB];
  }
  Value *forEdge(BasicBlock *From, BasicBlock *To) {
    assert(From->getParent() == &F && "query for BB from wrong function");
    assert(To->getParent() == &F && "query for BB from wrong function");
    return ForEdge[From][To];
  }

  void print(raw_ostream &os) {
    auto Dump = [&](Value *V) {
      if (isa<Instruction>(V)) {
        os << V->getName() << '\n';
      } else {
        V->print(os);
      }
    };
    for (BasicBlock &BB : F) {
      os << "Block condition " << BB.getName() << ": ";
      Dump(ForBlock[&BB]);
    }
    for (auto &From : F) {
      for (auto &To : F) {
        if (ForEdge[&From].count(&To) == 0) {
          continue;
        }
        os << "Edge condition " << From.getName() << " -> " << To.getName()
           << ": ";
        Dump(ForEdge[&From][&To]);
      }
    }
  }
};

struct InstVectorizeVisitor : InstVisitor<InstVectorizeVisitor> {
  LLVMContext &Context;
  Function &VectorFunc;
  DenseMap<Instruction *, Value *> Scalar2Vector;
  DenseMap<BasicBlock *, BasicBlock *> BlockMap;
  const DenseMap<Function *, Function *> &VectorizedFunctions;

  BasicBlock *VecBB = nullptr;

  SmallVector<Value *, 8> Arguments;

  InstVectorizeVisitor(
      Function &VectorFunc,
      const DenseMap<Function *, Function *> &VectorizedFunctions)
      : Context(VectorFunc.getContext()), VectorFunc(VectorFunc),
        VectorizedFunctions(VectorizedFunctions) {
    for (auto &Arg : VectorFunc.args()) {
      Arguments.push_back(&Arg);
    }
  }

  IRBuilder<> getBuilder() { return IRBuilder<>(VecBB); }

  void visitBasicBlock(BasicBlock &BB) {
    auto NewBB = BasicBlock::Create(Context, BB.getName(), &VectorFunc);
    if (VecBB) {
      getBuilder().CreateBr(NewBB);
    }
    VecBB = NewBB;
    BlockMap[&BB] = VecBB;
  }

  void visitInstruction(Instruction &I) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "LowerSPMD cannot vectorize instruction: ";
    I.print(Msg);
    report_fatal_error(Msg.str());
  }

  void visitReturnInst(ReturnInst &Ret) {
    // Previous steps ensure there's only one return per function.
    if (auto RetVal = Ret.getReturnValue()) {
      auto VecReturn = getBuilder().CreateRet(getVectorized(RetVal));
      setVectorized(&Ret, VecReturn);
    } else {
      getBuilder().CreateRetVoid();
    }
  }

  void visitAllocaInst(AllocaInst &Alloca) {
    if (Alloca.isArrayAllocation()) {
      report_fatal_error("TODO: implement array alloca");
    }
    if (Alloca.isUsedWithInAlloca()) {
      report_fatal_error("TODO: support inalloca");
    }

    auto I32Ty = IntegerType::get(Context, 32);
    auto ArraySize = ConstantInt::get(I32Ty, SIMD_WIDTH);
    auto WholeAlloca =
        getBuilder().CreateAlloca(Alloca.getAllocatedType(), ArraySize);
    WholeAlloca->setAlignment(Alloca.getAlignment());

    SmallVector<Constant *, SIMD_WIDTH> LaneIds;
    for (uint64_t i = 0; i < SIMD_WIDTH; ++i) {
      LaneIds.push_back(ConstantInt::get(I32Ty, i));
    }
    auto VecAllocas = getBuilder().CreateGEP(
        Alloca.getAllocatedType(), WholeAlloca, ConstantVector::get(LaneIds));

    setVectorized(&Alloca, VecAllocas, /* Conditional= */ true);
  }

  void visitIntrinsicInst(IntrinsicInst &Call) {
    switch (Call.getIntrinsicID()) {
    default:
      // Let's treat it like a normal call and hope it works
      visitCallInst(Call);
      break;
    case Intrinsic::spmd_lane_id: {
      auto I32Ty = IntegerType::get(Context, 32);
      SmallVector<Constant *, SIMD_WIDTH> LaneIds;
      for (uint64_t i = 0; i < SIMD_WIDTH; ++i) {
        LaneIds.push_back(ConstantInt::get(I32Ty, i));
      }
      // As this function is pure, ignoring the mask is OK.
      setVectorized(&Call, ConstantVector::get(LaneIds),
                    /* Conditional: */ true);
      break;
    }
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // For now just drop lifetime intrinsics.
      // TODO: Support them, there's a 1:1 mapping between allocas so it
      // should be easy and useful
      break;
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
      // For now just drop debug info.
      // Might want to support debug info in the future, but that is
      // low priority and it's not obvious whether it's easy.
      break;
    case Intrinsic::expect:
      // TODO use these to guide linearization (and also branch weights, btw)
      // For now, just ignore the hint and use the value passed in
      setVectorized(&Call, getVectorized(Call.getArgOperand(0)),
                    /* Conditional: */ true /* FIXME this is wrong */);
      break;
    }
  }

  void visitCallInst(CallInst &Call) {
    auto Callee = Call.getCalledFunction();
    if (auto VectorCallee = VectorizedFunctions.lookup(Callee)) {
      SmallVector<Value *, 8> VectorArgs;
      auto Mask = getVectorized(getCondition(&Call));
      VectorArgs.push_back(Mask);
      for (auto &Arg : Call.arg_operands()) {
        VectorArgs.push_back(getVectorized(&*Arg));
      }
      // Only do the call if any lanes need it. This can not only save
      // time for expensive functions, it also makes recursion work.
      auto Builder = getBuilder();
      auto MaskIntTy = IntegerType::get(Context, SIMD_WIDTH);
      auto MaskAsInt = Builder.CreateBitCast(Mask, MaskIntTy);
      auto NeedCall = Builder.CreateICmpNE(
          MaskAsInt, ConstantInt::get(MaskIntTy, 0), "need_call");

      auto CallBB =
          BasicBlock::Create(Context, "call." + Callee->getName(), &VectorFunc);
      IRBuilder<> CallBuilder(CallBB);
      // TODO transplant attributes as appropriate
      // (some, especially argument attributes, may not apply)
      auto VectorCall = CallBuilder.CreateCall(VectorCallee, VectorArgs);

      auto NextBB = BasicBlock::Create(Context, VecBB->getName(), &VectorFunc);
      CallBuilder.CreateBr(NextBB);
      Builder.CreateCondBr(NeedCall, CallBB, NextBB);

      auto OldBB = VecBB;
      auto RetTy = VectorCallee->getReturnType();
      if (!RetTy->isVoidTy()) {
        VecBB = NextBB;
        auto Phi = getBuilder().CreatePHI(RetTy, 2);
        auto Undef = UndefValue::get(RetTy);
        Phi->addIncoming(VectorCall, CallBB);
        Phi->addIncoming(Undef, OldBB);
        setVectorized(&Call, Phi, /* Conditional: */ true);
      } else {
        VecBB = CallBB;
        setVectorized(&Call, VectorCall, /* Conditional: */ true);
      }
      VecBB = NextBB;
    } else {
      // Slow path for functions that don't have vectorized versions
      auto Mask = getVectorized(getCondition(&Call));
      SmallVector<Value *, 4> VecArgs;
      for (Use &Arg : Call.arg_operands()) {
        VecArgs.push_back(getVectorized(&*Arg));
      }
      SmallVector<Value *, SIMD_WIDTH> LaneResults;
      for (uint64_t i = 0; i < SIMD_WIDTH; ++i) {
        LaneResults.push_back(createConditionalCall(Call, VecArgs, Mask, i));
      }
      if (!Call.getType()->isVoidTy()) {
        setVectorized(&Call, assembleVector(&Call, LaneResults),
                      /* Conditional: */ true);
      }
    }
  }

  Value *createConditionalCall(CallInst &Call,
                               SmallVectorImpl<Value *> &VecArgs, Value *Mask,
                               uint64_t lane) {
    auto Callee = Call.getCalledFunction();
    assert(Callee);

    auto CondBB = BasicBlock::Create(Context, "maskedcall." + Callee->getName(),
                                     &VectorFunc);
    auto NewVecBB = BasicBlock::Create(Context, "linearized", &VectorFunc);
    CondBB->moveAfter(VecBB);
    NewVecBB->moveAfter(CondBB);

    auto Condition = getBuilder().CreateExtractElement(Mask, lane);
    BranchInst::Create(CondBB, NewVecBB, Condition, VecBB);
    IRBuilder<> CondBuilder(CondBB);
    SmallVector<Value *, 4> ExtractedArgs;
    for (Value *VecArg : VecArgs) {
      ExtractedArgs.push_back(CondBuilder.CreateExtractElement(VecArg, lane));
    }
    auto LaneResult =
        CondBuilder.CreateCall(Callee, ExtractedArgs, Call.getName());
    CondBuilder.CreateBr(NewVecBB);

    auto PrevBB = VecBB;
    VecBB = NewVecBB;

    // As the instruction is now conditional, it does not dominate the following
    // uses any more. But because we know the value is only used if the
    // condition is met, we can insert a phi like this to fix that:
    //   phi [ undef, CurrBB ], [ Replacement, CondBB ]
    auto Type = Call.getType();
    if (Type->isVoidTy()) {
      return nullptr;
    } else {
      auto Phi = getBuilder().CreatePHI(Type, 2);
      auto Undef = UndefValue::get(Type);
      Phi->addIncoming(LaneResult, CondBB);
      Phi->addIncoming(Undef, PrevBB);
      return Phi;
    }
  }

  void visitCmpInst(CmpInst &Cmp) {
    auto LHS = getVectorized(Cmp.getOperand(0));
    auto RHS = getVectorized(Cmp.getOperand(1));
    auto VecCmp =
        CmpInst::Create(Cmp.getOpcode(), Cmp.getPredicate(), LHS, RHS);
    setVectorized(&Cmp, VecCmp);
  }

  void visitBinaryOperator(BinaryOperator &Op) {
    auto LHS = getVectorized(Op.getOperand(0));
    auto RHS = getVectorized(Op.getOperand(1));
    auto VecOp = BinaryOperator::Create(Op.getOpcode(), LHS, RHS);
    setVectorized(&Op, VecOp);
  }

  void visitCastInst(CastInst &Cast) {
    auto VecVal = getVectorized(Cast.getOperand(0));
    auto VecCast = CastInst::Create(Cast.getOpcode(), VecVal,
                                    getVectorType(Cast.getDestTy()));
    setVectorized(&Cast, VecCast);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    SmallVector<Value *, 16> IdxList;
    for (auto It = GEP.idx_begin(), End = GEP.idx_end(); It != End; ++It) {
      IdxList.push_back(getVectorized(It->get()));
    }
    auto PtrVec = getVectorized(GEP.getPointerOperand());
    auto VecGEP =
        GetElementPtrInst::Create(GEP.getSourceElementType(), PtrVec, IdxList);
    setVectorized(&GEP, VecGEP);
  }

  void visitLoadInst(LoadInst &Load) {
    auto PtrVec = getVectorized(Load.getOperand(0));
    auto Mask = getVectorized(getCondition(&Load));
    auto Gather =
        getBuilder().CreateMaskedGather(PtrVec, Load.getAlignment(), Mask);
    setVectorized(&Load, Gather, /* Conditional= */ true);
  }

  void visitStoreInst(StoreInst &Store) {
    auto ValVec = getVectorized(Store.getOperand(0));
    auto PtrVec = getVectorized(Store.getOperand(1));
    auto Mask = getVectorized(getCondition(&Store));
    auto Scatter = getBuilder().CreateMaskedScatter(ValVec, PtrVec,
                                                    Store.getAlignment(), Mask);
    setVectorized(&Store, Scatter, /* Conditional= */ true);
  }

  void visitSelectInst(SelectInst &Sel) {
    auto Mask = getVectorized(Sel.getCondition());
    auto TrueVec = getVectorized(Sel.getTrueValue());
    auto FalseVec = getVectorized(Sel.getFalseValue());
    auto VecSel = SelectInst::Create(Mask, TrueVec, FalseVec);
    setVectorized(&Sel, VecSel);
  }

  Value *getCondition(Instruction *I) {
    if (auto MD = I->getMetadata(CONDITION_METADATA_ID)) {
      return cast<ValueAsMetadata>(&*MD->getOperand(0))->getValue();
    } else {
      std::string msg;
      raw_string_ostream Msg(msg);
      Msg << "LowerSPMD: condition metadata missing on: ";
      I->print(Msg);
      report_fatal_error(Msg.str());
    }
  }

  void setVectorized(Instruction *Scalar, Value *Vectorized,
                     bool Conditional = false) {
    if (auto VecInst = dyn_cast<Instruction>(Vectorized)) {
      if (!VecInst->getParent()) {
        VecBB->getInstList().push_back(VecInst);
      } else {
        assert(VecInst->getParent() == VecBB);
      }
    }
    Vectorized->setName(Scalar->getName());

    DEBUG(dbgs() << "Replacing scalar instruction:\n");
    DEBUG(Scalar->print(dbgs()));
    DEBUG(dbgs() << "\nwith vectorized instruction:\n");
    DEBUG(Vectorized->print(dbgs()));
    DEBUG(dbgs() << "\n");

    if (Conditional) {
      assert(Scalar->getMetadata(CONDITION_METADATA_ID) &&
             "Unconditional instruction masked");
    } else {
      assert(!Scalar->getMetadata(CONDITION_METADATA_ID) &&
             "Conditional instruction not masked");
    }
    Scalar2Vector[Scalar] = Vectorized;
  }

  Value *getVectorized(Value *Scalar) {
    if (auto I = dyn_cast<Instruction>(Scalar)) {
      assert(Scalar2Vector.count(I));
      return Scalar2Vector[I];
    } else if (auto Arg = dyn_cast<Argument>(Scalar)) {
      return Arguments[Arg->getArgNo()];
    } else {
      assert(isa<Constant>(Scalar));
      return getBuilder().CreateVectorSplat(SIMD_WIDTH, Scalar);
    }
  }

  Type *getVectorType(Type *ScalarTy) {
    return VectorType::get(ScalarTy, SIMD_WIDTH);
  }

  Value *assembleVector(Value *Scalar, ArrayRef<Value *> Elems) {
    assert(Elems.size() == SIMD_WIDTH);
    auto VectorTy = getVectorType(Scalar->getType());
    Value *Vector = UndefValue::get(VectorTy);
    auto Builder = getBuilder();
    for (unsigned i = 0; i < SIMD_WIDTH; ++i) {
      Vector = Builder.CreateInsertElement(Vector, Elems[i], i);
    }
    return Vector;
  }
};

struct FunctionVectorizer {
  LLVMContext &Context;
  Function *LinearFunc;
  Function *VectorFunc;
  const DenseMap<Function *, Function *> &VectorizedFunctions;
  DominatorTree DomTree;
  LoopInfo LI;

  FunctionVectorizer(
      Function &F, Function &VF,
      const DenseMap<Function *, Function *> &VectorizedFunctions)
      : Context(F.getContext()), VectorFunc(&VF),
        VectorizedFunctions(VectorizedFunctions) {
    // We need to clone the source function at the very least to change the
    // signature. But since we already have a copy, we also mangle that copy in-
    // place, which would be incorrect otherwise (scalar code could call it).
    ValueToValueMapTy ArgMapping;
    SmallVector<ReturnInst *, 1> Returns;
    LinearFunc = prepareScalarFunction(F, ArgMapping);
    CloneFunctionInto(LinearFunc, &F, ArgMapping,
                      /* ModuleLevelChanges: */ false, Returns);

    if (Returns.size() != 1) {
      report_fatal_error("LowerSPMD: cannot vectorize function with multiple "
                         "returns, run mergereturn");
    }

    DomTree = DominatorTree(*LinearFunc);
    LI = LoopInfo(DomTree);
    if (!LI.empty()) {
      DEBUG(LI.print(dbgs()));
      DEBUG(dbgs() << '\n');
      report_fatal_error("TODO support loops");
    }
  }

  // Create an empty function with the same prototype as F, except that an
  // additional i1 argument (the condition for the entry block) is inserted
  // before the first argument.
  // Also fills out a mapping from the input function's arguments to the output
  // function's arguments, for CloneFunctionInto
  Function *prepareScalarFunction(Function &F, ValueToValueMapTy &ArgMapping) {
    SmallVector<Type *, 8> ArgsPlusCond;
    ArgsPlusCond.push_back(Type::getInt1Ty(F.getContext()));
    for (auto ArgTy : F.getFunctionType()->params()) {
      ArgsPlusCond.push_back(ArgTy);
    }
    assert(!F.isVarArg()); // TODO necessary?
    auto NewFT = FunctionType::get(F.getReturnType(), ArgsPlusCond,
                                   /* isVarArg: */ false);
    auto NewF =
        Function::Create(NewFT, F.getLinkage(), F.getName(), F.getParent());
    auto NewArgsIter = NewF->arg_begin();
    ++NewArgsIter; // skip the condition we added
    for (auto &OldArg : F.args()) {
      assert(NewArgsIter != NewF->arg_end());
      ArgMapping[&OldArg] = &*NewArgsIter;
      ++NewArgsIter;
    }
    assert(NewArgsIter == NewF->arg_end());
    return NewF;
  }

  void createBlockConditions(Conditions &Conds) {
    BBMap<Value *> PlaceholderConds = Conds.ForBlock;
    BBMap<std::vector<Value *>> IncomingConds;
    for (auto &From : *LinearFunc) {
      for (auto KVPair : Conds.ForEdge[&From]) {
        BasicBlock &To = *KVPair.getFirst();
        IncomingConds[&To].push_back(Conds.ForEdge[&From][&To]);
      }
    }

    for (BasicBlock &BB : *LinearFunc) {
      IRBuilder<> Builder(&BB, BB.getFirstInsertionPt());
      Value *Cond = nullptr;
      for (Value *IncomingCond : IncomingConds[&BB]) {
        if (!Cond) {
          Cond = IncomingCond;
        } else {
          Cond = Builder.CreateOr(Cond, IncomingCond, BB.getName() + ".exec");
        }
      }
      if (!Cond) {
        assert(&BB == &LinearFunc->getEntryBlock() && "unreachable block");
        // The condition for the entry block is passed in as the first argument,
        Cond = &*LinearFunc->arg_begin();
      }
      assert(Cond->getType() == IntegerType::get(Context, 1));
      Conds.forBlock(&BB) = Cond;
    }

    // Some of the edge conditions are block condition placeholders,
    // so we need to keep track of the replacements to apply them later.
    // We could use a ValueHandle, but (1) I'm too stupid to get that to work,
    // and (2) it complicates type signatures everywhere for something that
    // can be solved locally here.
    DenseMap<Value *, Value *> Replace;
    for (auto &BB : *LinearFunc) {
      Value *Cond = Conds.forBlock(&BB);
      Value *Placeholder = PlaceholderConds[&BB];
      Placeholder->replaceAllUsesWith(Cond);
      Replace[Placeholder] = Cond;
    }
    for (auto &BB : *LinearFunc) {
      auto NewV = Replace.find(Conds.forBlock(&BB));
      if (NewV != Replace.end()) {
        Conds.forBlock(&BB) = NewV->getSecond();
      }

      for (auto KVPair : Conds.ForEdge[&BB]) {
        Value *V = KVPair.getSecond();
        auto NewV = Replace.find(V);
        if (NewV != Replace.end()) {
          Conds.ForEdge[&BB][KVPair.getFirst()] = NewV->getSecond();
        }
      }
    }
    for (auto KVPair : PlaceholderConds) {
      cast<Instruction>(KVPair.getSecond())->eraseFromParent();
    }
  }

  void createEdgeConditions(BasicBlock &BB, Conditions &Conds) {
    auto *BlockCond = Conds.forBlock(&BB);
    auto *T = BB.getTerminator();
    if (isa<UnreachableInst>(T) || isa<ReturnInst>(T)) {
      // No successors => nothing to do.
      return;
    }
    auto *Br = dyn_cast<BranchInst>(T);
    if (!Br) {
      std::string msg;
      raw_string_ostream Msg(msg);
      Msg << "LowerSPMD: cannot handle terminator: ";
      T->print(Msg);
      report_fatal_error(Msg.str());
    }
    auto FoundEdgeCond = [&](BasicBlock *Succ, Value *Cond) {
      DEBUG(dbgs() << "Jump " << BB.getName() << " -> " << Succ->getName()
                   << " with condition:\n");
      DEBUG(Cond->print(dbgs()));
      DEBUG(dbgs() << "\n");
      Conds.addEdge(&BB, Succ, Cond);
    };

    if (Br->isConditional()) {
      IRBuilder<> Builder(Br);
      auto S0 = Br->getSuccessor(0), S1 = Br->getSuccessor(1);
      auto *Cond0 = Br->getCondition();
      auto *Cond1 = Builder.CreateNot(Cond0, "not." + Cond0->getName());
      Cond0 = Builder.CreateAnd(BlockCond, Cond0,
                                BB.getName() + ".to." + S0->getName());
      Cond1 = Builder.CreateAnd(BlockCond, Cond1,
                                BB.getName() + ".to." + S1->getName());
      FoundEdgeCond(S0, Cond0);
      FoundEdgeCond(S1, Cond1);
    } else {
      // Single successor => just re-use the block condition.
      auto Succ = Br->getSuccessor(0);
      FoundEdgeCond(Succ, BlockCond);
    }
  }

  void createBlockConditionPlaceholders(Conditions &Conds) {
    Value *False = ConstantInt::getFalse(Context);
    for (BasicBlock &BB : *LinearFunc) {
      Instruction *ip = &*BB.getFirstInsertionPt();
      Instruction *Placeholder =
          BinaryOperator::Create(BinaryOperator::Or, False, False,
                                 "exec.placeholder." + BB.getName(), ip);
      Conds.forBlock(&BB) = Placeholder;
    }
  }

  void markAsConditional(Instruction *Inst, Value *Condition) {
    assert(!Inst->getMetadata(CONDITION_METADATA_ID) &&
           "Instruction already conditional");
    auto CondMD = ValueAsMetadata::get(Condition);
    Inst->setMetadata(CONDITION_METADATA_ID, MDNode::get(Context, CondMD));
  }

  void PhiToSelect(PHINode *Phi, Conditions &Conds, BasicBlock *OldBB,
                   BasicBlock *LinearBB) {
    auto Result = Phi->getIncomingValue(0);
    auto Cond = Conds.forEdge(Phi->getIncomingBlock(0), OldBB);
    IRBuilder<> Builder(LinearBB);
    for (unsigned i = 1; i < Phi->getNumIncomingValues(); ++i) {
      Result = Builder.CreateSelect(Cond, Result, Phi->getIncomingValue(i),
                                    Phi->getName());
      Cond = Conds.forEdge(Phi->getIncomingBlock(i), OldBB);
    }
    Phi->replaceAllUsesWith(Result);
    Phi->eraseFromParent();
  }

  void linearizeCFG(Conditions &Conds) {
    ReversePostOrderTraversal<Function *> RPOT(LinearFunc);
    std::vector<BasicBlock *> BBsInTopologicalOrder(RPOT.begin(), RPOT.end());

    auto *LinearBB = BasicBlock::Create(Context, "linearized", LinearFunc,
                                        &LinearFunc->getEntryBlock());
    Instruction *Return = nullptr;

    for (auto SourceBB : BBsInTopologicalOrder) {
      auto BlockCond = Conds.forBlock(SourceBB);
      while (!SourceBB->empty()) {
        Instruction *I = &SourceBB->front();
        if (auto *Phi = dyn_cast<PHINode>(I)) {
          PhiToSelect(Phi, Conds, SourceBB, LinearBB);
        } else if (isa<ReturnInst>(I)) {
          // The constructor ensured there's only one return.
          // Because of unreachable instructions, the BB with the return
          // may not be last in the topological order, so to ensure
          // the return is at the end of the linearized function we remember
          // it and put it at the very end after we've visited all BBs.
          assert(!Return);
          Return = I;
          I->removeFromParent();
        } else if (isa<TerminatorInst>(I)) {
          I->eraseFromParent();
        } else {
          if (!isSafeToSpeculativelyExecute(I)) {
            markAsConditional(I, BlockCond);
          }
          // HACK should support unmasked gather/scatter and intrinsic calls
          else if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<CallInst>(I)) {
            markAsConditional(I, BlockCond);
          }
          I->removeFromParent();
          LinearBB->getInstList().push_back(I);
        }
      }
    }

    assert(Return);
    LinearBB->getInstList().push_back(Return);

    // We're turning phis into selects on the fly, so we can't
    // remove BBs in the loop since we might access them while
    // processing later phis.
    for (auto BB : BBsInTopologicalOrder) {
      BB->eraseFromParent();
    }
  }

  Function *run() {
    Conditions Conds(*LinearFunc);
    createBlockConditionPlaceholders(Conds);

    // First, we materialize the (scalar) condition for *every* outgoing edge.
    // Some of these values are constant (unconditional branches) or redundant
    // (the negated condition for the `false` branch of a conditional branch),
    // but they will be needed later to construct the masks for vectorization.
    for (auto &BB : *LinearFunc) {
      createEdgeConditions(BB, Conds);
    }
    createBlockConditions(Conds);

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(Conds.print(dbgs()));
    DEBUG(dbgs() << "\n");

    // TODO compute loop masks, loop live values, and more generally:
    // TODO support loops at all

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(LinearFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    linearizeCFG(Conds);
    DEBUG(LinearFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    InstVectorizeVisitor IVV(*VectorFunc, VectorizedFunctions);
    IVV.visit(*LinearFunc);

    LinearFunc->eraseFromParent();
    return VectorFunc;
  }
};

struct FindDirectCalls : InstVisitor<FindDirectCalls> {
  SmallVector<Function *, 16> Callees;

  void visitCallInst(CallInst &Call) {
    if (auto F = Call.getCalledFunction()) {
      Callees.push_back(F);
    }
  }
};

Function *unwrapFunctionBitcast(Value *Bitcast) {
  return cast<Function>(cast<ConstantExpr>(Bitcast)->getOperand(0));
}

std::vector<Function *>
findFunctionsToVectorize(const std::vector<CallInst *> &Roots) {
  SmallPtrSet<Function *, 2> Seen;
  std::vector<Function *> Worklist;
  std::vector<Function *> Result;

  auto Consider = [&](Function *F) {
    if (!F) {
      report_fatal_error("Indirect call in SPMD code");
    }
    if (Seen.insert(F).second) {
      Worklist.push_back(F);
    }
  };

  // The roots of the collection process are the call sites of the intrinsic.
  for (auto Call : Roots) {
    Consider(unwrapFunctionBitcast(Call->getArgOperand(0)));
  };

  while (!Worklist.empty()) {
    Function *F = Worklist.back();
    Worklist.pop_back();
    if (F->empty()) {
      // This is an external function (e.g., intrinsic) so we have to make do
      // without it.
      continue;
    }
    Result.push_back(F);
    FindDirectCalls FDC;
    FDC.visit(F);
    std::for_each(FDC.Callees.begin(), FDC.Callees.end(), Consider);
  }

  return Result;
}

// Create a function with the same signature as ScalarFunc, except that:
// (1) it takes a mask argument as new first argument, and
// (2) all arguments and the return value are turned into vectors
Function *predefineVectorizedFunction(Function &ScalarFunc) {
  auto ScalarFT = ScalarFunc.getFunctionType();
  SmallVector<Type *, 8> VectorArgTys;
  auto I1Ty = IntegerType::get(ScalarFunc.getContext(), 1);
  VectorArgTys.push_back(VectorType::get(I1Ty, SIMD_WIDTH));
  for (auto ArgTy : ScalarFT->params()) {
    VectorArgTys.push_back(VectorType::get(ArgTy, SIMD_WIDTH));
  }
  assert(!ScalarFT->isVarArg());
  auto VoidTy = Type::getVoidTy(ScalarFunc.getContext());
  auto ScalarReturnTy = ScalarFT->getReturnType();
  auto VectorReturnTy = ScalarReturnTy->isVoidTy()
                            ? VoidTy
                            : VectorType::get(ScalarReturnTy, SIMD_WIDTH);
  auto VectorFT = FunctionType::get(VectorReturnTy, VectorArgTys,
                                    /* isVarArg: */ false);
  return Function::Create(VectorFT, ScalarFunc.getLinkage(),
                          ScalarFunc.getName() + ".vector",
                          ScalarFunc.getParent());
}

struct LowerSPMD : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  LowerSPMD() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    std::vector<CallInst *> RootCalls;
    auto Intr = Intrinsic::getDeclaration(&M, Intrinsic::spmd_call);
    for (User *U : Intr->users()) {
      RootCalls.push_back(cast<CallInst>(U));
    }

    auto ScalarFuncs = findFunctionsToVectorize(RootCalls);
    DenseMap<Function *, Function *> Vectorized;
    // Vectorized functions may call each other (including recursively), so
    // declare vectorized versions of all functions before defining any.
    for (Function *F : ScalarFuncs) {
      Vectorized[F] = predefineVectorizedFunction(*F);
    }
    for (Function *F : ScalarFuncs) {
      FunctionVectorizer FV(*F, *Vectorized[F], Vectorized);
      FV.run();
    }
    DEBUG(dbgs() << "===================================================\n");

    for (auto Call : RootCalls) {
      auto ScalarF = unwrapFunctionBitcast(Call->getArgOperand(0));
      auto Arg = Call->getArgOperand(1);
      IRBuilder<> Builder(Call);
      assert(Vectorized.count(ScalarF));

      SmallVector<Value *, 2> Args;
      auto True = ConstantInt::getTrue(ScalarF->getContext());
      Args.push_back(Builder.CreateVectorSplat(SIMD_WIDTH, True));
      Args.push_back(Builder.CreateVectorSplat(SIMD_WIDTH, Arg));
      Builder.CreateCall(Vectorized[ScalarF], Args);
      Call->eraseFromParent();
    }

    return true;
  }
};
}

char LowerSPMD::ID = 0;
INITIALIZE_PASS(LowerSPMD, "lowerspmd", "Vectorize SPMD regions", false, false)

namespace llvm {
Pass *createLowerSPMDPass() { return new LowerSPMD(); }
}
