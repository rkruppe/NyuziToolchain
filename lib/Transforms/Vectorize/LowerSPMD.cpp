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

#define DEBUG_TYPE "LowerSPMD"

namespace {

const unsigned SIMD_WIDTH = 16;
const char *CONDITION_METADATA_ID = "spmd-conditional";

template <typename T> using BBMap = DenseMap<BasicBlock *, T>;
using CondVec = SmallVector<Value *, 2>;

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
  Function &ResultF;
  DenseMap<Instruction *, Value *> Scalar2Vector;
  DenseMap<BasicBlock *, BasicBlock *> BlockMap;

  BasicBlock *VecBB;

  SmallVector<Value *, 8> ArgumentSplats;

  InstVectorizeVisitor(Function &ResultF)
      : Context(ResultF.getContext()), ResultF(ResultF) {
    // TODO: at latest once multiple functions are supported, broadcasting is
    // not always correct and should thus be left to the caller -- instead, the
    // function signature should change to take vectors.
    VecBB = BasicBlock::Create(Context, "", &ResultF);
    auto Builder = getBuilder();
    for (auto It = ResultF.arg_begin(), End = ResultF.arg_end(); It != End;
         ++It) {
      ArgumentSplats.push_back(Builder.CreateVectorSplat(SIMD_WIDTH, &*It));
    }
  }

  IRBuilder<> getBuilder() { return IRBuilder<>(VecBB); }

  void visitBasicBlock(BasicBlock &BB) {
    auto NewBB = BasicBlock::Create(Context, BB.getName(), &ResultF);
    getBuilder().CreateBr(NewBB);
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
    assert(!Ret.getReturnValue());
    getBuilder().CreateRetVoid();
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
    auto WholeAlloca = getBuilder().CreateAlloca(Alloca.getAllocatedType(), ArraySize);
    WholeAlloca->setAlignment(Alloca.getAlignment());

    SmallVector<Constant *, SIMD_WIDTH> LaneIds;
    for (uint64_t i = 0; i < SIMD_WIDTH; ++i) {
      LaneIds.push_back(ConstantInt::get(I32Ty, i));
    }
    auto VecAllocas = getBuilder().CreateGEP(Alloca.getAllocatedType(), WholeAlloca,
                                             ConstantVector::get(LaneIds));

    setVectorized(&Alloca, VecAllocas, /* Conditional= */ true);
  }

  void visitCallInst(CallInst &Call) {
    auto Callee = Call.getCalledFunction();
    if (Callee->getIntrinsicID() == Intrinsic::spmd_lane_id) {
      auto I32Ty = IntegerType::get(Context, 32);
      SmallVector<Constant *, SIMD_WIDTH> LaneIds;
      for (uint64_t i = 0; i < SIMD_WIDTH; ++i) {
        LaneIds.push_back(ConstantInt::get(I32Ty, i));
      }
      // As this function is pure, ignoring the mask is OK.
      setVectorized(&Call, ConstantVector::get(LaneIds),
                    /* Conditional= */ true);
    } else if (Callee->getIntrinsicID() == Intrinsic::lifetime_start ||
               Callee->getIntrinsicID() == Intrinsic::lifetime_end) {
      // Just drop lifetime intrinsics
      // TODO: reconsider
    } else {
      // TODO handle vectorized functions with a single call
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
        setVectorized(&Call, assembleVector(&Call, LaneResults));
      }
    }
  }

  Value *createConditionalCall(CallInst &Call,
                               SmallVectorImpl<Value *> &VecArgs, Value *Mask,
                               uint64_t lane) {
    auto Callee = Call.getCalledFunction();
    assert(Callee);

    auto CondBB = BasicBlock::Create(Context, "maskedcall." + Callee->getName(),
                                     &ResultF);
    auto NewVecBB = BasicBlock::Create(Context, "linearized", &ResultF);
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
    // This isn't what we want for simple operations that can get vectorized
    // with a mask later (e.g., loads and stores) but it's simple enough to
    // recognize.
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
      report_fatal_error("Expected condition metadata");
    }
  }

  void setVectorized(Instruction *Scalar, Value *Vectorized,
                     bool Conditional = false) {
/*
    DEBUG(dbgs() << "Vectorizing: ");
    DEBUG(Scalar->print(dbgs()));
    DEBUG(dbgs() << " => \t");
    DEBUG(Vectorized->print(dbgs()));
    DEBUG(dbgs() << "\n");
*/
    if (auto VecInst = dyn_cast<Instruction>(Vectorized)) {
      if (!VecInst->getParent()) {
        VecBB->getInstList().push_back(VecInst);
      } else {
        assert(VecInst->getParent() == VecBB);
      }
    }
    if (Conditional) {
      assert(Scalar->getMetadata(CONDITION_METADATA_ID) &&
             "Unconditional instruction masked");
    } else {
      assert(!Scalar->getMetadata(CONDITION_METADATA_ID) &&
             "Conditional instruction not masked");
    }
    Vectorized->setName(Scalar->getName());
    Scalar2Vector[Scalar] = Vectorized;
  }

  Value *getVectorized(Value *Scalar) {
    if (auto I = dyn_cast<Instruction>(Scalar)) {
      if (!Scalar2Vector.count(I)) // TODO HACK
        return UndefValue::get(getVectorType(Scalar->getType()));
      assert(Scalar2Vector.count(I));
      return Scalar2Vector[I];
    } else if (auto Arg = dyn_cast<Argument>(Scalar)) {
      return ArgumentSplats[Arg->getArgNo()];
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
  Function *VectorFunc;
  DominatorTree DomTree;
  LoopInfo LI;

  FunctionVectorizer(Function &F) : Context(F.getContext()) {
    if (!LI.empty()) {
      DEBUG(LI.print(dbgs()));
      DEBUG(dbgs() << '\n');
      report_fatal_error("TODO support loops");
    }
    ValueToValueMapTy Scalar2Vector;
    // TODO create a function with an added argument for the entry block mask,
    // then CloneIntoFunction that

    // Clone the function because there might be calls from scalar code.
    VectorFunc = CloneFunction(&F, Scalar2Vector);
    DomTree = DominatorTree(*VectorFunc);
    LI = LoopInfo(DomTree);
  }

  void createBlockConditions(Conditions &Conds) {
    BBMap<Value *> PlaceholderConds = Conds.ForBlock;
    BBMap<std::vector<Value *>> IncomingConds;
    for (auto &From : *VectorFunc) {
      for (auto KVPair : Conds.ForEdge[&From]) {
        BasicBlock &To = *KVPair.getFirst();
        IncomingConds[&To].push_back(Conds.ForEdge[&From][&To]);
      }
    }

    for (BasicBlock &BB : *VectorFunc) {
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
        assert(&BB == &VectorFunc->getEntryBlock() && "unreachable block");
        Cond = Builder.getTrue();
        assert(Cond->getType() == IntegerType::get(Context, 1));
      }
      Conds.forBlock(&BB) = Cond;
    }

    // Some of the edge conditions are block condition placeholders,
    // so we need to keep track of the replacements to apply them later.
    // We could use a ValueHandle, but (1) I'm too stupid to get that to work,
    // and (2) it complicates type signatures everywhere for something that
    // can be solved locally here.
    DenseMap<Value *, Value *> Replace;
    for (auto &BB : *VectorFunc) {
      Value *Cond = Conds.forBlock(&BB);
      Value *Placeholder = PlaceholderConds[&BB];
      Placeholder->replaceAllUsesWith(Cond);
      Replace[Placeholder] = Cond;
    }
    for (auto &BB : *VectorFunc) {
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
    auto *Br = dyn_cast<BranchInst>(T);
    if (!Br) {
      if (isa<UnreachableInst>(T) || isa<ReturnInst>(T)) {
        // No successors => nothing to do.
        return;
      } else {
        std::string msg;
        raw_string_ostream Msg(msg);
        Msg << "LowerSPMD cannot handle terminator: ";
        T->print(Msg);
        report_fatal_error(Msg.str());
      }
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
    for (BasicBlock &BB : *VectorFunc) {
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
    assert(Phi->getNumIncomingValues() == 2 &&
           "PHI with 1 or 3+ incoming values not suported yet");
    auto Cond0 = Conds.forEdge(Phi->getIncomingBlock(0), OldBB);
    auto Val0 = Phi->getIncomingValue(0);
    auto Val1 = Phi->getIncomingValue(1);
    IRBuilder<> Builder(LinearBB);
    auto Select = Builder.CreateSelect(
        Cond0, Val0, Val1, "mix." + Val0->getName() + "." + Val1->getName());
    Phi->replaceAllUsesWith(Select);
    Phi->eraseFromParent();
  }

  void linearizeCFG(Conditions &Conds) {
    ReversePostOrderTraversal<Function *> RPOT(VectorFunc);
    std::vector<BasicBlock *> BBsInTopologicalOrder(RPOT.begin(), RPOT.end());

    auto *LinearBB = BasicBlock::Create(Context, "linearized", VectorFunc,
                                        &VectorFunc->getEntryBlock());

    for (auto SourceBB : BBsInTopologicalOrder) {
      auto BlockCond = Conds.forBlock(SourceBB);
      while (!SourceBB->empty()) {
        Instruction *I = &SourceBB->front();
        if (auto *Phi = dyn_cast<PHINode>(I)) {
          PhiToSelect(Phi, Conds, SourceBB, LinearBB);
        } else if (auto *Ret = dyn_cast<ReturnInst>(I)) {
          // TODO handle return values
          assert(!Ret->getReturnValue());
          I->eraseFromParent();
        } else if (isa<TerminatorInst>(I)) {
          I->eraseFromParent();
        } else {
          if (!isSafeToSpeculativelyExecute(I)) {
            markAsConditional(I, BlockCond);
          }
          I->removeFromParent();
          LinearBB->getInstList().push_back(I);
        }
      }
    }

    // We're turning phis into selects on the fly, so we can't
    // remove BBs in the loop since we might access them while
    // processing later phis.
    for (auto BB : BBsInTopologicalOrder) {
      BB->eraseFromParent();
    }

    IRBuilder<> Builder(LinearBB);
    Builder.CreateRetVoid();
  }

  Function *run() {
    Conditions Conds(*VectorFunc);
    createBlockConditionPlaceholders(Conds);

    // First, we materialize the (scalar) condition for *every* outgoing edge.
    // Some of these values are constant (unconditional branches) or redundant
    // (the negated condition for the `false` branch of a conditional branch),
    // but they will be needed later to construct the masks for vectorization.
    // The representation of choise is: For each BB, store a vector where the
    // i-th value is the condition for the i-th outgoing edge (i.e., the edge
    // to the i-th successor).
    for (auto &BB : *VectorFunc) {
      createEdgeConditions(BB, Conds);
    }
    createBlockConditions(Conds);

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(Conds.print(dbgs()));
    DEBUG(dbgs() << "\n");

    // TODO compute loop masks, loop live values, and more generally:
    // TODO support loops at all

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(VectorFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    linearizeCFG(Conds);
    DEBUG(VectorFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    auto ResultF = Function::Create(
        VectorFunc->getFunctionType(), VectorFunc->getLinkage(),
        VectorFunc->getName() + ".vector", VectorFunc->getParent());
    InstVectorizeVisitor IVV(*ResultF);
    IVV.visit(*VectorFunc);
    // DEBUG(ResultF->print(dbgs()));

    VectorFunc->eraseFromParent();
    return ResultF;
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

struct LowerSPMD : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  LowerSPMD() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    std::vector<CallInst *> RootCalls;
    auto Intr = Intrinsic::getDeclaration(&M, Intrinsic::spmd_call);
    for (User *U : Intr->users()) {
      RootCalls.push_back(cast<CallInst>(U));
    }

    DenseMap<Function *, Function *> Vectorized;
    for (Function *F : findFunctionsToVectorize(RootCalls)) {
      FunctionVectorizer FV(*F);
      Vectorized[F] = FV.run();
    }

    for (auto Call : RootCalls) {
      auto ScalarF = unwrapFunctionBitcast(Call->getArgOperand(0));
      auto Arg = Call->getArgOperand(1);
      IRBuilder<> Builder(Call);
      assert(Vectorized.count(ScalarF));
      Builder.CreateCall(Vectorized[ScalarF], Arg);
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
