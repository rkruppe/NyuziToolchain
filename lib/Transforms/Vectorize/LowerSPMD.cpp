#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
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
  Function &VectorFunc;
  LLVMContext &Context;
  const LoopInfo &LI;
  const DenseMap<Function *, Function *> &VectorizedFunctions;

  DenseMap<Instruction *, Value *> Scalar2Vector;
  BBMap<BasicBlock *> BlockMap;

  BasicBlock *CurrentBB = nullptr;

  SmallVector<Value *, 8> Arguments;

  enum class MaskMode {
    Ignore,
    Masked,
    Unmasked,
  };

  InstVectorizeVisitor(
      Function &VectorFunc, const LoopInfo &LI,
      const DenseMap<Function *, Function *> &VectorizedFunctions)
      : VectorFunc(VectorFunc), Context(VectorFunc.getContext()), LI(LI),
        VectorizedFunctions(VectorizedFunctions) {
    for (auto &Arg : VectorFunc.args()) {
      Arguments.push_back(&Arg);
    }
  }

  IRBuilder<> getBuilder() { return IRBuilder<>(CurrentBB); }

  void visitFunction(Function &F) {
    for (auto &BB : F) {
      BlockMap[&BB] = BasicBlock::Create(Context, BB.getName(), &VectorFunc);
    }
  }

  void visitBasicBlock(BasicBlock &BB) { CurrentBB = BlockMap[&BB]; }

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
      record(&Ret, VecReturn, MaskMode::Unmasked);
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
    if (!Alloca.isStaticAlloca()) {
      report_fatal_error("dynamic alloca not supported (yet)");
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

    record(&Alloca, VecAllocas, MaskMode::Ignore);
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
      record(&Call, ConstantVector::get(LaneIds), MaskMode::Ignore);
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
      record(&Call, getVectorized(Call.getArgOperand(0)), MaskMode::Ignore);
      break;
    }
  }

  // Insert I in a new block conditional on Cond, like this:
  //
  //  +----------------------+ , original CurrentBB
  //  |  ... current BB ...  |
  //  | br %cond, %bb1, %bb2 |
  //  +----------------------+
  //     | true       | false
  //     v            |
  //  +------------+  | ; newly created BB
  //  | %I = ...   |  |
  //  +------------+  |
  //     |            |
  //     v            v
  //  +-------------------------+ ; new CurrentBB
  //  | %result = phi %I, undef |
  //  |           ...           |
  //
  // The PHI is only created if %I has a non-void result.
  // The PHI is returned in that case, nullptr otherwise.
  //
  Instruction *insertConditional(Value *Cond, Instruction *I,
                                 const Twine &CondBlockName) {
    assert(!I->getParent());
    auto OldBB = CurrentBB;
    auto CondBB = BasicBlock::Create(Context, CondBlockName, &VectorFunc);
    auto NextBB =
        BasicBlock::Create(Context, CurrentBB->getName(), &VectorFunc);

    getBuilder().CreateCondBr(Cond, CondBB, NextBB);
    IRBuilder<> CondBuilder(CondBB);
    CondBuilder.Insert(I);
    CondBuilder.CreateBr(NextBB);

    CurrentBB = NextBB;
    if (I->getType()->isVoidTy()) {
      return nullptr;
    } else {
      auto Phi = getBuilder().CreatePHI(I->getType(), 2);
      auto Undef = UndefValue::get(I->getType());
      Phi->addIncoming(I, CondBB);
      Phi->addIncoming(Undef, OldBB);
      return Phi;
    }
  }

  void visitCallInst(CallInst &Call) {
    auto Callee = Call.getCalledFunction();
    SmallVector<Value *, 4> VectorArgs;
    for (auto &Arg : Call.arg_operands()) {
      VectorArgs.push_back(getVectorized(&*Arg));
    }
    if (auto VectorCallee = VectorizedFunctions.lookup(Callee)) {
      auto Mask = requireMask(&Call);
      VectorArgs.insert(VectorArgs.begin(), Mask);
      // Only do the call if any lanes need it. This can not only save
      // time for expensive functions, it also makes recursion work.
      auto MaskIntTy = IntegerType::get(Context, SIMD_WIDTH);
      auto Builder = getBuilder();
      auto NeedCall =
          Builder.CreateICmpNE(Builder.CreateBitCast(Mask, MaskIntTy),
                               ConstantInt::get(MaskIntTy, 0), "need_call");

      // TODO transplant attributes as appropriate
      // (some, especially argument attributes, may not apply)
      auto VectorCall = CallInst::Create(VectorCallee, VectorArgs);
      if (auto Result = insertConditional(NeedCall, VectorCall,
                                          "call." + Callee->getName())) {
        record(&Call, Result, MaskMode::Masked);
      }
    } else {
      // Slow path for functions that don't have vectorized versions
      auto Mask = requireMask(&Call);
      SmallVector<Value *, SIMD_WIDTH> LaneResults;
      for (uint64_t i = 0; i < SIMD_WIDTH; ++i) {
        auto MaskBit = getBuilder().CreateExtractElement(Mask, i);
        SmallVector<Value *, 8> LaneArgs;
        for (Value *VecArg : VectorArgs) {
          LaneArgs.push_back(getBuilder().CreateExtractElement(VecArg, i));
        }
        auto LaneCall = CallInst::Create(Callee, LaneArgs);
        LaneResults.push_back(
            insertConditional(MaskBit, LaneCall, "call." + Callee->getName()));
      }
      if (!Call.getType()->isVoidTy()) {
        record(&Call, assembleVector(&Call, LaneResults), MaskMode::Masked);
      }
    }
  }

  void visitCmpInst(CmpInst &Cmp) {
    auto LHS = getVectorized(Cmp.getOperand(0));
    auto RHS = getVectorized(Cmp.getOperand(1));
    auto VecCmp =
        CmpInst::Create(Cmp.getOpcode(), Cmp.getPredicate(), LHS, RHS);
    record(&Cmp, VecCmp, MaskMode::Unmasked);
  }

  void visitBinaryOperator(BinaryOperator &Op) {
    auto LHS = getVectorized(Op.getOperand(0));
    auto RHS = getVectorized(Op.getOperand(1));
    auto VecOp = BinaryOperator::Create(Op.getOpcode(), LHS, RHS);
    // TODO division probably needs to be masked?
    record(&Op, VecOp, MaskMode::Unmasked);
  }

  void visitCastInst(CastInst &Cast) {
    auto VecVal = getVectorized(Cast.getOperand(0));
    auto VecCast = CastInst::Create(Cast.getOpcode(), VecVal,
                                    getVectorType(Cast.getDestTy()));
    record(&Cast, VecCast, MaskMode::Unmasked);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    SmallVector<Value *, 16> IdxList;
    for (auto It = GEP.idx_begin(), End = GEP.idx_end(); It != End; ++It) {
      IdxList.push_back(getVectorized(It->get()));
    }
    auto PtrVec = getVectorized(GEP.getPointerOperand());
    auto VecGEP =
        GetElementPtrInst::Create(GEP.getSourceElementType(), PtrVec, IdxList);
    record(&GEP, VecGEP, MaskMode::Unmasked);
  }

  void visitLoadInst(LoadInst &Load) {
    auto PtrVec = getVectorized(Load.getOperand(0));
    auto Mask = tryGetMask(&Load);
    bool IsMasked = Mask != nullptr;
    // TODO consider always masking loads even if they are save to speculate
    if (!Mask) {
      Mask = ConstantVector::getSplat(SIMD_WIDTH, getBuilder().getTrue());
    }
    auto Gather =
        getBuilder().CreateMaskedGather(PtrVec, Load.getAlignment(), Mask);
    record(&Load, Gather, IsMasked ? MaskMode::Masked : MaskMode::Unmasked);
  }

  void visitStoreInst(StoreInst &Store) {
    auto ValVec = getVectorized(Store.getOperand(0));
    auto PtrVec = getVectorized(Store.getOperand(1));
    auto Scatter = getBuilder().CreateMaskedScatter(
        ValVec, PtrVec, Store.getAlignment(), requireMask(&Store));
    record(&Store, Scatter, MaskMode::Masked /* TODO is this right? */);
  }

  void visitSelectInst(SelectInst &Sel) {
    auto Mask = getVectorized(Sel.getCondition());
    auto TrueVec = getVectorized(Sel.getTrueValue());
    auto FalseVec = getVectorized(Sel.getFalseValue());
    auto VecSel = SelectInst::Create(Mask, TrueVec, FalseVec);
    record(&Sel, VecSel, MaskMode::Unmasked);
  }

  void visitPHINode(PHINode &Phi) {
    // assert(LI.isLoopHeader(Phi.getParent()) && Phi.getType()->isIntegerTy(1)
    // &&
    //       "Vectorizing PHI that does not seem to be a loop mask");
    // Since this ought to be a loop mask, there is nothing to vectorize
    auto NewPhi = getBuilder().CreatePHI(getVectorType(Phi.getType()),
                                         Phi.getNumIncomingValues());
    for (unsigned i = 0; i < Phi.getNumIncomingValues(); ++i) {
      auto VecIncoming = getVectorized(Phi.getIncomingValue(i));
      NewPhi->addIncoming(VecIncoming, BlockMap[Phi.getIncomingBlock(i)]);
    }
    record(&Phi, NewPhi, MaskMode::Ignore);
  }

  void visitBranchInst(BranchInst &Br) {
    // As we're post-linearization, there is nothing to vectorize.
    assert(!Br.isConditional() && "TODO need to support this for loops");
    auto VecTarget = BlockMap[Br.getSuccessor(0)];
    auto Br2 = getBuilder().CreateBr(VecTarget);
    record(&Br, Br2, MaskMode::Ignore);
  }

  Value *tryGetMask(Instruction *I) {
    if (auto MD = I->getMetadata(CONDITION_METADATA_ID)) {
      auto Cond = cast<ValueAsMetadata>(&*MD->getOperand(0))->getValue();
      return getVectorized(Cond);
    }
    return nullptr;
  }

  Value *requireMask(Instruction *I) {
    if (auto Mask = tryGetMask(I)) {
      return Mask;
    } else {
      std::string msg;
      raw_string_ostream Msg(msg);
      Msg << "LowerSPMD: condition metadata missing on: ";
      I->print(Msg);
      report_fatal_error(Msg.str());
    }
  }

  void record(Instruction *Scalar, Value *Vectorized, MaskMode Mode,
              BasicBlock *BB = nullptr) {
    if (auto VecInst = dyn_cast<Instruction>(Vectorized)) {
      if (!BB) {
        BB = CurrentBB;
      }

      if (VecInst->getParent()) {
        assert(VecInst->getParent() == BB);
      } else {
        BB->getInstList().push_back(VecInst);
      }
    }
    Vectorized->setName(Scalar->getName());

    DEBUG(dbgs() << "Replacing scalar instruction:\n");
    DEBUG(Scalar->print(dbgs()));
    DEBUG(dbgs() << "\nwith vectorized instruction:\n");
    DEBUG(Vectorized->print(dbgs()));
    DEBUG(dbgs() << "\n");

    switch (Mode) {
    case MaskMode::Unmasked:
      assert(!Scalar->getMetadata(CONDITION_METADATA_ID) &&
             "Instruction supposedly masked, but has no condition");
      break;
    case MaskMode::Masked:
      assert(Scalar->getMetadata(CONDITION_METADATA_ID) &&
             "Instruction supposedly unmasked, but has condition");
      break;
    case MaskMode::Ignore:
      /* Nothing */
      break;
    }
    Scalar2Vector[Scalar] = Vectorized;
  }

  Value *getVectorized(Value *Scalar) {
    if (auto I = dyn_cast<Instruction>(Scalar)) {
      // assert(Scalar2Vector.count(I));
      if (!Scalar2Vector.count(I))
        return UndefValue::get(getVectorType(Scalar->getType())); // XXX
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

struct Linearizer {
  Conditions &Conds;
  Function *F;
  LLVMContext &Context;
  BasicBlock *LastBB = nullptr;
  ReturnInst *SoleReturn = nullptr;

  Linearizer(Conditions &Conds, Function *F)
      : Conds(Conds), F(F), Context(F->getContext()) {}

  // Duplicates scc_iterator::hasLoop, but for various reasons this class
  // can't work with those iterators directly
  bool hasLoop(const std::vector<BasicBlock *> &SCC) {
    if (SCC.size() > 1) {
      return true;
    }
    auto BB = SCC[0];
    auto Succs = successors(BB);
    return std::any_of(Succs.begin(), Succs.end(),
                       [&](BasicBlock *Succ) { return Succ == BB; });
  }

  void visitSCC(const std::vector<BasicBlock *> &SCC) {
    if (hasLoop(SCC)) {
      visitLoop(SCC);
    } else {
      visitSingleBlock(SCC[0]);
    }
  }

  void visitSingleBlock(BasicBlock *BB) {
    auto BlockCond = Conds.forBlock(BB);

    if (LastBB) {
      IRBuilder<> Builder(LastBB);
      Builder.CreateBr(BB);
    }
    LastBB = BB;
    for (auto Iter = BB->begin(); Iter != BB->end();) {
      Instruction *I = &*Iter;
      ++Iter;
      if (auto *Phi = dyn_cast<PHINode>(I)) {
        if (Phi != Conds.forBlock(BB)) // XXX
          PhiToSelect(Phi);
      } else if (auto Ret = dyn_cast<ReturnInst>(I)) {
        // The constructor ensured there's only one return.
        // Because of unreachable instructions, the BB with the return may not
        // be last in the topological order, so to ensure the return is at the
        // end of the linearized function we remember it and put it at the very
        // end after we've visited all BBs.
        assert(!SoleReturn);
        SoleReturn = Ret;
        I->removeFromParent();
      } else if (isa<TerminatorInst>(I)) {
        I->eraseFromParent();
      } else if (!isSafeToSpeculativelyExecute(I)) {
        markAsConditional(I, BlockCond);
      }
    }
  }

  void visitLoop(const std::vector<BasicBlock *> &SCC) {
    visitSingleBlock(SCC[0]);
  }

  void markAsConditional(Instruction *Inst, Value *Condition) {
    assert(!Inst->getMetadata(CONDITION_METADATA_ID) &&
           "Instruction already conditional");
    auto CondMD = ValueAsMetadata::get(Condition);
    Inst->setMetadata(CONDITION_METADATA_ID, MDNode::get(Context, CondMD));
  }

  void PhiToSelect(PHINode *Phi) {
    auto Result = Phi->getIncomingValue(0);
    auto BB = Phi->getParent();
    auto Cond = Conds.forEdge(Phi->getIncomingBlock(0), BB);
    IRBuilder<> Builder(Phi);
    for (unsigned i = 1; i < Phi->getNumIncomingValues(); ++i) {
      Result = Builder.CreateSelect(Cond, Result, Phi->getIncomingValue(i),
                                    Phi->getName());
      Cond = Conds.forEdge(Phi->getIncomingBlock(i), BB);
    }
    Phi->replaceAllUsesWith(Result);
    Phi->eraseFromParent();
  }

  void finish() {
    assert(SoleReturn);
    assert(LastBB);
    LastBB->getInstList().push_back(SoleReturn);
  }
};

// Whether the use is across loop boundaries (including loop *iterations*).
bool isCrossLoopUse(Loop *L, Use &U) {
  if (auto Phi = dyn_cast<PHINode>(U.getUser())) {
    // Uses across loop *iterations* must happen in a phi in the loop *header*:
    // (1) It must be a phi, because defs from earlier iterations don't dominate
    // uses in later iterations.
    // (2) The phi must be in the loop header, as phis *within* the loop can't
    // distinguish between the loop back edge and other edges entering the loop.
    return Phi->getParent() == L->getHeader();
  }
  // The only other way to have a use across loop boundaries is to
  // use the def *outside* the loop, which we can query directly.
  return !L->contains(cast<Instruction>(U.getUser()));
}

struct FunctionVectorizer {
  LLVMContext &Context;
  Function *LinearFunc;
  Function *VectorFunc;
  const DenseMap<Function *, Function *> &VectorizedFunctions;
  LoopInfo LI;

  FunctionVectorizer(
      Function &F, Function &VF,
      const DenseMap<Function *, Function *> &VectorizedFunctions)
      : Context(F.getContext()), VectorFunc(&VF),
        VectorizedFunctions(VectorizedFunctions) {
    // We need to clone the source function already before mask computation to
    // insert the i1 argument (the condition for the entry block). We then
    // linearize that copy in-place. Vectorization, however, creates a new
    // function (it can't easily work in-place because the types of all
    // instructions change).
    // TODO find a better name than "LinearFunc"
    ValueToValueMapTy ArgMapping;
    SmallVector<ReturnInst *, 1> Returns;
    LinearFunc = prepareScalarFunction(F, ArgMapping);
    CloneFunctionInto(LinearFunc, &F, ArgMapping,
                      /* ModuleLevelChanges: */ false, Returns);

    if (Returns.size() != 1) {
      report_fatal_error("LowerSPMD: cannot vectorize function with multiple "
                         "returns, run mergereturn");
    }

    DominatorTree DomTree(*LinearFunc);
    LI = LoopInfo(DomTree);
    if (!LI.empty()) {
      DEBUG(LI.print(dbgs()));
      for (Loop *L : LI) {
        DEBUG(L->dumpVerbose());
        DEBUG(dbgs() << "Loop live values for " << L->getName() << ":\n");
        for (auto LLV : loopLiveValues(L)) {
          DEBUG(LLV->dump());
        }
      }
      DEBUG(dbgs() << '\n');
    }
  }

  DenseSet<Instruction *> loopLiveValues(Loop *L) {
    DenseSet<Instruction *> LiveValues;
    for (auto BB : L->blocks()) {
      for (auto &I : *BB) {
        if (std::any_of(I.use_begin(), I.use_end(),
                        [=](Use &U) { return isCrossLoopUse(L, U); })) {
          LiveValues.insert(&I);
        }
      }
    }
    return LiveValues;
  }

  // Create an empty function with the same prototype as F, except that an
  // additional i1 argument (the condition for the entry block) is inserted
  // before the first argument.
  // Also fills out a mapping from the input function's arguments to the
  // output
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

    for (BasicBlock &BB : *LinearFunc) {
      IRBuilder<> Builder(&BB, BB.getFirstInsertionPt());
      Value *BlockCond = nullptr;
      if (LI.isLoopHeader(&BB)) {
        // The phi we're inserting will *not* be converted into a select later
        // on, so we make sure it's *before* those other phi nodes.
        Builder.SetInsertPoint(&BB, BB.begin());
        auto Phi = Builder.CreatePHI(Builder.getInt1Ty(),
                                     2 /* TODO this is a guess */);
        for (auto Pred : predecessors(&BB)) {
          Phi->addIncoming(Conds.forEdge(Pred, &BB), Pred);
        }
        Phi->setName(BB.getName() + ".exec");
        BlockCond = Phi;
      } else {
        for (auto Pred : predecessors(&BB)) {
          auto IncomingCond = Conds.forEdge(Pred, &BB);
          if (!BlockCond) {
            BlockCond = IncomingCond;
          } else {
            BlockCond = Builder.CreateOr(BlockCond, IncomingCond,
                                         BB.getName() + ".exec");
          }
        }
      }
      if (!BlockCond) {
        assert(&BB == &LinearFunc->getEntryBlock() && "unreachable block");
        // The condition for the entry block is passed in as the first
        // argument,
        BlockCond = &*LinearFunc->arg_begin();
      }
      assert(BlockCond->getType() == Builder.getInt1Ty());
      Conds.forBlock(&BB) = BlockCond;
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

  void linearizeCFG(Conditions &Conds) {
    std::vector<std::vector<BasicBlock *>> SCCBBs;
    for (auto I = scc_begin(LinearFunc); !I.isAtEnd(); ++I) {
      DEBUG(dbgs() << "SCC: ");
      for (auto BB : *I) {
        DEBUG(dbgs() << BB->getName() << "@" << BB << " ");
      }
      DEBUG(dbgs() << "\n");
      SCCBBs.push_back(*I);
    }
    // XXX scc_iterator on the Inverse graph doesn't seem to work?!
    std::reverse(SCCBBs.begin(), SCCBBs.end());

    Linearizer Lin(Conds, LinearFunc);

    for (auto const &SCC : SCCBBs) {
      Lin.visitSCC(SCC);
    }

    Lin.finish();
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
    InstVectorizeVisitor IVV(*VectorFunc, LI, VectorizedFunctions);
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
