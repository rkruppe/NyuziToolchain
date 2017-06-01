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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

#define DEBUG_TYPE "lowerspmd"

namespace {

const unsigned SIMD_WIDTH = 16;
const char *CONDITION_METADATA_ID = "spmd-conditional";

template <typename T> using BBMap = DenseMap<const BasicBlock *, T>;

// This table holds the various scalar conditions needed to linearize a
// function. Because it is built and used before vectorization, it stores i1
// *conditions*, which are often redundant, and may be computed in an
// unnecessarily convoluted way even when not redundant. This is so that
// vectorization does not have to deal with control flow very much and can
// simply vectorize every instruction 1:1.
class ConditionTable {
  using VH = TrackingVH<Value>;

  // For each BB, stores the block condition, i.e., the scalar equivalent of the
  // mask that encodes which lanes are executing a block. For loop headers this
  // is exactly the "loop condition" (i.e., scalar loop mask).
  // In scalar terms, it is an i1 that is true iff the block is executed
  // (although it is only actually computed in the block itself).
  BBMap<VH> BlockConditions;

  // For each pair of blocks (BB1, BB2) that have an edge between them, stores
  // an i1 that is true iff there was a jump from BB1 to BB2. This may be a
  // combination of the branch condition and the source block's condition, or a
  // loop exit mask.
  // FIXME the representation is fine, but the mask generation will fall apart
  // if there are multiple edges between two blocks. Other code may also be
  // rewritten to support this.
  BBMap<BBMap<VH>> JumpConditions;

  // Like the jump condition, but not accumulated across iterations in case of
  // loop exits. In other words, this condition indicates whether the *current*
  // iteration left the loop. This is only needed for loop results.
  BBMap<BBMap<VH>> SingleJumpConditions;

  // For each loop, stores the combined loop exit condition, i.e. the scalar
  // equivalent of the combined loop exit mask which encodes which lanes left
  // the loop in the current iteration.
  DenseMap<Loop *, VH> CombinedLoopExitConditions;

  Function &F;

public:
  ConditionTable(Function &F) : F(F) {}

  void addBlock(const BasicBlock *BB, Value *Cond) {
    assert(BlockConditions.count(BB) == 0 && "Added block twice");
    BlockConditions[BB] = Cond;
  }

  void addJump(const BasicBlock *From, const BasicBlock *To, Value *Cond) {
    assert(JumpConditions[From].count(To) == 0 && "Added edge twice");
    JumpConditions[From][To] = Cond;
  }

  void addSingleJump(const BasicBlock *From, const BasicBlock *To,
                     Value *Cond) {
    assert(SingleJumpConditions[From].count(To) == 0 && "Added edge twice");
    SingleJumpConditions[From][To] = Cond;
  }

  void addCombinedLoopExit(Loop *L, Value *Cond) {
    assert(CombinedLoopExitConditions.count(L) == 0 && "Added loop twice");
    CombinedLoopExitConditions[L] = Cond;
  }

  Value *blockCondition(const BasicBlock *BB) const {
    assert(BlockConditions.count(BB) && "Unknown block");
    return BlockConditions.lookup(BB);
  }

  Value *jumpCondition(const BasicBlock *From, const BasicBlock *To) const {
    assert(JumpConditions.lookup(From).count(To) && "Unknown jump");
    return JumpConditions.lookup(From).lookup(To);
  }

  Value *singleJumpCondition(const BasicBlock *From,
                             const BasicBlock *To) const {
    assert(SingleJumpConditions.lookup(From).count(To) && "Unknown jump");
    return SingleJumpConditions.lookup(From).lookup(To);
  }

  Value *combinedLoopExitCondition(Loop *L) const {
    assert(CombinedLoopExitConditions.count(L) && "Unknown loop");
    return CombinedLoopExitConditions.lookup(L);
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
      Dump(BlockConditions[&BB]);
    }
    for (auto &From : F) {
      for (auto &To : F) {
        if (JumpConditions[&From].count(&To) == 0) {
          continue;
        }
        os << "Jump condition " << From.getName() << " -> " << To.getName()
           << ": ";
        Dump(JumpConditions[&From][&To]);
      }
    }
    // TODO print combined loop exit conditions
  }
};

bool isLoopExit(BasicBlock *From, BasicBlock *To, LoopInfo &LI) {
  return LI.getLoopDepth(From) > LI.getLoopDepth(To);
}

Value *createLoopExitCondition(BasicBlock *From, BasicBlock *To, Value *Cond,
                               ConditionTable &CT, LoopInfo &LI) {
  // Create (1) an "update operation" at the exit that ORs the condition for
  // the actual jump with (2) a phi in the loop header that preserves the exit
  // condition from the previous iteration.
  // (1) becomes the real jump condition.
  // If the jump breaks out of multiple loops at once, we need to accumulate the
  // exit mask over all outer loop iterations, so we create phis in parent
  // loops up until the loop level right below the target of the exit.

  Loop *SourceLoop = LI.getLoopFor(From);
  Loop *TargetLoop = LI.getLoopFor(To);

  DenseMap<Loop *, PHINode *> Phis;
  // First create the phis for all loop levels
  for (Loop *L = SourceLoop; L != TargetLoop; L = L->getParentLoop()) {
    assert(L && "loop exit not nested in a parent loop??");
    IRBuilder<> PhiBuilder(&L->getHeader()->front());
    auto Phi = PhiBuilder.CreatePHI(PhiBuilder.getInt1Ty(), 2);
    Phis[L] = Phi;
  }

  IRBuilder<> UpdateBuilder(From->getTerminator());
  auto Update = UpdateBuilder.CreateOr(Phis[SourceLoop], Cond);

  // Now fill in the phis
  Value *False = UpdateBuilder.getFalse();
  for (Loop *L = SourceLoop; L != TargetLoop; L = L->getParentLoop()) {
    assert(L && "loop exit not nested in a parent loop??");
    Loop *ParentLoop = L->getParentLoop();
    auto Phi = Phis[L];
    auto BackedgeV = L == SourceLoop ? static_cast<Value *>(Update) : Phi;
    auto PreheaderV = ParentLoop == TargetLoop ? False : Phis[ParentLoop];
    Phi->addIncoming(BackedgeV, L->getLoopLatch());
    Phi->addIncoming(PreheaderV, L->getLoopPreheader());
  }
  return Update;
}

void createJumpCondition(BasicBlock *From, BasicBlock *To, Value *Cond,
                         ConditionTable &CT, LoopInfo &LI) {
  DEBUG(dbgs() << "Jump " << From->getName() << " -> " << To->getName()
               << " with condition: " << *Cond << "\n");
  CT.addSingleJump(From, To, Cond);
  if (Cond->getName().empty()) {
    Cond->setName(From->getName() + ".to." + To->getName());
  }
  if (isLoopExit(From, To, LI)) {
    Cond = createLoopExitCondition(From, To, Cond, CT, LI);
  }
  CT.addJump(From, To, Cond);
}

void createJumpConditionsFrom(BasicBlock *BB, ConditionTable &CT,
                              LoopInfo &LI) {
  auto *BlockCond = CT.blockCondition(BB);
  auto *Terminator = BB->getTerminator();
  if (isa<UnreachableInst>(Terminator) || isa<ReturnInst>(Terminator)) {
    // No successors => nothing to do.
    return;
  }
  auto *Br = cast<BranchInst>(Terminator);
  // TODO check up-front whether all terminators are unreachable/return/br
  if (Br->isConditional()) {
    IRBuilder<> Builder(Br);
    auto *Cond0 = Br->getCondition();
    auto *Cond1 = Builder.CreateNot(Cond0, "not." + Cond0->getName());
    createJumpCondition(BB, Br->getSuccessor(0),
                        Builder.CreateAnd(BlockCond, Cond0), CT, LI);
    createJumpCondition(BB, Br->getSuccessor(1),
                        Builder.CreateAnd(BlockCond, Cond1), CT, LI);
  } else {
    // Single successor => just re-use the block condition.
    createJumpCondition(BB, Br->getSuccessor(0), BlockCond, CT, LI);
  }
}

Value *createBlockCondition(BasicBlock *BB, const ConditionTable &CT,
                            LoopInfo &LI) {
  IRBuilder<> Builder(BB, BB->getFirstInsertionPt());
  Value *Cond = nullptr;
  if (LI.isLoopHeader(BB)) {
    auto Phi = Builder.CreatePHI(Builder.getInt1Ty(), 2);
    for (auto Pred : predecessors(BB)) {
      Phi->addIncoming(CT.jumpCondition(Pred, BB), Pred);
    }
    Cond = Phi;
  } else {
    for (auto Pred : predecessors(BB)) {
      auto JumpCond = CT.jumpCondition(Pred, BB);
      Cond = Cond ? Builder.CreateOr(Cond, JumpCond) : JumpCond;
    }
  }
  if (!Cond) {
    auto F = BB->getParent();
    assert(BB == &F->getEntryBlock() && "unreachable block");
    // The entry block condition is pass in as first argument.
    Cond = &*F->arg_begin();
  }
  Cond->setName(BB->getName() + ".exec");
  assert(Cond->getType() == Builder.getInt1Ty());
  return Cond;
}

void combinedLoopExitCondition(Loop *L, ConditionTable &CT) {
  IRBuilder<> Builder(L->getLoopLatch()->getTerminator());
  Value *Combined = Builder.getFalse();
  SmallVector<Loop::Edge, 4> Exits;
  L->getExitEdges(Exits);
  for (Loop::Edge E : Exits) {
    Value *ExitCond = CT.singleJumpCondition(E.first, E.second);
    Combined = Combined ? Builder.CreateOr(Combined, ExitCond) : ExitCond;
  }
  if (Combined->getName().empty()) {
    Combined->setName(L->getHeader()->getName() + ".combined.exit");
  }
  CT.addCombinedLoopExit(L, Combined);
}

ConditionTable createConditions(Function &F, LoopInfo &LI) {
  ConditionTable CT(F);

  // There are cyclic dependencies between block and jump conditions, so we
  // break the cycle by introducing placeholders for the block conditions which
  // are later RAUW'd the real block conditions.
  Value *False = ConstantInt::getFalse(F.getContext());
  for (auto &BB : F) {
    Instruction *ip = &*BB.getFirstInsertionPt();
    Instruction *Placeholder =
        BinaryOperator::Create(BinaryOperator::Or, False, False,
                               "exec.placeholder." + BB.getName(), ip);
    CT.addBlock(&BB, Placeholder);
  }

  // Now we create the jump conditions, including loop exit conditions
  for (auto &BB : F) {
    createJumpConditionsFrom(&BB, CT, LI);
  }

  for (auto &BB : F) {
    auto Placeholder = cast<Instruction>(CT.blockCondition(&BB));
    auto RealBlockCond = createBlockCondition(&BB, CT, LI);
    Placeholder->replaceAllUsesWith(RealBlockCond);
    Placeholder->eraseFromParent();
  }

  // Finally, combine the loop exit conditions.
  for (Loop *L : LI) {
    combinedLoopExitCondition(L, CT);
  }

  return CT;
}

struct InstVectorizeVisitor : InstVisitor<InstVectorizeVisitor> {
  Function &VectorFunc;
  LLVMContext &Context;
  const LoopInfo &LI;
  const DenseMap<Function *, Function *> &VectorizedFunctions;

  DenseMap<Instruction *, Value *> Scalar2Vector;
  // Map each scalar BB to the (first) vectorized BB.
  BBMap<BasicBlock *> BlockMap;
  // Map each vectorized BB to the (last) vectorized BB resulting from the
  // original scalar block. Said "last block" contains the vectorized equivalent
  // of the scalar terminator, needed for vectorizing phis.
  // TODO better names, possibly even an abstraction for these maps
  BBMap<BasicBlock *> SplitBlockEnd;

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
      auto VecBB = BasicBlock::Create(Context, BB.getName(), &VectorFunc);
      BlockMap[&BB] = VecBB;
      SplitBlockEnd[VecBB] = VecBB;
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
      record(&Ret, VecReturn, MaskMode::Ignore);
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
    auto NextBB = BasicBlock::Create(Context, CurrentBB->getName() + ".cont",
                                     &VectorFunc);
    SplitBlockEnd[OldBB] = NextBB;

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
    assert(LI.isLoopHeader(Phi.getParent()));
    // Because this is a loop, many incoming values aren't vectorized already.
    // So we don't attempt to lift the incoming values just yet, instead waiting
    // until we encounter the terminator of the loop latch.
    auto VecPhi = getBuilder().CreatePHI(getVectorType(Phi.getType()),
                                         Phi.getNumIncomingValues());
    record(&Phi, VecPhi, MaskMode::Ignore);
  }

  void visitBranchInst(BranchInst &Br) {
    Loop *L = LI.getLoopFor(Br.getParent());
    bool IsLatch = L && L->getLoopLatch() == Br.getParent();

    if (Br.isConditional()) {
      assert(IsLatch &&
             "conditional branch outside loop latch post-linearization");
      auto LoopMask = getVectorized(Br.getCondition());
      auto Builder = getBuilder();
      // During linearization, loop latches were canonicalized so that the back
      // edge is taken if the condition is true. So if any bit of the mask is
      // true, we need another iteration.
      auto MaskIntTy = Builder.getIntNTy(SIMD_WIDTH);
      auto LoopMaskInt = Builder.CreateBitCast(LoopMask, MaskIntTy);
      auto AnyLaneContinues =
          Builder.CreateICmpNE(LoopMaskInt, ConstantInt::get(MaskIntTy, 0));
      auto VecHeader = BlockMap[Br.getSuccessor(0)],
           VecExit = BlockMap[Br.getSuccessor(1)];
      auto VecBr = Builder.CreateCondBr(AnyLaneContinues, VecHeader, VecExit);
      record(&Br, VecBr, MaskMode::Ignore);
    } else {
      auto VecTarget = BlockMap[Br.getSuccessor(0)];
      auto Br2 = getBuilder().CreateBr(VecTarget);
      record(&Br, Br2, MaskMode::Ignore);
    }

    if (IsLatch) {
      // Since we're finished with the loop now, we can also go back and fill in
      // the phis at the start of the loop.
      fixupPhis(*L->getHeader());
    }
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

  void fixupPhis(BasicBlock &BB) {
    for (Instruction &I : BB) {
      auto Phi = dyn_cast<PHINode>(&I);
      // All phis are at the start, so if we can stop at the first non-phi.
      if (!Phi) {
        break;
      }
      auto VecPhi = cast<PHINode>(getVectorized(Phi));
      for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
        auto VecIncoming = getVectorized(Phi->getIncomingValue(i));
        auto IncomingBB = BlockMap[Phi->getIncomingBlock(i)];
        VecPhi->addIncoming(VecIncoming, SplitBlockEnd[IncomingBB]);
      }
    }
  }
};

bool isReducible(const std::vector<BasicBlock *> &SCC, Loop *L) {
  if (!L) {
    // If there's no Loop object for some BB in the SCC, it's part of a weird
    // loop that is not reducible.
    return false;
  }
  return std::all_of(SCC.begin(), SCC.end(),
                     [=](BasicBlock *BB) { return L->contains(BB); });
}

struct ToposortState {
  LoopInfo &LI;
  std::vector<BasicBlock *> Output;
  SmallPtrSet<BasicBlock *, 16> Seen;
  BBMap<std::vector<BasicBlock *>> SubLoopOrders;

  ToposortState(LoopInfo &LI) : LI(LI) {}
};

// Toposort the blocks that *directly* belong to the loop, and insert
// sub-loop toposorts after their respective pre-headers.
void loopToposortRec(BasicBlock *BB, Loop *L, ToposortState &State) {
  DEBUG(dbgs() << "loopToposortRec: " << BB->getName() << "\n");
  State.Seen.insert(BB);

  // We are only interested in blocks *immediately* contained in the
  // current loop, so we don't record blocks from inner loops. But we still need
  // to traverse the inner loops, because some blocks of the outer loop may not
  // be reachable otherwise.
  if (State.LI.getLoopDepth(BB) == L->getLoopDepth()) {
    State.Output.push_back(BB);
    // If this is a pre-header, now is the time to insert the sub-loop's
    // blocks into the topological order.
    if (State.SubLoopOrders.count(BB)) {
      const auto &SubLoopOrder = State.SubLoopOrders[BB];
      State.Output.insert(State.Output.end(), SubLoopOrder.begin(),
                          SubLoopOrder.end());
    }
  }

  for (auto Succ : successors(BB)) {
    if (!State.Seen.count(Succ) && L->contains(Succ)) {
      // We may have to process BBs from inner loops (see above), but we never
      // need to consider blocks that are outside the loop, and we better not
      // consider any blocks inside the loop twice.
      loopToposortRec(Succ, L, State);
    }
  }
}

std::vector<BasicBlock *> loopToposort(Loop *L, LoopInfo &LI) {
  DEBUG(dbgs() << "loopToposort: processing " << *L);
  // We need to keep each nested loops "together", so we can't simply to a
  // toposort of the whole SCC that _ignores_ back edges. Instead, we
  // recursively determine a linear order of each sub-loop, and insert these
  // linearizations at the right places: right after the sub-loops' pre-headers.
  ToposortState State(LI);
  for (Loop *SubLoop : *L) {
    auto Preheader = SubLoop->getLoopPreheader();
    assert(Preheader && "loop must be in canonical form (missing pre-header)");
    State.SubLoopOrders[Preheader] = loopToposort(SubLoop, LI);
  }
  loopToposortRec(L->getHeader(), L, State);
  DEBUG(dbgs() << "loopToposort: finished " << *L);
  return std::move(State.Output);
}

Optional<std::vector<BasicBlock *>> findLinearOrder(Function *F, LoopInfo &LI) {
  std::vector<std::vector<BasicBlock *>> SCCsLinearized;
  for (auto I = scc_begin(F); !I.isAtEnd(); ++I) {
    const std::vector<BasicBlock *> &SCC = *I;
    if (I.hasLoop()) {
      Loop *L = LI.getLoopFor(SCC[0]);
      while (L && L->getParentLoop()) {
        L = L->getParentLoop();
      }
      if (!isReducible(SCC, L)) {
        return None;
      }
      // TODO check whether the loop is canonical
      SCCsLinearized.push_back(loopToposort(L, LI));
    } else {
      SCCsLinearized.push_back({SCC[0]});
    }
  }

  std::reverse(SCCsLinearized.begin(), SCCsLinearized.end());
  std::vector<BasicBlock *> LinearOrder;
  for (const auto &LinearOrderForSCC : SCCsLinearized) {
    LinearOrder.insert(LinearOrder.end(), LinearOrderForSCC.begin(),
                       LinearOrderForSCC.end());
  }
  return LinearOrder;
}

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

  void linearizeCFG(ConditionTable &CT) {
    std::vector<BasicBlock *> LinearOrder;
    if (auto LinearOrderOpt = findLinearOrder(LinearFunc, LI)) {
      LinearOrder = std::move(LinearOrderOpt.getValue());
    } else {
      report_fatal_error("TODO don't even go here");
    }

    for (size_t i = 0; i < LinearOrder.size(); ++i) {
      BasicBlock *BB = LinearOrder[i];

      Value *BlockCond = CT.blockCondition(BB);
      for (Instruction &I : *BB) {
        if (!isSafeToSpeculativelyExecute(&I)) {
          markAsConditional(&I, BlockCond);
        }
      }

      if (!LI.isLoopHeader(BB)) {
        PhisToSelect(BB, CT);
      }

      if (auto Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
        // Because of blocks ending with unreachable, the return instruction
        // may not come last in the topological ordering. So we forcibly put the
        // return into the last block.
        TerminatorInst *LastTerminator = LinearOrder.back()->getTerminator();
        if (Ret != LastTerminator) {
          assert(isa<UnreachableInst>(LastTerminator));
          ReplaceInstWithInst(LastTerminator, Ret);
          LastTerminator->eraseFromParent();
        }
      }

      if (i + 1 < LinearOrder.size()) {
        BasicBlock *NextBB = LinearOrder[i + 1];
        linearizeTerminator(BB, NextBB, CT);
      }
    }
  }

  // Turns phis into selects and records the masks for instructions
  // that need to be made conditional.
  void PhisToSelect(BasicBlock *BB, ConditionTable &CT) {
    // Need manual iterator fiddling because we remove the *current*
    // instruction, so we can't advance the iterator at the *end* of the loop
    // iteration as usual
    for (auto Iter = BB->begin(); Iter != BB->end();) {
      Instruction *I = &*Iter;
      ++Iter;
      if (auto *Phi = dyn_cast<PHINode>(I)) {
        PhiToSelect(Phi, CT);
      }
    }
  }

  void linearizeTerminator(BasicBlock *CurrentBB, BasicBlock *NextBB,
                           ConditionTable &CT) {
    Loop *L = LI.getLoopFor(CurrentBB);
    if (L && L->getLoopLatch() == CurrentBB) {
      BasicBlock *Header = L->getHeader();
      assert(!L->contains(NextBB));
      assert(Header);
      auto Br = cast<BranchInst>(CurrentBB->getTerminator());
      if (Br->isConditional()) {
        assert(Br->getSuccessor(0) == Header || Br->getSuccessor(1) == Header);
        assert(!L->contains(Br->getSuccessor(0)) ||
               !L->contains(Br->getSuccessor(1)));
      } else {
        assert(Br->getSuccessor(0) == Header);
      }
      if (Br->isConditional()) {
        // Canonicalize conditional branches so that the backedge happens on
        // `true` and on `false` we carry on with the linearization.
        Br->setSuccessor(0, Header);
        Br->setSuccessor(1, NextBB);
        Br->setCondition(CT.jumpCondition(CurrentBB, Header));
      }
    } else {
      ReplaceInstWithInst(CurrentBB->getTerminator(),
                          BranchInst::Create(NextBB));
    }
  }

  void markAsConditional(Instruction *Inst, Value *Condition) {
    assert(!Inst->getMetadata(CONDITION_METADATA_ID) &&
           "Instruction already conditional");
    auto CondMD = ValueAsMetadata::get(Condition);
    Inst->setMetadata(CONDITION_METADATA_ID, MDNode::get(Context, CondMD));
  }

  void PhiToSelect(PHINode *Phi, ConditionTable &CT) {
    auto Result = Phi->getIncomingValue(0);
    auto BB = Phi->getParent();
    auto Cond = CT.jumpCondition(Phi->getIncomingBlock(0), BB);
    IRBuilder<> Builder(Phi);
    for (unsigned i = 1; i < Phi->getNumIncomingValues(); ++i) {
      Result = Builder.CreateSelect(Cond, Result, Phi->getIncomingValue(i),
                                    Phi->getName());
      Cond = CT.jumpCondition(Phi->getIncomingBlock(i), BB);
    }
    Phi->replaceAllUsesWith(Result);
    Phi->eraseFromParent();
  }

  Value *createLoopResultFor(Instruction &I, const ConditionTable &CT) {
    Loop *DefLoop = LI.getLoopFor(I.getParent());
    DenseMap<Loop *, PHINode *> Phis;
    // First, insert phis in all loop levels, but don't fill them yet.
    for (Loop *L = DefLoop; L; L = L->getParentLoop()) {
      IRBuilder<> PhiBuilder(&*L->getHeader()->getFirstInsertionPt());
      Phis[L] = PhiBuilder.CreatePHI(I.getType(), 2);
    }

    // Then, insert the actual update operation in the latch
    auto CombExitCond = CT.combinedLoopExitCondition(DefLoop);
    IRBuilder<> UpdateBuilder(DefLoop->getLoopLatch()->getTerminator());
    auto Update = UpdateBuilder.CreateSelect(CombExitCond, &I, Phis[DefLoop]);
    Update->setName("loopres." + I.getName());

    // Now fill in the phis.
    for (Loop *L = DefLoop; L; L = L->getParentLoop()) {
      Loop *ParentLoop = L->getParentLoop();
      auto ParentValue = ParentLoop ? static_cast<Value *>(Phis[ParentLoop])
                                    : UndefValue::get(I.getType());
      Phis[L]->addIncoming(ParentValue, L->getLoopPreheader());
      Phis[L]->addIncoming(Update, L->getLoopLatch());
    }
    // Finally, all uses outside the loop should refer to the update:
    return Update;
  }

  Value *rewriteUsesToLoopResult(Instruction &I, const ConditionTable &CT) {
    DEBUG(dbgs() << "rewriteUsesToLoopResult: " << I << "\n");
    Loop *L = LI.getLoopFor(I.getParent());
    Value *LoopResult = nullptr;
    for (Use &U : I.uses()) {
      auto UserInst = dyn_cast<Instruction>(U.getUser());
      if (!UserInst || L->contains(UserInst)) {
        continue;
      }
      DEBUG(dbgs() << "\tGonna replace use: " << *UserInst << "\n");
      // Create loop result vectors for all loop levels in one shot, even if we
      // may not need  all of them. This could be a bit wasteful, but it avoids
      // a bunch of complexity and loops usually aren't very deeply nested.
      if (!LoopResult) {
        LoopResult = createLoopResultFor(I, CT);
      }
      U.set(LoopResult);
    }
    return LoopResult;
  }

  void insertLoopResultsRec(Loop *L, const ConditionTable &CT) {
    for (Loop *SubLoop : *L) {
      insertLoopResultsRec(SubLoop, CT);
    }
    // We're inserting instructions computing loop results as we go, so
    // we need to remember which instructions we inserted to avoid infinite
    // loops.
    SmallPtrSet<Value *, 8> LoopResults;
    // TODO does this visit instructions in nested loops multiple times?
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (!LoopResults.count(&I)) {
          auto LoopResult = rewriteUsesToLoopResult(I, CT);
          if (LoopResult) {
            LoopResults.insert(LoopResult);
          }
        }
      }
    }
  }

  void insertLoopResults(const ConditionTable &CT) {
    DEBUG(LinearFunc->dump());
    for (Loop *L : LI) {
      insertLoopResultsRec(L, CT);
    }
  }

  Function *run() {
    ConditionTable CT = createConditions(*LinearFunc, LI);

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(CT.print(dbgs()));
    DEBUG(dbgs() << "\n");

    DEBUG(dbgs() << "===================================================\n");
    insertLoopResults(CT);

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(LinearFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    linearizeCFG(CT);
    DEBUG(LinearFunc->print(dbgs()));

    // TODO handle cross loop uses

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
