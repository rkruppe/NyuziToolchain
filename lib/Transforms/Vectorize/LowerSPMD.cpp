#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
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
#include <deque>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "lowerspmd"
STATISTIC(NumFunctionsSPMDVectorized,
          "Number of functions that were vectorized by LowerSPMD");
STATISTIC(NumFunctionsNotSPMDVectorizable,
          "Number of functions that could not be vectorized by LowerSPMD");

namespace {

const unsigned SIMD_WIDTH = 16;

template <typename T> using BBMap = DenseMap<const BasicBlock *, T>;
using Edge = std::pair<BasicBlock *, BasicBlock *>;

// This utility function only exists because Loop::getExitEdges gives out const
// pointers and apparently doesn't have a non-const version.
void getExitEdges(Loop *L, SmallVectorImpl<Edge> &ExitEdges) {
  for (auto BB : L->blocks()) {
    for (auto Succ : successors(BB)) {
      if (!L->contains(Succ)) {
        ExitEdges.emplace_back(BB, Succ);
      }
    }
  }
}

// This table holds the various masks needed to linearize a function. Because it
// is built and used before vectorization, it stores scalar booleans, but we
// will mostly treat them like ordinary masks for a vector width of 1.
// Their semantics and how they are computed is specifically designed to do The
// Right Thing after vectorization.
class MaskTable {
  using VH = TrackingVH<Value>;
  template <typename T>
  using EdgeMap =
      DenseMap<std::pair<const BasicBlock *, const BasicBlock *>, T>;

  // For each BB, stores the block mask, which lanes are executing the block.
  BBMap<VH> BlockMasks;

  // For each pair of blocks (BB1 -> BB2) that have an edge between them, this
  // mask tells us which lanes took that edge to arrive at BB2. For loop exits
  // (where BB1 may be executed several times for every time BB2 is executed)
  // this mask accumulates over the loop iterations.
  // FIXME the representation is fine, but the mask generation will fall apart
  // if there are multiple edges between two blocks. Other code may also need to
  // be rewritten to support this.
  EdgeMap<VH> EdgeMasks;

  // Like the edge mask, but not accumulated across iterations in case of loop
  // exits. In other words, this mask indicates whether the *current*
  // iteration left the loop. This is only needed for loop results.
  // FIXME see EdgeMasks re: multiple edges between a pair of blocks
  EdgeMap<VH> SingleJumpMasks;

  // The loop exit mask for a single exit with respect to a loop L.
  // This mask is accumulated over all iterations of child loops of L, but
  // unlike the edge mask for (From, To) it is not accumulated over
  // iterations of L and parent loops of L.
  EdgeMap<DenseMap<Loop *, VH>> LoopExitMasks;

  // For each loop, stores the combined loop exit mask, i.e. the scalar
  // equivalent of the combined loop exit mask which encodes which lanes left
  // the loop in the current iteration.
  DenseMap<Loop *, VH> CombinedLoopExitMasks;

  Function &F;

public:
  MaskTable(Function &F) : F(F) {}

  void addBlock(const BasicBlock *BB, Value *Mask) {
    assert(BlockMasks.count(BB) == 0 && "Added block twice");
    BlockMasks[BB] = Mask;
  }

  void addEdge(const BasicBlock *From, const BasicBlock *To, Value *Mask) {
    auto E = std::make_pair(From, To);
    assert(EdgeMasks.count(E) == 0 && "Added edge twice");
    EdgeMasks[E] = Mask;
  }

  void addSingleJump(const BasicBlock *From, const BasicBlock *To,
                     Value *Mask) {
    auto E = std::make_pair(From, To);
    assert(SingleJumpMasks.count(E) == 0 && "Added edge twice");
    SingleJumpMasks[E] = Mask;
  }

  void addLoopExit(Loop *L, const BasicBlock *From, const BasicBlock *To,
                   Value *Mask) {
    auto E = std::make_pair(From, To);
    // FIXME the .lookup copies the entire inner map
    assert(LoopExitMasks.lookup(E).count(L) == 0 && "Added loop exit twice");
    LoopExitMasks[E][L] = Mask;
  }

  void addCombinedLoopExit(Loop *L, Value *Mask) {
    assert(CombinedLoopExitMasks.count(L) == 0 && "Added loop twice");
    CombinedLoopExitMasks[L] = Mask;
  }

  Value *blockMask(const BasicBlock *BB) const {
    assert(BlockMasks.count(BB) && "Unknown block");
    return BlockMasks.lookup(BB);
  }

  Value *edgeMask(const BasicBlock *From, const BasicBlock *To) const {
    auto E = std::make_pair(From, To);
    assert(EdgeMasks.count(E) && "Unknown edge");
    return EdgeMasks.lookup(E);
  }

  Value *exitMask(Loop *L, const BasicBlock *From, const BasicBlock *To) const {
    auto E = std::make_pair(From, To);
    // FIXME lookup copies the entire inner map
    const auto &TableForEdge = LoopExitMasks.lookup(E);
    assert(TableForEdge.count(L) && "Unknown exit");
    return TableForEdge.lookup(L);
  }

  Value *singleJumpMask(const BasicBlock *From, const BasicBlock *To) const {
    auto E = std::make_pair(From, To);
    assert(SingleJumpMasks.count(E) && "Unknown edge");
    return SingleJumpMasks.lookup(E);
  }

  Value *combinedLoopExitMask(Loop *L) const {
    assert(CombinedLoopExitMasks.count(L) && "Unknown loop");
    return CombinedLoopExitMasks.lookup(L);
  }

  void print(raw_ostream &os) {
    auto Dump = [&](Value *V) {
      if (isa<Instruction>(V)) {
        os << V->getName() << '\n';
      } else {
        V->print(os);
        os << "\n";
      }
    };
    for (BasicBlock &BB : F) {
      os << "Block mask " << BB.getName() << ": ";
      Dump(BlockMasks[&BB]);
    }
    for (auto &From : F) {
      for (auto To : successors(&From)) {
        os << "Edge mask " << From.getName() << " -> " << To->getName() << ": ";
        Dump(EdgeMasks[std::make_pair(&From, To)]);
      }
    }
    for (auto KVPair : CombinedLoopExitMasks) {
      os << "Combined loop exit mask for "
         << KVPair.first->getHeader()->getName() << ": " << *KVPair.second
         << "\n";
    }
    // TODO print loop exit masks
  }
};

void createSingleJumpMasksFrom(BasicBlock *BB, MaskTable &MT, LoopInfo &LI) {
  auto *BlockMask = MT.blockMask(BB);
  auto *Terminator = BB->getTerminator();
  if (isa<UnreachableInst>(Terminator) || isa<ReturnInst>(Terminator)) {
    // No successors => nothing to do.
    return;
  }
  auto *Br = cast<BranchInst>(Terminator);
  if (Br->isConditional()) {
    IRBuilder<> Builder(Br);
    auto *Cond0 = Br->getCondition();
    auto *Cond1 = Builder.CreateNot(Cond0, "not." + Cond0->getName());
    MT.addSingleJump(BB, Br->getSuccessor(0),
                     Builder.CreateAnd(BlockMask, Cond0));
    MT.addSingleJump(BB, Br->getSuccessor(1),
                     Builder.CreateAnd(BlockMask, Cond1));
  } else {
    // Single successor => just re-use the block mask.
    MT.addSingleJump(BB, Br->getSuccessor(0), BlockMask);
  }
}

Value *createBlockMask(BasicBlock *BB, const MaskTable &MT, LoopInfo &LI) {
  IRBuilder<> Builder(BB, BB->getFirstInsertionPt());
  Value *Mask = nullptr;
  if (LI.isLoopHeader(BB)) {
    auto Phi = Builder.CreatePHI(Builder.getInt1Ty(), 2);
    for (auto Pred : predecessors(BB)) {
      Phi->addIncoming(MT.edgeMask(Pred, BB), Pred);
    }
    Mask = Phi;
  } else {
    for (auto Pred : predecessors(BB)) {
      auto JumpMask = MT.edgeMask(Pred, BB);
      Mask = Mask ? Builder.CreateOr(Mask, JumpMask) : JumpMask;
    }
  }
  if (!Mask) {
    auto F = BB->getParent();
    assert(BB == &F->getEntryBlock() && "unreachable block");
    // The entry block mask is pass in as first argument.
    Mask = &*F->arg_begin();
  }
  Mask->setName(BB->getName() + ".exec");
  assert(Mask->getType() == Builder.getInt1Ty());
  return Mask;
}

void createExitMask(Loop *ExitedLoop, const BasicBlock *From,
                    const BasicBlock *To, MaskTable &MT, LoopInfo &LI) {
  DEBUG(dbgs() << "Creating loop exit mask for exit " << From->getName()
               << " -> " << To->getName() << " out of loop "
               << ExitedLoop->getHeader()->getName() << "\n");
  Value *SingleMask = MT.singleJumpMask(From, To);
  Loop *SourceLoop = LI.getLoopFor(From);
  assert(ExitedLoop->contains(SourceLoop));
  if (SourceLoop == ExitedLoop) {
    MT.addLoopExit(ExitedLoop, From, To, SingleMask);
    return;
  }
  DenseMap<Loop *, PHINode *> Phis;
  for (auto L = SourceLoop; L != ExitedLoop; L = L->getParentLoop()) {
    assert(L != nullptr);
    IRBuilder<> Builder(&*L->getHeader()->getFirstNonPHI());
    Phis[L] =
        Builder.CreatePHI(Builder.getInt1Ty(), 2,
                          ExitedLoop->getHeader()->getName() + ".loopexit.phi");
    DEBUG(dbgs() << "\t" << *Phis[L] << "\n");
  }
  IRBuilder<> UpdateBuilder(SourceLoop->getLoopLatch()->getTerminator());
  assert(LI.getLoopFor(From));
  assert(SingleMask);
  auto Update =
      UpdateBuilder.CreateOr(SingleMask, Phis[SourceLoop],
                             ExitedLoop->getHeader()->getName() + ".loopexit");

  Value *False = UpdateBuilder.getFalse();
  for (auto L = LI.getLoopFor(From); L != ExitedLoop; L = L->getParentLoop()) {
    auto ParentLoop = L->getParentLoop();
    auto BackedgeV = cast<Value>(Update);
    auto PreheaderV = ParentLoop == ExitedLoop ? False : Phis[ParentLoop];
    Phis[L]->addIncoming(BackedgeV, L->getLoopLatch());
    Phis[L]->addIncoming(PreheaderV, L->getLoopPreheader());
  }
  MT.addLoopExit(ExitedLoop, From, To, Update);
}

void createExitMasksForLoop(Loop *L, MaskTable &MT, LoopInfo &LI) {
  SmallVector<Edge, 4> Exits;
  getExitEdges(L, Exits);
  for (Edge E : Exits) {
    createExitMask(L, E.first, E.second, MT, LI);
  }
}

void createEdgeMask(BasicBlock *From, BasicBlock *To, MaskTable &MT,
                    LoopInfo &LI) {
  auto SourceLoop = LI.getLoopFor(From);
  auto TargetLoop = LI.getLoopFor(To);
  if (LI.getLoopDepth(From) <= LI.getLoopDepth(To)) {
    // This edge must not be a loop exit, or go from a pre-header to a
    // header.
    assert(SourceLoop == TargetLoop ||
           (TargetLoop && TargetLoop->getLoopPreheader() == From &&
            TargetLoop->getHeader() == To));
    MT.addEdge(From, To, MT.singleJumpMask(From, To));
    return;
  }
  assert(SourceLoop && "Edge is apparently a loop exit, but there's no loop");
  // The exit mask for this edge w.r.t. the outermost loop being exited is
  // almost what we want, we just need to accumulate it over iterations of
  // that loop (nested loops are already accounted for in the exit mask).
  Loop *ExitedLoop = SourceLoop;
  while (ExitedLoop->getParentLoop() != TargetLoop) {
    ExitedLoop = ExitedLoop->getParentLoop();
  }
  IRBuilder<> PhiBuilder(&*ExitedLoop->getHeader()->getFirstNonPHI());
  auto Phi = PhiBuilder.CreatePHI(PhiBuilder.getInt1Ty(), 2);
  IRBuilder<> UpdateBuilder(SourceLoop->getLoopLatch()->getTerminator());
  auto Update = UpdateBuilder.CreateOr(Phi, MT.exitMask(ExitedLoop, From, To));
  Phi->addIncoming(PhiBuilder.getFalse(), ExitedLoop->getLoopPreheader());
  Phi->addIncoming(Update, ExitedLoop->getLoopLatch());
  MT.addEdge(From, To, Update);
}

void createCombinedLoopExitMask(Loop *L, MaskTable &MT) {
  IRBuilder<> Builder(L->getLoopLatch()->getTerminator());
  Value *Combined = nullptr;
  SmallVector<Loop::Edge, 4> Exits;
  L->getExitEdges(Exits);
  for (Loop::Edge E : Exits) {
    Value *ExitCond = MT.exitMask(L, E.first, E.second);
    Combined = Combined ? Builder.CreateOr(Combined, ExitCond) : ExitCond;
  }
  if (!Combined) {
    Combined = Builder.getFalse();
  }
  if (Combined->getName().empty()) {
    Combined->setName(L->getHeader()->getName() + ".combined.exit");
  }
  MT.addCombinedLoopExit(L, Combined);
}

// Helper functions for visiting all loops in the loop tree
template <typename F, typename... OtherArgs>
void forallNestedLoops(Loop *L, F f, OtherArgs &&... args) {
  f(L, args...);
  for (Loop *ChildLoop : *L) {
    forallNestedLoops(ChildLoop, f, args...);
  }
}

template <typename F, typename... OtherArgs>
void forallNestedLoops(LoopInfo &LI, F f, OtherArgs &&... args) {
  for (Loop *L : LI) {
    forallNestedLoops(L, f, args...);
  }
}

MaskTable createMasks(Function &F, LoopInfo &LI) {
  MaskTable MT(F);

  // There are cyclic dependencies between block and other masks, so we
  // break the cycle by introducing placeholders for the block masks which
  // are later RAUW'd the real block masks.
  Value *False = ConstantInt::getFalse(F.getContext());
  for (auto &BB : F) {
    Instruction *ip = &*BB.getFirstInsertionPt();
    Instruction *Placeholder =
        BinaryOperator::Create(BinaryOperator::Or, False, False,
                               "exec.placeholder." + BB.getName(), ip);
    MT.addBlock(&BB, Placeholder);
  }

  for (auto &BB : F) {
    createSingleJumpMasksFrom(&BB, MT, LI);
  }

  forallNestedLoops(LI, createExitMasksForLoop, MT, LI);

  for (auto &From : F) {
    for (auto To : successors(&From)) {
      createEdgeMask(&From, To, MT, LI);
    }
  }

  forallNestedLoops(LI, createCombinedLoopExitMask, MT);

  for (auto &BB : F) {
    auto Placeholder = cast<Instruction>(MT.blockMask(&BB));
    auto RealBlockMask = createBlockMask(&BB, MT, LI);
    Placeholder->replaceAllUsesWith(RealBlockMask);
    Placeholder->eraseFromParent();
  }

  return MT;
}

bool shouldBeMasked(Instruction *I) {
  // TODO return false for phis, terminators, and possibly other stuff
  return !isSafeToSpeculativelyExecute(I);
}

struct InstVectorizeVisitor : InstVisitor<InstVectorizeVisitor> {
  Function &VectorFunc;
  LLVMContext &Context;
  const LoopInfo &LI;
  const MaskTable &MT;
  const DenseMap<Function *, Function *> &VectorizedFunctions;

  DenseMap<Instruction *, Value *> Scalar2Vector;
  // Map each scalar BB to the (first) vectorized BB.
  BBMap<BasicBlock *> BlockMap;
  // Map each vectorized BB to the (last) vectorized BB resulting from the
  // original scalar block. Said "last block" contains the vectorized equivalent
  // of the scalar terminator, needed for vectorizing phis.
  // TODO this SUCKS and DOESN'T EVEN WORK
  BBMap<BasicBlock *> SplitBlockEnd;

  BasicBlock *CurrentBB = nullptr;

  SmallVector<Value *, 8> Arguments;

  enum class MaskMode {
    Ignore,
    Masked,
    Unmasked,
  };

  InstVectorizeVisitor(
      Function &VectorFunc, Function &ScalarFunc, const LoopInfo &LI,
      const MaskTable &MT,
      const DenseMap<Function *, Function *> &VectorizedFunctions)
      : VectorFunc(VectorFunc), Context(VectorFunc.getContext()), LI(LI),
        MT(MT), VectorizedFunctions(VectorizedFunctions) {
    for (auto &Arg : VectorFunc.args()) {
      Arguments.push_back(&Arg);
    }
    for (auto &BB : ScalarFunc) {
      auto VecBB = BasicBlock::Create(Context, BB.getName(), &VectorFunc);
      BlockMap[&BB] = VecBB;
    }
  }

  IRBuilder<> getBuilder() { return IRBuilder<>(CurrentBB); }

  void visitBasicBlock(BasicBlock &BB) { CurrentBB = BlockMap[&BB]; }

  void visitInstruction(Instruction &I) {
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "LowerSPMD cannot vectorize instruction: ";
    I.print(Msg);
    report_fatal_error(Msg.str());
  }

  void visitReturnInst(ReturnInst &Ret) {
    // Previous steps ensure there's only one return, at the end of the
    // function.
    if (auto RetVal = Ret.getReturnValue()) {
      auto VecReturn = getBuilder().CreateRet(getVectorized(RetVal));
      record(&Ret, VecReturn, MaskMode::Ignore);
    } else {
      getBuilder().CreateRetVoid();
    }
  }

  void visitUnreachable(UnreachableInst &Unreachable) {
    // Normally unreachables are removed in favor of a branch to the next block
    // or the return. However, if there is no return, an unreachable might
    // survive linearization. In that case function can't return normally, so we
    // should just keep it... except that the function might be called with an
    // all-zero mask, in which case none of the lanes would execute the
    // unreachable. So to be safe, we just return undef.
    auto RetTy = VectorFunc.getReturnType();
    if (RetTy->isVoidTy()) {
      getBuilder().CreateRetVoid();
    } else {
      getBuilder().CreateRet(UndefValue::get(RetTy));
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
    CondBB->moveAfter(OldBB);
    auto NextBB = BasicBlock::Create(Context, CurrentBB->getName() + ".cont",
                                     &VectorFunc);
    NextBB->moveAfter(CondBB);
    SplitBlockEnd[OldBB] = NextBB; // BUG: this creates a chain through the
                                   // split blocks, not a direct link from the
                                   // start to the last one

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
                               ConstantInt::get(MaskIntTy, 0));

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
    if (auto Mask = tryGetMask(&Op)) {
      auto Builder = getBuilder();
      assert(Op.getOpcode() == Instruction::UDiv ||
             Op.getOpcode() == Instruction::URem ||
             Op.getOpcode() == Instruction::SDiv ||
             Op.getOpcode() == Instruction::SRem);
      auto Bits = Op.getType()->getIntegerBitWidth();
      // In lanes where the mask is 0, executing the operation as-is may be UB
      // (e.g., because the divisor is zero). So we replace those lanes with a
      // value that makes the operation defined.
      auto SafeValue = Builder.getIntN(Bits, 1);
      auto SafeRHS = Builder.CreateSelect(
          Mask, RHS, Builder.CreateVectorSplat(SIMD_WIDTH, SafeValue));
      auto VecOp = BinaryOperator::Create(Op.getOpcode(), LHS, SafeRHS);
      record(&Op, VecOp, MaskMode::Masked);
    } else {
      auto VecOp = BinaryOperator::Create(Op.getOpcode(), LHS, RHS);
      record(&Op, VecOp, MaskMode::Unmasked);
    }
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
    // TODO always mask loads even if they are safe to speculate, to conserve
    // memory bandwidth
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
    // Incoming values from the backedge aren't vectorized yet.
    // So we delay handling the incoming values  until we encounter the
    // terminator of the loop latch.
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
    if (shouldBeMasked(I)) {
      auto ScalarMask = MT.blockMask(I->getParent());
      return getVectorized(ScalarMask);
    }
    return nullptr;
  }

  Value *requireMask(Instruction *I) {
    if (auto Mask = tryGetMask(I)) {
      return Mask;
    } else {
      std::string msg;
      raw_string_ostream Msg(msg);
      Msg << "LowerSPMD: requested mask even though instruction should be "
             "unmasked: ";
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

    switch (Mode) {
    case MaskMode::Unmasked:
      if (shouldBeMasked(Scalar)) {
        std::string Msg;
        raw_string_ostream OS(Msg);
        OS << "Instruction should have been masked but wasn't: " << *Scalar
           << "\n";
        report_fatal_error(Msg.c_str());
      }
      break;
    case MaskMode::Masked:
      if (!shouldBeMasked(Scalar)) {
        std::string Msg;
        raw_string_ostream OS(Msg);
        OS << "Instruction was masked but shouldn't be: " << *Scalar << "\n";
        report_fatal_error(Msg.c_str());
      }
      break;
    case MaskMode::Ignore:
      /* Nothing */
      break;
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
        // TODO get rid of this hack via a better data structure
        // (or maybe a scheme that avoids the data structure entirely)
        while (auto ContinuedBlock = SplitBlockEnd[IncomingBB]) {
          IncomingBB = ContinuedBlock;
        }
        VecPhi->addIncoming(VecIncoming, IncomingBB);
      }
    }
  }
};

Loop *largestLoopIfReducible(const std::vector<BasicBlock *> &SCC,
                             LoopInfo &LI) {
  assert(!SCC.empty());
  // The SCC is reducible iff there is a Loop containing all blocks.
  Loop *L = LI.getLoopFor(SCC[0]);
  while (L && L->getParentLoop()) {
    L = L->getParentLoop();
  }
  if (L && std::all_of(SCC.begin(), SCC.end(),
                       [=](BasicBlock *BB) { return L->contains(BB); })) {
    return L;
  } else {
    return nullptr;
  }
}

struct ToposortState {
  LoopInfo &LI;
  std::deque<BasicBlock *> Output;
  BBMap<std::deque<BasicBlock *>> SubLoopOrders;
  SmallPtrSet<BasicBlock *, 16> Seen;

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
  for (auto Succ : successors(BB)) {
    // We never need to consider blocks that are outside the loop, though,
    // and we better not consider any blocks twice.
    if (!State.Seen.count(Succ) && L->contains(Succ)) {
      loopToposortRec(Succ, L, State);
    }
  }

  if (State.LI.getLoopDepth(BB) == L->getLoopDepth()) {
    // If this is a pre-header, now is the time to insert the sub-loop's
    // blocks into the topological order.
    if (State.SubLoopOrders.count(BB)) {
      const auto &SubLoopOrder = State.SubLoopOrders[BB];
      State.Output.insert(State.Output.begin(), SubLoopOrder.begin(),
                          SubLoopOrder.end());
    }
    State.Output.push_front(BB);
  }
}

std::deque<BasicBlock *> loopToposort(Loop *L, LoopInfo &LI) {
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
  std::vector<std::deque<BasicBlock *>> SCCsLinearized;
  for (auto I = scc_begin(F); !I.isAtEnd(); ++I) {
    const std::vector<BasicBlock *> &SCC = *I;
    if (I.hasLoop()) {
      Loop *L = largestLoopIfReducible(SCC, LI);
      if (!L) {
        return None;
      }
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

struct FunctionVectorizer {
  LLVMContext &Context;
  LoopInfo LI;
  Function *LinearFunc;
  Function *VectorFunc;
  const DenseMap<Function *, Function *> &VectorizedFunctions;
  SmallVector<Instruction *, 8> LoopLiveValues;

  FunctionVectorizer(
      Function &F, Function &VF,
      const DenseMap<Function *, Function *> &VectorizedFunctions)
      : Context(F.getContext()), VectorFunc(&VF),
        VectorizedFunctions(VectorizedFunctions) {
    // We need to clone the source function already before mask computation to
    // insert the i1 argument (the mask for the entry block). We then linearize
    // that copy in-place. Vectorization, however, creates a new function (it
    // can't easily work in-place because the types of all instructions change).
    // TODO find a better name than "LinearFunc"
    ValueToValueMapTy ArgMapping;
    SmallVector<ReturnInst *, 1> Returns;
    LinearFunc = prepareScalarFunction(F, ArgMapping);
    CloneFunctionInto(LinearFunc, &F, ArgMapping,
                      /* ModuleLevelChanges: */ false, Returns);

    if (Returns.size() > 1) {
      report_fatal_error("LowerSPMD: cannot vectorize function with multiple "
                         "returns, run mergereturn");
    }

    DominatorTree DomTree(*LinearFunc);
    LI = LoopInfo(DomTree);

    // We're only interested in values used *outside* the loop
    // Uses in later iterations of the same loop (i.e., uses in phis in the loop
    // header) don't need result vectors.
    for (auto &BB : *LinearFunc) {
      Loop *L = LI.getLoopFor(&BB);
      if (!L) {
        continue;
      }
      for (auto &I : BB) {
        bool usedOutsideLoop =
            std::any_of(I.user_begin(), I.user_end(), [=](User *U) {
              return !L->contains(cast<Instruction>(U));
            });
        if (usedOutsideLoop) {
          LoopLiveValues.push_back(&I);
        }
      }
    }
  }

  // Create an empty function with the same prototype as F, except that an
  // additional i1 argument (the mask for the entry block) is inserted
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
    ++NewArgsIter; // skip the mask we added
    for (auto &OldArg : F.args()) {
      assert(NewArgsIter != NewF->arg_end());
      ArgMapping[&OldArg] = &*NewArgsIter;
      ++NewArgsIter;
    }
    assert(NewArgsIter == NewF->arg_end());
    return NewF;
  }

  std::vector<BasicBlock *> linearizeCFG(MaskTable &MT) {
    std::vector<BasicBlock *> LinearOrder;
    if (auto LinearOrderOpt = findLinearOrder(LinearFunc, LI)) {
      LinearOrder = std::move(LinearOrderOpt.getValue());
    } else {
      report_fatal_error(
          "LowerSPMD run on function with irreducible control flow");
    }

    DEBUG(dbgs() << "Linear order: ");
    for (BasicBlock *BB : LinearOrder) {
      DEBUG(dbgs() << BB->getName() << " ");
    }
    DEBUG(dbgs() << "\n");

    for (size_t i = 0; i < LinearOrder.size(); ++i) {
      BasicBlock *BB = LinearOrder[i];
      DEBUG(dbgs() << "Linearizing " << BB->getName() << "\n");

      if (!LI.isLoopHeader(BB)) {
        PhisToSelect(BB, MT);
      }

      if (i + 1 < LinearOrder.size()) {
        if (auto Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
          // Because of blocks ending with unreachable, the return instruction
          // may not come last in the topological ordering. So we overwrite the
          // last terminator with a return.
          TerminatorInst *LastTerminator = LinearOrder.back()->getTerminator();
          assert(isa<UnreachableInst>(LastTerminator));
          ReplaceInstWithInst(
              LastTerminator,
              ReturnInst::Create(Context, Ret->getReturnValue()));
        }

        BasicBlock *NextBB = LinearOrder[i + 1];
        linearizeTerminator(BB, NextBB, MT);
        NextBB->moveAfter(BB);
      }
    }
    return LinearOrder;
  }

  // Turns phis into selects and records the masks for instructions
  // that need to be masked.
  void PhisToSelect(BasicBlock *BB, MaskTable &MT) {
    // Need manual iterator fiddling because we remove the *current*
    // instruction, so we can't advance the iterator at the *end* of the loop
    // iteration as usual
    for (auto Iter = BB->begin(); Iter != BB->end();) {
      Instruction *I = &*Iter;
      ++Iter;
      if (auto *Phi = dyn_cast<PHINode>(I)) {
        PhiToSelect(Phi, MT);
      }
    }
  }

  void linearizeTerminator(BasicBlock *CurrentBB, BasicBlock *NextBB,
                           MaskTable &MT) {
    Loop *L = LI.getLoopFor(CurrentBB);
    if (L && L->getLoopLatch() == CurrentBB) {
      DEBUG(dbgs() << "linearizeTerminator: " << CurrentBB->getName() << " -> "
                   << NextBB->getName() << " loop latch of " << *L);
      BasicBlock *Header = L->getHeader();
      assert(!L->contains(NextBB));
      assert(Header);
      auto Br = cast<BranchInst>(CurrentBB->getTerminator());
      if (Br->isConditional()) {
        assert(Br->getSuccessor(0) == Header || Br->getSuccessor(1) == Header);
        assert(!L->contains(Br->getSuccessor(0)) ||
               !L->contains(Br->getSuccessor(1)));
        // Canonicalize the branch so that the backedge is taken on `true` and
        // the  loop is exited on `false`.
        Br->setSuccessor(0, Header);
        Br->setSuccessor(1, NextBB);
        Br->setCondition(MT.edgeMask(CurrentBB, Header));
      } else {
        assert(Br->getSuccessor(0) == Header);
        ReplaceInstWithInst(
            CurrentBB->getTerminator(),
            BranchInst::Create(Header, NextBB, MT.edgeMask(CurrentBB, Header)));
      }
      // We need to *make* it conditional, because there's a "next" BB in the
      // linear order and thus there are loop exits that might be taken.
    } else {
      ReplaceInstWithInst(CurrentBB->getTerminator(),
                          BranchInst::Create(NextBB));
    }
  }

  void PhiToSelect(PHINode *Phi, MaskTable &MT) {
    auto Result = Phi->getIncomingValue(0);
    auto BB = Phi->getParent();
    IRBuilder<> Builder(Phi);
    for (unsigned i = 1; i < Phi->getNumIncomingValues(); ++i) {
      auto Mask = MT.edgeMask(Phi->getIncomingBlock(i), BB);
      Result = Builder.CreateSelect(Mask, Phi->getIncomingValue(i), Result,
                                    Phi->getName());
    }
    Phi->replaceAllUsesWith(Result);
    Phi->eraseFromParent();
  }

  Value *createLoopResultInstructions(Instruction &I, const MaskTable &MT) {
    Loop *DefLoop = LI.getLoopFor(I.getParent());
    DenseMap<Loop *, PHINode *> Phis;
    // First, insert phis in all loop levels, but don't fill them yet.
    for (Loop *L = DefLoop; L; L = L->getParentLoop()) {
      IRBuilder<> PhiBuilder(&*L->getHeader()->getFirstInsertionPt());
      Phis[L] = PhiBuilder.CreatePHI(I.getType(), 2);
    }

    // Then, insert the actual update operation in the latch
    auto CombExitCond = MT.combinedLoopExitMask(DefLoop);
    IRBuilder<> UpdateBuilder(DefLoop->getLoopLatch()->getTerminator());
    auto Update = UpdateBuilder.CreateSelect(CombExitCond, &I, Phis[DefLoop]);
    Update->setName("loopres." + I.getName());
    DEBUG(dbgs() << "Created loop result update: " << *Update << "\n");

    // Now fill in the phis.
    for (Loop *L = DefLoop; L; L = L->getParentLoop()) {
      Loop *ParentLoop = L->getParentLoop();
      auto ParentValue = ParentLoop ? static_cast<Value *>(Phis[ParentLoop])
                                    : UndefValue::get(I.getType());
      Phis[L]->addIncoming(ParentValue, L->getLoopPreheader());
      Phis[L]->addIncoming(Update, L->getLoopLatch());
      DEBUG(dbgs() << "Created loop result phi: " << *Phis[L] << "\n");
    }
    // Finally, all uses outside the loop should refer to the update:
    return Update;
  }

  void insertLoopResult(Instruction &I, const MaskTable &MT) {
    Loop *L = LI.getLoopFor(I.getParent());
    SmallVector<Use *, 8> UsesOutsideLoop;
    for (Use &U : I.uses()) {
      auto UserInst = dyn_cast<Instruction>(U.getUser());
      if (!UserInst || L->contains(UserInst)) {
        continue;
      }
      DEBUG(dbgs() << "Will rewrite use of " << I.getName()
                   << " outside loop: " << *UserInst << "\n");
      UsesOutsideLoop.push_back(&U);
    }
    assert(UsesOutsideLoop.size() > 0 &&
           "Inserted loop result for instruction that does not need one");
    auto LoopResult = createLoopResultInstructions(I, MT);
    for (Use *U : UsesOutsideLoop) {
      U->set(LoopResult);
    }
  }

  Function *run() {
    MaskTable MT = createMasks(*LinearFunc, LI);

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(dbgs() << "Vectorizing " << LinearFunc->getName() << "\n");
    DEBUG(MT.print(dbgs()));
    DEBUG(LI.print(dbgs()));
    DEBUG(dbgs() << "\n");

    DEBUG(dbgs() << "===================================================\n");
    for (Instruction *I : LoopLiveValues) {
      insertLoopResult(*I, MT);
    }

    DEBUG(dbgs() << "===================================================\n");
    DEBUG(LinearFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    auto LinearOrder = linearizeCFG(MT);
    DEBUG(LinearFunc->print(dbgs()));

    DEBUG(dbgs() << "===================================================\n");
    InstVectorizeVisitor IVV(*VectorFunc, *LinearFunc, LI, MT,
                             VectorizedFunctions);
    for (BasicBlock *BB : LinearOrder) {
      IVV.visit(BB);
    }
    DEBUG(dbgs() << "===================================================\n");
    DEBUG(VectorFunc->print(dbgs()));

    LinearFunc->eraseFromParent();
    return VectorFunc;
  }
};

Function *unwrapFunctionBitcast(Value *Bitcast) {
  return cast<Function>(cast<ConstantExpr>(Bitcast)->getOperand(0));
}

struct VectorizationObstacle {
  Value *Location;
  StringRef Description;
};

// This visitor determines whether an instruction involves a vector type (in a
// way that will cause a problem for this pass)
struct PreconditionCheckerVisitor : InstVisitor<PreconditionCheckerVisitor> {
  std::vector<VectorizationObstacle> Obstacles;
  SmallPtrSet<Type *, 16> SeenTypes;
  bool SeenReturn = false;

  void checkType(Value *V, Type *Ty,
                 const char *Msg = "unsupported value type") {
    if (SeenTypes.count(Ty)) {
      return;
    }
    bool TypeOK = Ty->isIntegerTy() || Ty->isFloatTy() || Ty->isPointerTy() ||
                  Ty->isVoidTy();
    if (auto IntTy = dyn_cast<IntegerType>(Ty)) {
      // TODO apparently i64 vectors trigger a backend error?
      TypeOK &= IntTy->getBitWidth() <= 32;
    }
    if (!TypeOK) {
      SeenTypes.insert(Ty);
      Obstacles.push_back({V, Msg});
    }
  }

  void checkType(Value *V) { checkType(V, V->getType()); }

  void visitFunction(Function &F) {
    checkType(nullptr, F.getReturnType(), "unsupported return type");
    for (auto &Arg : F.args()) {
      checkType(&Arg, Arg.getType(), "unsupported argument type");
    }
    SeenReturn = false;
  }

  void visitInstruction(Instruction &I) {
    Obstacles.push_back({&I, "unsupported instruction kind"});
  }

  void visitReturnInst(ReturnInst &Ret) {
    // Don't need to check type, already checked return type of function
    if (SeenReturn) {
      Obstacles.push_back({&Ret, "multiple returns"});
    }
    SeenReturn = true;
  }

  void visitUnreachableInst(UnreachableInst &) {
    // Nothing to do
  }

  void visitAllocaInst(AllocaInst &Alloca) {
    if (Alloca.isArrayAllocation()) {
      Obstacles.push_back({&Alloca, "array alloca"});
    }
    if (Alloca.isUsedWithInAlloca()) {
      Obstacles.push_back({&Alloca, "inalloca"});
    }
    if (!Alloca.isStaticAlloca()) {
      Obstacles.push_back({&Alloca, "dynamic alloca"});
    }
  }

  void visitCallInst(CallInst &Call) {
    checkType(&Call);
    std::for_each(Call.arg_begin(), Call.arg_end(),
                  [&](Value *V) { checkType(V); });
    if (!Call.getCalledFunction()) {
      Obstacles.push_back({&Call, "unknown callee"});
    }
    if (!shouldBeMasked(&Call)) {
      Obstacles.push_back({&Call, "unmasked calls not yet implemented"});
    }
  }

  void visitIntrinsicInst(IntrinsicInst &Intr) {
    switch (Intr.getIntrinsicID()) {
    default:
      visitCallInst(Intr);
      break;
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // These get dropped and their i64 argument causes issues
      // FIXME this is not ideal, we want and should be able to keep them
      break;
    }
  }

  void visitDbgInfoIntrinsic(DbgInfoIntrinsic &) {
    // These get dropped
  }

  void visitCmpInst(CmpInst &Cmp) {
    checkType(&Cmp);
    checkType(Cmp.getOperand(0));
    checkType(Cmp.getOperand(1));
  }

  void visitBinaryOperator(BinaryOperator &Op) {
    checkType(&Op);
    checkType(Op.getOperand(0));
    checkType(Op.getOperand(1));
    auto Opc = Op.getOpcode();
    if (shouldBeMasked(&Op) && Opc != Instruction::UDiv &&
        Opc != Instruction::URem && Opc != Instruction::SDiv &&
        Opc != Instruction::SRem) {
      Obstacles.push_back({&Op, "masked binops not yet implemented"});
    }
  }

  void visitCastInst(CastInst &Cast) {
    checkType(&Cast);
    checkType(Cast.getOperand(0));
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    // Nothing to do, operands and return values are all integers, pointers,
    // and
    // vectors thereof.
  }

  void visitLoadInst(LoadInst &Load) { checkType(&Load); }

  void visitStoreInst(StoreInst &Store) { checkType(Store.getValueOperand()); }

  void visitSelectInst(SelectInst &Sel) {
    checkType(&Sel);
    // Operands have the same type, condition must be i1 or i1 vector
  }

  void visitPHINode(PHINode &Phi) { checkType(&Phi); }

  void visitBranchInst(BranchInst &Br) {
    // Nothing to do.
  }
};

bool shouldVectorize(Function &F) {
  PreconditionCheckerVisitor PCV;
  PCV.visit(F);
  // FIXME: for performance, only accumulate obstacles in debug mode
  // in release mode we should short circuit once we find any obstacle
  auto Obstacles = std::move(PCV.Obstacles);

  // Check reducibility
  DominatorTree DomTree(F);
  LoopInfo LI(DomTree);
  for (auto I = scc_begin(&F); !I.isAtEnd(); ++I) {
    const std::vector<BasicBlock *> &SCC = *I;
    if (I.hasLoop()) {
      Loop *L = largestLoopIfReducible(SCC, LI);
      if (!L) {
        Obstacles.push_back({nullptr, "irreducible control flow"});
        break;
      }
    }
  }

  // Check for vararg
  if (F.getFunctionType()->isVarArg()) {
    Obstacles.push_back({nullptr, "vararg function"});
  }

  // Check if loop-simplify was run
  forallNestedLoops(LI, [&](Loop *L) {
    if (!L->isLoopSimplifyForm()) {
      Obstacles.push_back({nullptr, "loop not in canonical form"});
    }
  });

  // TODO check other obstacles
  if (Obstacles.empty()) {
    return true;
  }
  DEBUG(dbgs() << "Will not vectorize function " << F.getName()
               << " because:\n");
  for (auto &O : Obstacles) {
    DEBUG(dbgs() << "\t" << O.Description);
    if (O.Location) {
      DEBUG(dbgs() << ": " << *O.Location);
    }
    DEBUG(dbgs() << "\n");
  }
  return false;
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
  auto VectorFunc = Function::Create(VectorFT, ScalarFunc.getLinkage(),
                                     ScalarFunc.getName() + ".vector",
                                     ScalarFunc.getParent());
  if (ScalarFunc.hasFnAttribute(Attribute::NoInline)) {
    VectorFunc->addFnAttr(Attribute::NoInline);
  }
  VectorFunc->setLinkage(ScalarFunc.getLinkage());
  return VectorFunc;
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

    std::vector<Function *> ScalarFuncs;
    for (auto &F : M) {
      if (F.empty()) {
        continue;
      }
      if (shouldVectorize(F)) {
        ScalarFuncs.push_back(&F);
        NumFunctionsSPMDVectorized++;
      } else {
        NumFunctionsNotSPMDVectorizable++;
      }
    }

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

    for (auto Call : RootCalls) {
      auto ScalarF = unwrapFunctionBitcast(Call->getArgOperand(0));
      auto Arg = Call->getArgOperand(1);
      DEBUG(dbgs() << "Replacing spmd.call to " << ScalarF->getName() << "\n");
      IRBuilder<> Builder(Call);
      auto VectorF = Vectorized.lookup(ScalarF);
      if (!VectorF) {
        std::string Msg;
        {
          raw_string_ostream OS(Msg);
          OS << "SPMD call to function that could not be vectorized: "
             << ScalarF->getName() << "\n";
        }
        report_fatal_error(Msg.c_str());
      }

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
