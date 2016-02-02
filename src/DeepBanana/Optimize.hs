{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Optimize (
    sgd
  , VanillaT
  , vanilla
  , runVanilla
  , MomentumT
  , momentum
  , runMomentum
  , RMSPropT
  , rmsprop
  , runRMSProp
  ) where
import Control.Monad.Morph
import Control.Monad.Trans
import Control.Monad.State
import Control.Monad.RWS
import Control.Monad.Reader
import Data.VectorSpace

import Pipes
import Pipes.Lift

import DeepBanana.Layer
import DeepBanana.Tensor

sgd :: (Monad m, MonadTrans t, Monad (t (Pipe a (Scalar w, w) m)), Monad (t m), MFunctor t, VectorSpace w)
    => (Scalar w -> w -> w -> t m w) -- update function
    -> (w -> a -> m (Scalar w, w)) -- cost function gradient
    -> w -- initial weights
    -> t (Pipe a (Scalar w, w) m) () -- streams out cost weights for each iteration
sgd update cost_grad w_0 =
  distribute $ sgd' update cost_grad w_0

sgd' :: (Monad m, MonadTrans t, Monad (t m), VectorSpace w)
     => (Scalar w -> w -> w -> t m w) -- update function
     -> (w -> a -> m (Scalar w, w)) -- cost function gradient
     -> w -- initial weights
     -> Pipe a (Scalar w, w) (t m) () -- streams out cost weights for each iteration
sgd' update cost_grad w_0 = evalStateT action w_0 
  where action = forever $ do
          w_t <- get
          batch <- lift $ await
          (cost, grad) <- lift $ lift $ lift $ cost_grad w_t batch
          w_tp1 <- lift $ lift $ update cost grad w_t
          lift $ yield (cost, w_tp1)
          put w_tp1

-- "Vanilla" stochastic gradient descent
data VanillaReader s = VanillaReader {
  vLearningRate :: s
  }

type VanillaT s = ReaderT (VanillaReader s)

runVanilla learningRate action =
  runReaderT action (VanillaReader learningRate)

vanilla :: (VectorSpace w, Monad m,
            MonadReader (VanillaReader (Scalar w))
            (ReaderT (VanillaReader (Scalar w)) m))
        => Scalar w -> w -> w -> VanillaT (Scalar w) m w
vanilla cost grad w_t = do
  lr <- asks vLearningRate
  return $ w_t ^-^ lr *^ grad


-- Momentum
data MomentumState w = MomentumState {
  mPreviousUpdate :: w
  }

data MomentumReader s = MomentumReader {
    mMomentumFactor :: s
  , mLearningRate :: s
  }

type MomentumT w s = RWST (MomentumReader s) () (MomentumState w)

runMomentum :: (VectorSpace w, Monad m)
            => Scalar w -> Scalar w -> MomentumT w (Scalar w) m a -> m a
runMomentum learningRate momentumFactor action = do
  (x,_,_) <- runRWST
             action
             (MomentumReader momentumFactor learningRate)
             (MomentumState zeroV)
  return x

momentum :: (VectorSpace w, Monad m,
             MonadReader (MomentumReader (Scalar w))
             (RWST (MomentumReader (Scalar w)) () (MomentumState w) m))
         => Scalar w -> w -> w -> MomentumT w (Scalar w) m w
momentum cost grad w_t = do
  lr <- asks mLearningRate
  mf <- asks mMomentumFactor
  dtm1 <- gets mPreviousUpdate
  let dt = mf *^ dtm1 ^-^ lr *^ grad
  modify (\st -> st {mPreviousUpdate = dt})
  return $ w_t ^+^ dt

data RMSPropState s w = RMSPropState {
  rmsqr :: HLSpace s w
  }

data RMSPropReader s = RMSPropReader {
    rmsqrFactor :: s
  , rlearningRate :: s
  , rminNorm :: s
  }

type RMSPropT s w = RWST (RMSPropReader s) () (RMSPropState s w)

runRMSProp :: (Monad m, AdditiveGroup (HLSpace s w), Fractional s,
               Scalar (HLSpace s w) ~ s)
           => s -> s -> s -> RMSPropT s w m a -> m a
runRMSProp learningRate msqrFactor maxRate action = do
  (x,_,_) <- runRWST
             action
             (RMSPropReader msqrFactor learningRate (learningRate / maxRate))
             (RMSPropState zeroV)
  return x

class HLElemwiseMax s w where
  hlElemwiseMax :: HLSpace s w -> HLSpace s w -> HLSpace s w

instance HLElemwiseMax s '[] where
  hlElemwiseMax _ _ = HLS HNil

instance forall s w n
         . (TensorScalar s, HLElemwiseMax s w, Scalar (HLSpace s w) ~ s, Shape (Dim n))
         => HLElemwiseMax s (Tensor n s ': w) where
  hlElemwiseMax (HLS (HCons t1 w1)) (HLS (HCons t2 w2)) =
    let t = case elementwiseMax t1 t2 of 
              Left err -> error $ "Couldn't compute elementwise max: " ++ err
              Right out -> out in
    HLS $ HCons t (unHLS (hlElemwiseMax (HLS w1) (HLS w2) :: HLSpace s w))

rmsprop :: (VectorSpace (HLSpace s w), Monad m, HLElemwiseMax s w, Real s,
            Scalar (HLSpace s w) ~ s, Floating (HLSpace s w))
        => s -> HLSpace s w -> HLSpace s w -> RMSPropT s w m (HLSpace s w)
rmsprop cost grad w_t = do
  lr <- asks rlearningRate
  msqr_t <- gets rmsqr
  msqrFact <- asks rmsqrFactor
  minNorm' <- asks rminNorm
  let new_msqr = (msqrFact *^ msqr_t) + ((1 - msqrFact) *^ (grad * grad))
  modify (\s -> s {rmsqr = new_msqr})
  return $ w_t - lr *^ (grad / hlElemwiseMax (sqrt new_msqr) (realToFrac minNorm'))
