module DeepBanana.Optimize (
    sgd
  , VanillaT
  , vanilla
  , runVanilla
  , MomentumT
  , momentum
  , runMomentum
  ) where
import Control.Monad.Morph
import Control.Monad.Trans
import Control.Monad.State
import Control.Monad.RWS
import Control.Monad.Reader
import Data.VectorSpace

import Pipes
import Pipes.Lift

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
