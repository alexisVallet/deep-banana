{-# LANGUAGE RankNTypes #-}
module DeepBanana.Optimize (
    sgd
  , Update(..)
  , vanilla
  , momentum
  ) where
import ClassyPrelude
import Pipes.Lift

import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception

sgd :: (Monad m, MonadTrans t, Monad (t (Pipe a (Scalar w, w) m)), VectorSpace w)
    => Update t w -- update function
    -> (w -> a -> m (Scalar w, w)) -- cost function gradient
    -> w -- initial weights
    -> Pipe a (Scalar w, w) m () -- streams out cost weights for each iteration
sgd (Update update run) cost_grad w_0 = run $ evalStateT action w_0 
  where action = forever $ do
          w_t <- get
          batch <- lift $ lift $ await
          (cost, grad) <- lift $ lift $ lift $ cost_grad w_t batch
          w_tp1 <- lift $ update cost grad w_t
          lift $ lift $ yield (cost, w_tp1)
          put w_tp1

data Update t w = Update {
    update :: forall m . (Monad m) => Scalar w -> w -> w -> t m w
  , run :: forall a m . (Monad m) => t m a -> m a
  }

vanilla :: (VectorSpace w) => Scalar w -> Update IdentityT w
vanilla lr = Update {
    update = \cost grad w_t -> do
       return $ w_t ^-^ lr *^ grad
  , run = runIdentityT
  }


momentum :: (VectorSpace w) => Scalar w -> Scalar w -> Update (StateT (Maybe w)) w
momentum lr mfactor = Update {
    update = \cost grad w_t -> do
       mprev <- get
       let new_update = case mprev of
             Nothing -> (negateV $ lr *^ grad)
             Just prev -> (mfactor *^ prev ^-^ lr *^ grad)
       put $ Just new_update
       return $ w_t ^+^ new_update
  , run = flip evalStateT Nothing
  }
