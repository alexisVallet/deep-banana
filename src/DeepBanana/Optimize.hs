{-# LANGUAGE RankNTypes, TypeFamilies, UndecidableInstances #-}
module DeepBanana.Optimize (
    sgd
  , Update(..)
  , vanilla
  , momentum
  , rmsprop
  ) where
import ClassyPrelude
import Pipes.Lift

import DeepBanana.Device
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception

sgd :: (Monad m, MonadTrans t, Monad (t (Pipe a (Scalar w, w) m)), VectorSpace w)
    => Update t (Pipe a (Scalar w, w) m) w -- update function
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

data Update t m w = Update {
    update :: Scalar w -> w -> w -> t m w
  , run :: forall a . t m a -> m a
  }

vanilla :: (VectorSpace w, Monad m) => Scalar w -> Update IdentityT m w
vanilla lr = Update {
    update = \cost grad w_t -> do
       return $ w_t ^-^ lr *^ grad
  , run = runIdentityT
  }


momentum :: (VectorSpace w, Monad m)
         => Scalar w -> Scalar w -> Update (StateT (Maybe w)) m w
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

class (Monad m, VectorSpace t) => HasElemwiseMax m t where
  elemwiseMax :: t -> Scalar t -> m t

instance (MonadError t m, Variant t OutOfMemory, Variant t IncompatibleShape,
          Shape (Dim n), Device d, TensorScalar a)
         => HasElemwiseMax m (Tensor d n a) where
  elemwiseMax t x = do
    ones <- ones (shape t)
    elementwiseMax t $ x *^ ones

instance (Monad m) => HasElemwiseMax m (Weights s '[]) where
  elemwiseMax _ _ = return $ W Z

instance forall m s l e
         . (HasElemwiseMax m (Weights s l), HasElemwiseMax m e,
            Scalar e ~ s, Scalar (Weights s l) ~ s)
         => HasElemwiseMax m (Weights s (e ': l)) where
  elemwiseMax (W ((:.) x xs)) y = do
    head <- elemwiseMax x y
    tail <- elemwiseMax (W xs) y :: m (Weights s l)
    return $ W $ (:.) head $ unWeights tail

rmsprop :: (HasElemwiseMax m w, Floating w, Floating (Scalar w))
        => Scalar w -> Scalar w -> Scalar w -> Update (StateT (Maybe w)) m w
rmsprop lr msqrFact maxRate = Update {
    update = \cost grad w_t -> do
       mMsqr <- get
       let newMsqr = case mMsqr of
             Nothing -> grad * grad
             Just msqr -> msqrFact *^ msqr + (1 - msqrFact) *^ (grad * grad)
           minNorm = lr / maxRate
       put $ Just newMsqr
       clipNorm <- lift $ elemwiseMax (sqrt newMsqr) minNorm
       return $ w_t ^-^ lr *^ (grad / clipNorm)
  , run = flip evalStateT Nothing
  }
