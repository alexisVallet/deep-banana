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
import DeepBanana.Weights
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception

sgd :: (Monad m, MonadTrans t, Monad (t (Pipe a (FixScalar w, w) m)), FixShape w)
    => Update t (Pipe a (FixScalar w, w) m) w -- update function
    -> (w -> a -> m (FixScalar w, w)) -- cost function gradient
    -> w -- initial weights
    -> Pipe a (FixScalar w, w) m () -- streams out cost weights for each iteration
sgd (Update update run) cost_grad w_0 = run $ evalStateT action w_0 
  where action = forever $ do
          w_t <- get
          batch <- lift $ lift $ await
          (cost, grad) <- lift $ lift $ lift $ cost_grad w_t batch
          w_tp1 <- lift $ update cost grad w_t
          lift $ lift $ yield (cost, w_tp1)
          put w_tp1

data Update t m w = Update {
    update :: FixScalar w -> w -> w -> t m w
  , run :: forall a . t m a -> m a
  }

vanilla :: (FixShape w, MonadError e m, Variant e IncompatibleShape)
        => FixScalar w -> Update IdentityT m w
vanilla lr = Update {
    update = \_ grad w_t ->
     lift $ liftVec (\grad' w_t' -> w_t' ^-^ lr *^ grad') grad w_t
  , run = runIdentityT
  }

momentum :: (FixShape w, MonadError e m, Variant e IncompatibleShape)
         => FixScalar w -> FixScalar w -> Update (StateT (Maybe w)) m w
momentum lr mfactor = Update {
    update = \cost grad w_t -> do
       mprev <- get
       new_update <- case mprev of
         Nothing -> lift $ liftVec (\_ grad' -> (negateV $ lr *^ grad')) grad grad
         Just prev -> lift $ liftVec (\prev' grad' -> mfactor *^ prev' ^-^ lr *^ grad')
                      prev grad
       put $ Just new_update
       liftVec (^+^) w_t new_update
  , run = flip evalStateT Nothing
  }

class (Monad m, FixShape t) => HasElemwiseMax m t where
  elemwiseMax :: t -> FixScalar t -> m t

instance (MonadError t m, Variant t OutOfMemory, Variant t IncompatibleShape,
          Shape s, Device d, TensorScalar a)
         => HasElemwiseMax m (Tensor d s a) where
  elemwiseMax t x = do
    case toAnyFixed $ shape t of
     AnyFixed fshp -> do
       ft <- shapeConvert fshp t
       ones <- ones fshp
       res <- elementwiseMax ft $ x *^ ones
       shapeConvert (shape t) res

instance (MonadError t m, Variant t IncompatibleShape, Floating s)
         => HasElemwiseMax m (Weights s '[]) where
  elemwiseMax _ _ = return $ W Z

instance forall m s l e
         . (HasElemwiseMax m (Weights s l), HasElemwiseMax m e,
            FixScalar e ~ s, FixScalar (Weights s l) ~ s)
         => HasElemwiseMax m (Weights s (e ': l)) where
  elemwiseMax (W ((:.) x xs)) y = do
    head <- elemwiseMax x y
    tail <- elemwiseMax (W xs) y :: m (Weights s l)
    return $ W $ (:.) head $ unWeights tail

rmsprop :: (MonadError e m, Variant e OutOfMemory, Variant e IncompatibleShape,
            HasElemwiseMax m w, FixShape w)
        => FixScalar w -> FixScalar w -> FixScalar w
        -> Update (StateT (Maybe w)) m w
rmsprop lr msqrFact maxRate = Update {
    update = \cost grad w_t -> do
       mMsqr <- get
       newMsqr <- case mMsqr of
         Nothing -> lift $ liftVec (*) grad grad
         Just msqr -> lift $ liftVec
                      (\msqr' grad' ->
                        msqrFact *^ msqr' + (1 - msqrFact) *^ (grad' * grad'))
                      msqr grad
       let minNorm = lr / maxRate
       put $ Just newMsqr
       mnorm <- liftVec (\_ x -> sqrt x) newMsqr newMsqr
       clipNorm <- lift $ elemwiseMax mnorm minNorm
       
       liftVec (/) grad  clipNorm
       >>= liftVec (\w_t' normGrad -> w_t' ^-^ lr *^ normGrad) w_t
  , run = flip evalStateT Nothing
  }
