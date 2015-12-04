{-# LANGUAGE TemplateHaskell #-}
module HNN.Optimize.RMSProp (
   RMSPropT
 , runRMSProp
 , rmsprop
 ) where
import Control.Monad.RWS
import Control.Monad.Trans
import Data.VectorSpace
import Control.Lens

import HNN.Tensor

data RMSPropState w = RMSPropState {
  _msqr :: w
  }
makeLenses ''RMSPropState

data RMSPropReader s = RMSPropReader {
    _msqrFactor :: s
  , _learningRate :: s
  , _minNorm :: s
  }
makeLenses ''RMSPropReader

type RMSPropT w s = RWST (RMSPropReader s) () (RMSPropState w)

runRMSProp :: (VectorSpace w, Monad m)
           => Scalar w -> Scalar w -> Scalar w -> RMSPropT w (Scalar w) m a -> m a
runRMSProp learningRate msqrFactor maxRate action = do
  (x,_,_) <- runRWST
             action
             (RMSPropReader msqrFactor learningRate (learningRate / maxRate))
             (RMSPropState zeroV)
  return x

rmsprop :: (Floating w, VectorSpace w, Monad m)
        => Sacalar w -> w -> w -> RMSPropT w (Scalar w) m w
rmsprop cost grad w_t = do
  lr <- view learningRate
  msqr_t <- use msqr
  msqrFact <- view msqrFactor
  minNorm' <- view minNorm
  let new_msqr = msqrFact *^ msqr_t + (1 - msqrFact) *^ (grad <.> grad)
  msqr .= new_msqr
  return $ w_t - lr *^ grad / elementwiseMax (sqrt new_msqr) minNorm'
