{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA (
    module DeepBanana.Layer.CUDA.CuDNN
  , module DeepBanana.Layer.CUDA.Cublas
  , module DeepBanana.Layer.CUDA.CuRAND
  , module DeepBanana.Layer.CUDA.Numeric
  , CUDAT
  , CUDA
  , runCUDAT
  , runCUDA
  , softmax
  , lreshape
  , toScalar
  , mlrCost
  ) where

import Control.Monad.Identity
import Control.Monad.Except
import Data.Proxy
import GHC.TypeLits
import Prelude hiding (id, (.))
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.CuDNN
import DeepBanana.Layer.CUDA.Cublas
import DeepBanana.Layer.CUDA.CuRAND
import DeepBanana.Layer.CUDA.Numeric
import DeepBanana.Tensor

type CUDAT m = CuRANDT m

type CUDA = CUDAT Identity

runCUDAT :: (Monad m) => CULLong -> CUDAT m a -> m a
runCUDAT = runCuRANDT

runCUDA :: CULLong -> CUDA a -> a
runCUDA seed action = runIdentity $ runCUDAT seed action

-- "Naive" CUDA implementation of softmax, as workaround to bug in current
-- CuDNN-based softmax.
softmax :: (Monad m, TensorScalar a)
        => Dim 2 -> Layer m a '[] (Tensor 2 a) (Tensor 2 a)
softmax (n:.m:.Z) =
  lexp
  >+> id' &&& (sumRows >+> replicateAsCols m >+> inv)
  >+> multiply

lreshape :: (Monad m, TensorScalar a, Shape (Dim n), Shape (Dim k))
         => Dim k -> Layer m a '[] (Tensor n a) (Tensor k a)
lreshape newshp = combinePasses' fwdmul bwdmul
  where fwdmul x = return $ unsafeReshape newshp x
        bwdmul x _ = return $ \upgrad -> unsafeReshape (shape x) upgrad

toScalar :: (Monad m, TensorScalar a, Shape (Dim n))
         => Layer m a '[] (Tensor n a) a
toScalar = noWeights $ fwdBwdToScalar
  where fwdBwdToScalar x = do
          when (size (shape x) /= 1) $ error $
            "Can only convert to scalar tensor with size 1.\nshape: " ++ show (shape x)
          return (head $ toList x, \upgrad -> fromList (shape x) [upgrad])

mlrCost :: (Monad m, TensorScalar a)
        => Dim 2 -> Layer m a '[] (Tensor 2 a, Tensor 2 a) (Tensor 1 a)
mlrCost s@(n:.m:.Z) =
  id' *** (add -< 10E-5 >+> softmax s)
  >+> multiply
  >+> sumRows
  >+> add -< 10E-5
  >+> llog
  >+> lreshape (n:.1:.Z)
  >+> sumCols
  >+> scale -< (-1 / fromIntegral n)
