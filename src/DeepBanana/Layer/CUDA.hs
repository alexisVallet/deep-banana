{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA (
    module DeepBanana.Layer.CUDA.Monad
  , module DeepBanana.Layer.CUDA.CuDNN
  , module DeepBanana.Layer.CUDA.Cublas
  , module DeepBanana.Layer.CUDA.CuRAND
  , module DeepBanana.Layer.CUDA.Numeric
  , softmax
  , lreshape
  , toScalar
  , mlrCost
  ) where

import Control.Monad.Except
import Data.Proxy
import GHC.TypeLits
import Prelude hiding (id, (.))
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Layer.CUDA.CuDNN
import DeepBanana.Layer.CUDA.Cublas
import DeepBanana.Layer.CUDA.CuRAND
import DeepBanana.Layer.CUDA.Numeric
import DeepBanana.Tensor

-- "Naive" CUDA implementation of softmax, as workaround to bug in current
-- CuDNN-based softmax.
softmax :: (TensorScalar a)
        => Dim 2 -> Layer CUDA a '[] (Tensor 2 a) (Tensor 2 a)
softmax (n:.m:.Z) =
  lexp
  >+> id' &&& (sumRows >+> replicateAsCols m >+> inv)
  >+> multiply

lreshape :: (TensorScalar a, Shape (Dim n), Shape (Dim m))
         => Dim m -> Layer CUDA a '[] (Tensor n a) (Tensor m a)
lreshape newshp = combinePasses' fwdmul bwdmul
  where fwdmul x = reshape newshp x
        bwdmul x _ = return $ \upgrad -> unsafeReshape (shape x) upgrad

toScalar :: (TensorScalar a, Shape (Dim n))
         => Layer CUDA a '[] (Tensor n a) a
toScalar = noWeights $ fwdBwdToScalar
  where fwdBwdToScalar x = do
          when (size (shape x) /= 1) $ throwError $
            "Can only convert to scalar tensor with size 1.\nshape: " ++ show (shape x)
          return (head $ toList x, \upgrad -> fromList (shape x) [upgrad])

mlrCost :: (TensorScalar a)
        => Dim 2 -> Layer CUDA a '[] (Tensor 2 a, Tensor 2 a) (Tensor 1 a)
mlrCost s@(n:.m:.Z) =
  id' *** (add -< 10E-5 >+> softmax s)
  >+> multiply
  >+> sumRows
  >+> add -< 10E-5
  >+> llog
  >+> lreshape (n:.1:.Z)
  >+> sumCols
  >+> scale -< (-1 / fromIntegral n)
