{-# LANGUAGE TypeFamilies, RankNTypes #-}
module DeepBanana.Layer.CUDA (
    module DeepBanana.Layer.CUDA.CuDNN
  , module DeepBanana.Layer.CUDA.CuDNN.Exception
  , module DeepBanana.Layer.CUDA.Cublas
  , module DeepBanana.Layer.CUDA.CuRAND
  , module DeepBanana.Layer.CUDA.Exception
  , module DeepBanana.Layer.CUDA.Monad
  , module DeepBanana.Layer.CUDA.Numeric
  , lreshape
  , toScalar
  , mlrCost
  ) where

import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.CuDNN
import DeepBanana.Layer.CUDA.CuDNN.Exception
import DeepBanana.Layer.CUDA.Cublas
import DeepBanana.Layer.CUDA.CuRAND
import DeepBanana.Layer.CUDA.Exception
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Layer.CUDA.Numeric
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception

lreshape :: (MonadCuda m, TensorScalar a, Shape (Dim n), Shape (Dim k))
         => Dim k -> Layer m a '[] (Tensor n a) (Tensor k a)
lreshape newshp = combinePasses' fwdmul bwdmul
  where fwdmul x = reshape newshp x
        bwdmul x _ = return $ \upgrad -> reshape' (shape x) upgrad

toScalar :: (MonadCuda m, TensorScalar a, Shape (Dim n))
         => Layer m a '[] (Tensor n a) a
toScalar = noWeights $ fwdBwdToScalar
  where fwdBwdToScalar x = do
          when (size (shape x) /= 1) $ throwVariant $ incompatibleSize $ 
            "Can only convert to scalar tensor with size 1.\nshape: " ++ show (shape x)
          return (unsafeHead $ tensorToList x,
                  \upgrad -> tensorFromList' (shape x) [upgrad])

mlrCost :: (MonadCuda m, TensorScalar a)
        => Dim 2 -> Layer m a '[] (Tensor 2 a, Tensor 2 a) (Tensor 1 a)
mlrCost s@(n:.m:.Z) =
  id' *** (add -< 10E-5 >+> softmax softmax_accurate softmax_mode_instance)
  >+> multiply
  >+> sumRows
  >+> add -< 10E-5
  >+> llog
  >+> lreshape (n:.1:.Z)
  >+> sumCols
  >+> scale -< (-1 / fromIntegral n)
