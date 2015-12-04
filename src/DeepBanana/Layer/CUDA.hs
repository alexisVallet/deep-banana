{-# LANGUAGE TemplateHaskell #-}
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
softmax :: (TensorScalar a, KnownNat n, KnownNat m)
        => Layer CUDA a '[] (Tensor [n,m] a) (Tensor [n,m] a)
softmax =
  lexp
  >+> id &&& (sumRows >+> lreshape >+> replicateAsCols (Proxy :: Proxy m) >+> inv)
  >+> multiply

lreshape :: (TensorScalar a, Shape s1, Shape s2, Size s1 ~ Size s2)
         => Layer CUDA a '[] (Tensor s1 a) (Tensor s2 a)
lreshape = combinePasses' fwdmul bwdmul
  where fwdmul x = return $ reshape x
        bwdmul _ _ = return $ \upgrad -> reshape upgrad

toScalar :: (TensorScalar a, Shape s, Size s ~ 1)
         => Layer CUDA a '[] (Tensor s a) a
toScalar = noWeights $ fwdBwdToScalar
  where fwdBwdToScalar x = do
          return (head $ toList x, \upgrad -> fromList [upgrad])

mlrCost :: forall a n m . (TensorScalar a, KnownNat n, KnownNat m)
        => Layer CUDA a '[] (Tensor [n,m] a, Tensor [n,m] a) a
mlrCost =
  id *** (add -< 10E-5 >+> softmax)
  >+> multiply
  >+> sumRows
  >+> add -< 10E-5
  >+> llog
  >+> sumCols
  >+> scale -< (-1 / fromIntegral (natVal (Proxy :: Proxy n)))
  >+> toScalar
