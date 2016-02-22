{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA (
    module DeepBanana.Layer.CUDA.CuDNN
  , module DeepBanana.Layer.CUDA.CuDNN.Exception
  , module DeepBanana.Layer.CUDA.Cublas
  , module DeepBanana.Layer.CUDA.CuRAND
  , module DeepBanana.Layer.CUDA.Exception
  , module DeepBanana.Layer.CUDA.Numeric
  , CUDAT
  , CUDA
  , CUDAExceptions
  , runCUDAT
  , runCUDATEx
  , runCUDA
  , runCUDAEx
  , softmax
  , lreshape
  , toScalar
  , mlrCost
  ) where

import DeepBanana.Data.Exception
import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.CuDNN
import DeepBanana.Layer.CUDA.CuDNN.Exception
import DeepBanana.Layer.CUDA.Cublas
import DeepBanana.Layer.CUDA.CuRAND
import DeepBanana.Layer.CUDA.Exception
import DeepBanana.Layer.CUDA.Numeric
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception

type CUDAT m = ExceptT CUDAExceptions (CuRANDT m)

type CUDA = CUDAT Identity

type CUDAExceptions = Coproduct '[
                        AllocFailed
                      , BadParam
                      , NotSupported
                      , MemoryAllocation
                      , MappingError
                      , ExecutionFailed
                      , OutOfMemory
                      , IncompatibleShape
                      , IncompatibleSize
                      , EmptyBatch
                      ]

runCUDAT :: (Monad m) => CULLong -> CUDAT m a -> m (Either CUDAExceptions a)
runCUDAT seed action = runCuRANDT seed $ runExceptT action

runCUDATEx :: (MonadThrow m) => CULLong -> CUDAT m a -> m a
runCUDATEx seed action = do
  eres <- runCUDAT seed action
  case eres of
   Left err -> throwM err
   Right res -> return res

runCUDA :: CULLong -> CUDA a -> Either CUDAExceptions a
runCUDA seed action = runIdentity $ runCUDAT seed action

runCUDAEx :: CULLong -> CUDA a -> a
runCUDAEx seed action = unsafeRunExcept $ runCUDA seed action

-- "Naive" CUDA implementation of softmax, as workaround to bug in current
-- CuDNN-based softmax.
softmax :: (Monad m, MonadError t m, Variant t IncompatibleShape,
            Variant t OutOfMemory, Exception t, TensorScalar a)
        => Dim 2 -> Layer m a '[] (Tensor 2 a) (Tensor 2 a)
softmax (n:.m:.Z) =
  lexp
  >+> id' &&& (sumRows >+> replicateAsCols m >+> inv)
  >+> multiply

lreshape :: (Monad m, MonadError t m, Variant t IncompatibleSize,
             TensorScalar a, Shape (Dim n), Shape (Dim k))
         => Dim k -> Layer m a '[] (Tensor n a) (Tensor k a)
lreshape newshp = combinePasses' fwdmul bwdmul
  where fwdmul x = reshape newshp x
        bwdmul x _ = return $ \upgrad -> reshape' (shape x) upgrad

toScalar :: (Monad m, MonadError t m, Variant t IncompatibleSize,
             TensorScalar a, Shape (Dim n))
         => Layer m a '[] (Tensor n a) a
toScalar = noWeights $ fwdBwdToScalar
  where fwdBwdToScalar x = do
          when (size (shape x) /= 1) $ throwVariant $ incompatibleSize $ 
            "Can only convert to scalar tensor with size 1.\nshape: " ++ show (shape x)
          return (unsafeHead $ tensorToList x,
                  \upgrad -> tensorFromList' (shape x) [upgrad])

mlrCost :: (Monad m, MonadError t m, Variant t IncompatibleShape,
            Variant t IncompatibleSize, Variant t OutOfMemory, Exception t,
            TensorScalar a)
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
