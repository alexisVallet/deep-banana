{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA.CuRAND (
    dropout
  , uniform
  , normal
  , logNormal
  ) where

import qualified Foreign.CUDA.CuRAND as CuRAND
import Data.Proxy

import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Tensor
import DeepBanana.Tensor.Mutable (MTensor, IOTensor, emptyTensor, withDevicePtr)
import qualified DeepBanana.Tensor.Mutable as MT

-- dropout
dropout :: (TensorScalar a, Shape s)
        => a
        -> Layer CUDA a '[] (Tensor s a) (Tensor s a)
dropout drop_proba = noWeights fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd inp = do
          gen <- asks generator
          mask <- liftIO $ dropoutMaskIO gen drop_proba
          pure_mask <- unsafeFreeze mask
          return (inp * pure_mask, \upgrad -> upgrad * pure_mask)

-- dropout
-- compute a random mask of ones and zeros
-- to apply elementwise to the input tensor.
dropoutMaskIO :: forall a s . (TensorScalar a, Shape s)
              => CuRAND.Generator
              -> a
              -> IO (IOTensor s a)
dropoutMaskIO gen drop_proba = do
  -- Simple algo for dropout of activations:
  -- 1- generate an array of random values between 0 and 1
  -- 2- threshold that array with the dropout probability
  -- 3- elementwise multiply the input array with it
  rand_array <- emptyTensor
  withDevicePtr rand_array $ \randarrayptr -> do
    -- generate random array
    generateUniform gen randarrayptr $ fromIntegral $ size (Proxy :: Proxy s)
    -- threshold it
    thresh randarrayptr (fromIntegral $ size (Proxy :: Proxy s)) drop_proba randarrayptr
    return rand_array

-- generating random tensors
uniform :: forall a s . (TensorScalar a, Shape s) => CUDA (Tensor s a)
uniform = do
  gen <- asks generator
  liftIO $ do
    res <- MT.emptyTensor
    withDevicePtr res $ \resptr -> do
      generateUniform gen resptr $ fromIntegral $ size (Proxy :: Proxy s)
    unsafeFreeze res

normal :: forall a s . (TensorScalar a, Shape s) => a -> a -> CUDA (Tensor s a)
normal mean std = do
  gen <- asks generator
  liftIO $ do
    res <- MT.emptyTensor
    withDevicePtr res $ \resptr -> do
      generateNormal gen resptr (fromIntegral $ size (Proxy :: Proxy s)) mean std
    unsafeFreeze res

logNormal :: forall a s . (TensorScalar a, Shape s) => a -> a -> CUDA (Tensor s a)
logNormal mean std = do
  gen <- asks generator
  liftIO $ do
    res <- emptyTensor
    withDevicePtr res $ \resptr -> do
      generateLogNormal gen resptr (fromIntegral $ size (Proxy :: Proxy s)) mean std
    unsafeFreeze res
