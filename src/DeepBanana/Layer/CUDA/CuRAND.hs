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
dropout :: (TensorScalar a, Shape (Dim n))
        => a
        -> Layer CUDA a '[] (Tensor n a) (Tensor n a)
dropout drop_proba = noWeights fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd inp = do
          gen <- asks generator
          mask <- liftIO $ dropoutMaskIO gen (shape inp) drop_proba
          pure_mask <- unsafeFreeze mask
          return (inp * pure_mask, \upgrad -> upgrad * pure_mask)

-- dropout
-- compute a random mask of ones and zeros
-- to apply elementwise to the input tensor.
dropoutMaskIO :: (TensorScalar a, Shape (Dim n))
              => CuRAND.Generator
              -> Dim n
              -> a
              -> IO (IOTensor n a)
dropoutMaskIO gen shp drop_proba = do
  -- Simple algo for dropout of activations:
  -- 1- generate an array of random values between 0 and 1
  -- 2- threshold that array with the dropout probability
  -- 3- elementwise multiply the input array with it
  rand_array <- emptyTensor shp
  withDevicePtr rand_array $ \randarrayptr -> do
    -- generate random array
    generateUniform gen randarrayptr $ fromIntegral $ size shp
    -- threshold it
    thresh randarrayptr (fromIntegral $ size shp) drop_proba randarrayptr
    return rand_array

-- generating random tensors
uniform :: (TensorScalar a, Shape (Dim n)) => Dim n -> CUDA (Tensor n a)
uniform shp = do
  gen <- asks generator
  liftIO $ do
    res <- MT.emptyTensor shp
    withDevicePtr res $ \resptr -> do
      generateUniform gen resptr $ fromIntegral $ size shp
    unsafeFreeze res

normal :: (TensorScalar a, Shape (Dim n)) => Dim n -> a -> a -> CUDA (Tensor n a)
normal shp mean std = do
  gen <- asks generator
  liftIO $ do
    res <- MT.emptyTensor shp 
    withDevicePtr res $ \resptr -> do
      generateNormal gen resptr (fromIntegral $ size shp) mean std
    unsafeFreeze res

logNormal :: (TensorScalar a, Shape (Dim n)) => Dim n -> a -> a -> CUDA (Tensor n a)
logNormal shp mean std = do
  gen <- asks generator
  liftIO $ do
    res <- emptyTensor shp
    withDevicePtr res $ \resptr -> do
      generateLogNormal gen resptr (fromIntegral $ size shp) mean std
    unsafeFreeze res
