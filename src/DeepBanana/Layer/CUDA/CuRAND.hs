{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA.CuRAND (
    uniform
  , normal
  , logNormal
  , dropout
  ) where

import Foreign.Marshal
import qualified Foreign.CUDA.CuRAND as CuRAND
import System.IO.Unsafe

import DeepBanana.Device
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception
import qualified DeepBanana.Tensor.Mutable as MT

uniform :: forall m d n a
        . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
        => Dim n -> m (Tensor d n a)
uniform shp = do
  eres <- unsafeCuRandWrap $ \gen -> runExceptT $ do
    res <- MT.emptyTensor shp
    liftIO $ MT.withDevicePtr res $ \resptr -> do
      generateUniform (Proxy :: Proxy d) gen resptr $ fromIntegral $ size shp
    unsafeFreeze res
  embedExcept eres

normal :: forall m d n a
       . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
       =>  Dim n -> a -> a -> m (Tensor d n a)
normal shp mean std = do
  eres <- unsafeCuRandWrap $ \gen -> runExceptT $ do
    res <- MT.emptyTensor shp
    liftIO $ MT.withDevicePtr res $ \resptr -> do
      generateNormal (Proxy :: Proxy d) gen resptr (fromIntegral $ size shp) mean std
    unsafeFreeze res
  embedExcept eres

logNormal :: forall m d n a
          . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
          => Dim n -> a -> a -> m (Tensor d n a)
logNormal shp mean std = do
  eres <- unsafeCuRandWrap $ \gen -> runExceptT $ do
    res <- MT.emptyTensor shp
    liftIO $ MT.withDevicePtr res $ \resptr -> do
      generateLogNormal (Proxy :: Proxy d) gen resptr (fromIntegral $ size shp) mean std
    unsafeFreeze res
  embedExcept eres

-- dropout
dropout :: (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
        => a
        -> Layer m a '[] (Tensor d n a) (Tensor d n a)
dropout drop_proba = noWeights fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd inp = do
          epure_mask <- unsafeCuRandWrap $ \gen -> runExceptT $ do
            mask <- dropoutMaskIO gen (shape inp) drop_proba
            unsafeFreeze mask
          pure_mask <- embedExcept epure_mask
          return (inp * pure_mask,
                  broadcast' (shape inp) >>> \upgrad -> upgrad * pure_mask)

-- dropout
-- compute a random mask of ones and zeros
-- to apply elementwise to the input tensor.
dropoutMaskIO :: forall m t d n a
              . (MonadIO m, PrimMonad m, PrimState m ~ RealWorld,
                 MonadError t m, Variant t OutOfMemory,
                 TensorScalar a, Shape (Dim n), Device d)
              => CuRAND.Generator
              -> Dim n
              -> a
              -> m (MT.IOTensor d n a)
dropoutMaskIO gen shp drop_proba = do
  -- Simple algo for dropout of activations:
  -- 1- generate an array of random values between 0 and 1
  -- 2- threshold that array with the dropout probability
  -- 3- elementwise multiply the input array with it
  rand_array <- MT.emptyTensor shp
  liftIO $ MT.withDevicePtr rand_array $ \randarrayptr -> do
    -- generate random array
    generateUniform (Proxy :: Proxy d) gen randarrayptr $ fromIntegral $ size shp
    -- threshold it
    thresh (Proxy :: Proxy d) randarrayptr (fromIntegral $ size shp)
      drop_proba randarrayptr
    return rand_array
