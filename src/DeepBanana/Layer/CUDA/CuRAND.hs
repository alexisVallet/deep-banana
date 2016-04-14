{-# LANGUAGE TypeFamilies, OverloadedStrings #-}
module DeepBanana.Layer.CUDA.CuRAND (
    uniform
  , normal
  , logNormal
  , dropout
  ) where
import Debug.Trace
import Foreign.Marshal
import System.IO.Unsafe
import Unsafe.Coerce

import DeepBanana.Device
import qualified DeepBanana.Device.Monad as DeviceM
import qualified DeepBanana.Device.CuRAND as CuRAND
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception
import qualified DeepBanana.Tensor.Mutable as MT

uniform :: forall m n d a
        . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
        => Dim n -> m (Tensor d n a)
uniform shp = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  let outSize = size shp
  res <- MT.emptyTensor shp :: CudaT IO (MT.IOTensor d n a)  
  liftIO $ MT.withDevicePtr res $ \resptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ withGenerator gen $ \rawGen -> do
      generateUniform rawGen resptr (fromIntegral outSize)
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  unsafeFreeze res >>= return . unsafeCoerce

normal :: forall m d n a
       . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
       => Dim n -> a -> a -> m (Tensor d n a)
normal shp mean std = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  let outSize = size shp
  res <- MT.emptyTensor shp :: CudaT IO (MT.IOTensor d n a)  
  liftIO $ MT.withDevicePtr res $ \resptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ withGenerator gen $ \rawGen -> do
      generateNormal rawGen resptr (fromIntegral outSize) mean std
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  unsafeFreeze res >>= return . unsafeCoerce

logNormal :: forall m d n a
       . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
       => Dim n -> a -> a -> m (Tensor d n a)
logNormal shp mean std = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  let outSize = size shp
  res <- MT.emptyTensor shp :: CudaT IO (MT.IOTensor d n a)  
  liftIO $ MT.withDevicePtr res $ \resptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ withGenerator gen $ \rawGen -> do
      generateLogNormal rawGen resptr (fromIntegral outSize) mean std
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  unsafeFreeze res >>= return . unsafeCoerce

-- dropout
dropout :: forall m d n a
        . (MonadCuda m, Device d, TensorScalar a, Shape (Dim n))
        => a
        -> Layer m a '[] (Tensor d n a) (Tensor d n a)
dropout drop_proba = Layer $ \_ x -> embedCudaFromST $ do
  let outSize = size $ shape x
  mask <- embedCuda unsafeIOToPrim $ do
    mmask <- MT.emptyTensor $ shape x :: CudaT IO (MT.IOTensor d n a)
    gen <- get
    liftIO $ do
      MT.withDevicePtr mmask $ \maskptr -> do
        runDeviceM (Proxy :: Proxy d) $ do
          withGenerator gen $ \rawGen -> do
            generateUniform rawGen maskptr (fromIntegral outSize)
      MT.threshInplace mmask drop_proba
    unsafeFreeze mmask >>= return . unsafeCoerce
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  return (x * mask, broadcast' (shape x) >>> \upgrad -> (W Z, upgrad * mask))
