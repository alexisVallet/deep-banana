{-# LANGUAGE TypeFamilies, OverloadedStrings #-}
module DeepBanana.Layer.CUDA.CuRAND (
    splitGenerator
  , uniform
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
import qualified DeepBanana.Device.CUDA as CUDA
import qualified DeepBanana.Device.CuRAND as CuRAND
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception
import qualified DeepBanana.Tensor.Mutable as MT

-- Naively splits the generator by reseeding one with a random value from the
-- current generator. Probably not so great statistically, but should do the job
-- for our purposes.
splitGenerator :: forall m d . (MonadCuda m, Device d) => Proxy d -> m Generator
splitGenerator p = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  newSeed <- liftIO $ runDeviceM p $ withGenerator gen $ \rawGen -> do
    res <- CUDA.mallocArray 1
    CuRAND.generateLongLong rawGen res 1
    [newSeed] <- CUDA.peekListArray 1 res
    CUDA.free res
    return newSeed
  return $ Generator newSeed 0

uniform :: forall m s d a
        . (MonadCuda m, Device d, TensorScalar a, Shape s)
        => s -> m (Tensor d s a)
uniform shp = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  let outSize = size shp
  res <- MT.emptyTensor shp :: CudaT IO (MT.IOTensor d s a)  
  liftIO $ MT.withDevicePtr res $ \resptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ withGenerator gen $ \rawGen -> do
      generateUniform rawGen resptr (fromIntegral outSize)
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  unsafeFreeze res >>= return . unsafeCoerce

normal :: forall m d s a
       . (MonadCuda m, Device d, TensorScalar a, Shape s)
       => s -> a -> a -> m (Tensor d s a)
normal shp mean std = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  let outSize = size shp
  res <- MT.emptyTensor shp :: CudaT IO (MT.IOTensor d s a)  
  liftIO $ MT.withDevicePtr res $ \resptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ withGenerator gen $ \rawGen -> do
      generateNormal rawGen resptr (fromIntegral outSize) mean std
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  unsafeFreeze res >>= return . unsafeCoerce

logNormal :: forall m d s a
       . (MonadCuda m, Device d, TensorScalar a, Shape s)
       => s -> a -> a -> m (Tensor d s a)
logNormal shp mean std = embedCudaFromST $ embedCuda unsafeIOToPrim $ do
  gen <- get
  let outSize = size shp
  res <- MT.emptyTensor shp :: CudaT IO (MT.IOTensor d s a)  
  liftIO $ MT.withDevicePtr res $ \resptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ withGenerator gen $ \rawGen -> do
      generateLogNormal rawGen resptr (fromIntegral outSize) mean std
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  unsafeFreeze res >>= return . unsafeCoerce

-- dropout
dropout :: forall m d s a
        . (MonadCuda m, Device d, TensorScalar a, Shape s)
        => a
        -> Layer m a '[] (Tensor d s a) (Tensor d s a)
dropout drop_proba = Layer $ \_ x -> embedCudaFromST $ do
  let outSize = size $ shape x
  mask <- embedCuda unsafeIOToPrim $ do
    mmask <- MT.emptyTensor $ shape x :: CudaT IO (MT.IOTensor d s a)
    gen <- get
    liftIO $ do
      MT.withDevicePtr mmask $ \maskptr -> do
        runDeviceM (Proxy :: Proxy d) $ do
          withGenerator gen $ \rawGen -> do
            generateUniform rawGen maskptr (fromIntegral outSize)
      MT.threshInplace mmask drop_proba
    unsafeFreeze mmask >>= return . unsafeCoerce :: CudaT IO (Tensor d s a)
  modify (\gen -> gen {offset = offset gen + fromIntegral outSize})
  case toAnyFixed $ shape x of
   AnyFixed fshape -> do
     fx <- shapeConvert fshape x
     fmask <- shapeConvert fshape mask
     y <- shapeConvert (shape x) $ fx * fmask
     return (y, broadcast' (shape x) >>> \upgrad -> unsafeRunCudaError $ do
                fdy <- shapeConvert fshape upgrad
                dx <- shapeConvert (shape x) $ fdy * fmask
                return (W Z, dx))
