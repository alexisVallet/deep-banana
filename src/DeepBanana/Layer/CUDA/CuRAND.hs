{-# LANGUAGE TypeFamilies, OverloadedStrings #-}
module DeepBanana.Layer.CUDA.CuRAND (
    uniform
  , normal
  , logNormal
  , dropout
  ) where

import Foreign.Marshal
import System.IO.Unsafe
import Unsafe.Coerce

import DeepBanana.Device
import qualified DeepBanana.Device.CuRAND as CuRAND
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception
import qualified DeepBanana.Tensor.Mutable as MT

uniform :: (MonadCuda d m, Device d, TensorScalar a, Shape (Dim n))
        => Dim n -> m (Tensor d n a)
uniform shp = embedCudaFromST $ do
  res <- MT.emptyTensor shp
  gen <- get
  mgen <- unsafeThawGen gen
  uniformM mgen res
  unsafeFreezeGen mgen >>= put
  unsafeFreeze res

normal :: (MonadCuda d m, Device d, TensorScalar a, Shape (Dim n))
       => Dim n -> a -> a -> m (Tensor d n a)
normal shp mean std = embedCudaFromST $ do
  res <- MT.emptyTensor shp
  gen <- get
  mgen <- unsafeThawGen gen
  normalM mgen res mean std
  unsafeFreezeGen mgen >>= put
  unsafeFreeze res

logNormal :: (MonadCuda d m, Device d, TensorScalar a, Shape (Dim n))
          => Dim n -> a -> a -> m (Tensor d n a)
logNormal shp mean std = embedCudaFromST $ do
  res <- MT.emptyTensor shp
  gen <- get
  mgen <- unsafeThawGen gen
  logNormalM mgen res mean std
  unsafeFreezeGen mgen >>= put
  unsafeFreeze res  

-- dropout
dropout :: (MonadCuda d m, Device d, TensorScalar a, Shape (Dim n))
        => a
        -> Layer m a '[] (Tensor d n a) (Tensor d n a)
dropout drop_proba = Layer $ \_ x -> embedCudaFromST $ do
  rand <- MT.emptyTensor $ shape x
  gen <- get
  mgen <- unsafeThawGen gen
  uniformM mgen rand
  MT.threshInplace rand drop_proba
  unsafeFreezeGen mgen >>= put
  mask <- unsafeFreeze rand
  return (x * mask, \upgrad -> (W Z, upgrad * mask))

uniformM :: forall m n d a
         . (PrimMonad m, Shape (Dim n), Device d, TensorScalar a)
         => MGenerator (PrimState m) d
         -> MT.MTensor (PrimState m) d n a
         -> m ()
uniformM gen t = unsafeIOToPrim $ do
  withRawGen (unsafeCoerce gen :: MGenerator RealWorld d) $ \rawGen -> do
    MT.withDevicePtr (unsafeCoerce t :: MT.IOTensor d n a) $ \tptr -> do
      runDeviceM (Proxy :: Proxy d)
        $ generateUniform rawGen tptr (fromIntegral $ size $ MT.shape t)
      return ()

normalM :: forall m n d a
        . (PrimMonad m, Shape (Dim n), Device d, TensorScalar a)
        => MGenerator (PrimState m) d
        -> MT.MTensor (PrimState m) d n a
        -> a
        -> a
        -> m ()
normalM gen t mean std = unsafeIOToPrim $ withRawGen (unsafeCoerce gen :: MGenerator RealWorld d) $ \rawGen -> do
  MT.withDevicePtr (unsafeCoerce t :: MT.IOTensor d n a) $ \tptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ generateNormal rawGen tptr (fromIntegral $ size $ MT.shape t) mean std
    return ()

logNormalM :: forall m n d a
           . (PrimMonad m, Shape (Dim n), Device d, TensorScalar a)
           => MGenerator (PrimState m) d
           -> MT.MTensor (PrimState m) d n a
           -> a
           -> a
           -> m ()
logNormalM gen t mean std = unsafeIOToPrim $ withRawGen (unsafeCoerce gen :: MGenerator RealWorld d) $ \rawGen -> do
  MT.withDevicePtr (unsafeCoerce t :: MT.IOTensor d n a) $ \tptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ generateLogNormal rawGen tptr (fromIntegral $ size $ MT.shape t) mean std
    return ()
