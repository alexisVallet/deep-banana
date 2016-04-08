{-# LANGUAGE UndecidableInstances #-}
module DeepBanana.Device.Monad (
    Device(..)
  , DeviceM(..)
  , unsafeIOToDevice
  , runDeviceM
  , cudnnHandle
  , cublasHandle
  , deviceMallocArray
  ) where

import Control.Concurrent (runInBoundThread)
import Control.Monad.Primitive
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA as CUDA
import System.IO.Unsafe (unsafePerformIO)

import DeepBanana.Prelude

newtype DeviceM d a = DeviceM {
  unsafeRunDeviceM :: IO a
  }

instance Functor (DeviceM d) where
  fmap f (DeviceM action) = DeviceM $ fmap f action

instance Applicative (DeviceM d) where
  pure f = DeviceM $ pure f
  DeviceM mab <*> DeviceM ma = DeviceM $ mab <*> ma

instance Monad (DeviceM d) where
  return x = DeviceM $ return x
  DeviceM ma >>= admb = DeviceM $ do
    a <- ma
    let DeviceM mb = admb a
    mb

instance PrimMonad (DeviceM d) where
  type PrimState (DeviceM d) = RealWorld
  primitive action = DeviceM $ primitive action

class Device (d :: k) where
  deviceId :: Proxy d -> Int

instance (KnownNat n) => Device n where
  deviceId p = fromIntegral $ natVal p

runDeviceM :: forall d a . (Device d) => Proxy d -> DeviceM d a -> IO a
runDeviceM p action =
  runInBoundThread $ bracket
  (do
      prevDevice <- CUDA.get
      CUDA.set $ deviceId p
      return prevDevice)
  (\prevDevice -> CUDA.set prevDevice)
  (\_ -> unsafeRunDeviceM action)

unsafeIOToDevice :: IO a -> DeviceM d a
unsafeIOToDevice = DeviceM

onDevice :: forall d a . Device d => Proxy d -> IO a -> IO a
onDevice p action = runDeviceM p $ unsafeIOToDevice action

global_cudnn_handle :: IORef (Maybe CuDNN.Handle)
{-# NOINLINE global_cudnn_handle #-}
global_cudnn_handle = unsafePerformIO $ newIORef Nothing

cudnnHandle :: forall d . (Device d) => Proxy d -> CuDNN.Handle
cudnnHandle p = unsafePerformIO $ onDevice p $ do
  mh <- readIORef global_cudnn_handle
  case mh of
   Nothing -> do
     h <- alloca $ \hptr -> CuDNN.createHandle hptr >> peek hptr
     writeIORef global_cudnn_handle $ Just h
     return h
   Just h -> return h

global_cublas_handle :: IORef (Maybe Cublas.Handle)
{-# NOINLINE global_cublas_handle #-}
global_cublas_handle = unsafePerformIO $ newIORef Nothing

cublasHandle :: forall d . (Device d) => Proxy d -> Cublas.Handle
cublasHandle p = unsafePerformIO $ onDevice p $ do
  mh <- readIORef global_cublas_handle
  case mh of
   Nothing -> do
     h <- Cublas.create
     writeIORef global_cublas_handle $ Just h
     return h
   Just h -> return h

deviceMallocArray :: forall d a . (Device d, Storable a)
                  => Proxy d -> Int -> IO (CUDA.DevicePtr a)
deviceMallocArray p size = onDevice p $ CUDA.mallocArray size
