{-# LANGUAGE UndecidableInstances, BangPatterns, OverloadedStrings #-}
module DeepBanana.Device.Monad (
    Device(..)
  , DeviceM(..)
  , unsafeIOToDevice
  , runDeviceM
  , onDevice
  ) where

import Control.Concurrent (runInBoundThread)
import Control.Monad.Primitive
import Data.MemoTrie
import Debug.Trace
import qualified Foreign.CUDA as CUDA
import System.IO.Unsafe

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

class Device d where
  deviceId :: Proxy d -> Int

instance (KnownNat n) => Device n where
  deviceId p = fromIntegral $ natVal p

runDeviceM :: forall d a . (Device d) => Proxy d -> DeviceM d a -> IO a
runDeviceM p action = do
  runInBoundThread $ bracket
    (do
        prevDevice <- CUDA.get
        CUDA.set $ deviceId p
        return prevDevice)
    (\prevDevice -> do
        CUDA.set prevDevice)
    (\_ -> do
        unsafeRunDeviceM action)

unsafeIOToDevice :: IO a -> DeviceM d a
unsafeIOToDevice = DeviceM

onDevice :: forall d a . Device d => Proxy d -> IO a -> IO a
onDevice p action = runDeviceM p $ unsafeIOToDevice action

instance forall t . HasTrie (Proxy t) where
  newtype (Proxy t :->: a) = ProxyTrie a
  trie f = ProxyTrie $ f (Proxy :: Proxy t)
  untrie (ProxyTrie a) = \Proxy -> a
  enumerate (ProxyTrie a) = [(Proxy :: Proxy t, a)]
