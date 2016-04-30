{-# LANGUAGE UndecidableInstances, BangPatterns, OverloadedStrings, DeriveGeneric, GeneralizedNewtypeDeriving, ImplicitParams, RankNTypes #-}
{-|
Defines the @'Device'@ type class and @'DeviceM'@ monad, ensuring all low-level
operations happen with the right CUDA device set. This is necessary due to the fact
that the CUDA runtime library keeps track of a hidden current device state separately
for each OS thread. Not using this would result in horrendous crashes whenever one uses
haskell threads, or even when calling @'CUDA.set'@ between a thunk's creation and its
evaluation.

Currently, it does so by using @'runInBoundThread'@, which has the significant overhead
of creating an OS thread whenever launched from a non-bound thread. This overhead should
be insignificant compared running expensive convolutions on the GPU anyway, but I do
suggest using bound threads only with this library.
-}
module DeepBanana.Device.Monad (
    Device(..)
  , FixedDevice
  , getFixedDevice
  , ValidUniqueDevice(..)
  , withValidUniqueDevice
  , NoSuchDevice
  , DeviceM(..)
  , unsafeIOToDevice
  , runDeviceM
  , onDevice
  ) where

import Control.Concurrent (runInBoundThread)
import Control.Monad.Primitive
import Data.MemoTrie
import Data.Serialize (Serialize(..))
import Debug.Trace
import qualified Foreign.CUDA as CUDA
import System.IO.Unsafe

import DeepBanana.Exception
import DeepBanana.Prelude

-- | The @'DeviceM' d@ monad ensures all enclosed computations take place with the
-- right CUDA device set, barring unsafe operations.
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

newtype NoSuchDevice = NoSuchDevice (WithStack String)
                       deriving (Eq, Show, Typeable, Exception, Generic, NFData)

class (Eq d, Show d, Serialize d, NFData d) => Device d where
  deviceId :: d -> Natural

newtype FixedDevice (n :: Nat) = FixedDevice {
  unFixedDevice :: Proxy n
  } deriving (Eq, Ord, Show)

instance NFData (FixedDevice n) where
  rnf (FixedDevice !p) = ()

instance forall n . Serialize (FixedDevice n) where
  put _ = return ()
  get = return $ FixedDevice (Proxy :: Proxy n)

instance (KnownNat n) => Device (FixedDevice n) where
  deviceId f = fromIntegral $ natVal $ unFixedDevice f

getFixedDevice :: forall n m e
               . (MonadIO m, MonadError e m, Variant e NoSuchDevice, KnownNat n)
               => m (FixedDevice n)
getFixedDevice = do
  let nval = natVal (Proxy :: Proxy n)
  nbDevices <- liftIO $ CUDA.count
  when (fromIntegral nbDevices <= nval) $ throwVariant $ NoSuchDevice $ withStack $
    "No device with id " ++ show nval ++ ": only " ++ show nbDevices ++ " devices available."
  return $ FixedDevice (Proxy :: Proxy n)

-- | This type class should only be implemented by device datatypes which are unique
-- (i.e. any value of the type corresponds to a single common device id), and have
-- been checked at runtime to actually be valid, usable devices. Such an instance
-- can only be provided locally - yet completely safely - by @'withValidUniqueDevice'@.
class (Device d) => ValidUniqueDevice d where
  validUniqueDevice :: d

data ValidUniqueDeviceT (n :: Nat) = ValidUniqueDeviceT
                                     deriving (Eq, Ord, Show, Generic)

instance NFData (ValidUniqueDeviceT n)

instance Serialize (ValidUniqueDeviceT n)

instance forall n . (KnownNat n) => Device (ValidUniqueDeviceT n) where
  deviceId _ = fromIntegral $ natVal (Proxy :: Proxy n)

instance forall n . (KnownNat n) => ValidUniqueDevice (ValidUniqueDeviceT n) where
  validUniqueDevice = ValidUniqueDeviceT

-- | Locally converts an arbitrary device to a device that is guaranteed to be valid
-- and unique at the type level within a local context.
withValidUniqueDevice :: (Device d)
                      => d -> (forall d' . ValidUniqueDevice d' => d' -> a) -> a
withValidUniqueDevice d f =
  case someNatVal $ fromIntegral $ deviceId d of
   Just (SomeNat (p :: Proxy n)) -> f (ValidUniqueDeviceT :: ValidUniqueDeviceT n)
   Nothing -> error $ "Negative device id in withValidUniqueDevice: " ++ show d ++ ", this shouldn't happen!"

-- | Runs a computation on a specific device.
runDeviceM :: (Device d) => d -> DeviceM d a -> IO a
runDeviceM d action = do
  runInBoundThread $ bracket
    (do
        prevDevice <- CUDA.get
        CUDA.set $ fromIntegral $ deviceId d
        return prevDevice)
    (\prevDevice -> do
        CUDA.set prevDevice)
    (\_ -> do
        unsafeRunDeviceM action)

-- | Lifts an IO computation to a device computation. Note that this is only unsafe
-- whenever the computation changes the CUDA device and does not bring it back to its
-- original state before returning. Notably, calling @'runDeviceM'@ for some other
-- device within @'unsafeIOToDevice'@ is perfectly safe.
unsafeIOToDevice :: IO a -> DeviceM d a
unsafeIOToDevice = DeviceM

-- | Runs an IO action with a specific device.
onDevice :: Device d => d -> IO a -> IO a
onDevice d action = runDeviceM d $ unsafeIOToDevice action
