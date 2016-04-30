{-# LANGUAGE PolyKinds, MultiParamTypeClasses, TypeFamilies, UndecidableInstances, ScopedTypeVariables, OverloadedStrings, BangPatterns #-}
{-|
Device manipulation functions. Notably, initialization of a random number generator for
a given device, and a generic interface for transferring data from one device to another.
-}
module DeepBanana.Device (
  -- * Re-exports
    module DeepBanana.Device.Monad
  -- * Random number generator
  , Generator(..)
  , generator
  , withGenerator
  -- * Transferring data between devices
  , DeviceTransfer(..)
  , transfer'
  ) where
import qualified DeepBanana.Device.Monad as DeviceM
import DeepBanana.Device.Monad (
    Device(..)
  , FixedDevice
  , NoSuchDevice
  , getFixedDevice
  , ValidUniqueDevice(..)
  , withValidUniqueDevice
  , DeviceM
  , runDeviceM
  )
import Control.Monad.Primitive
import Debug.Trace
import Foreign
import Foreign.ForeignPtr

import qualified DeepBanana.Device.Cubits as Cubits
import qualified DeepBanana.Device.CuRAND as CuRAND
import DeepBanana.Exception
import DeepBanana.Tensor.Exception
import DeepBanana.Prelude

-- | Pure random number generator for use with CuRAND.
data Generator = Generator {
    seed :: CULLong
  , offset :: CULLong
  } deriving (Eq, Ord, Show)

-- | Initializes the generator with a given seed.
generator :: CULLong -> Generator
generator seed = Generator {
    seed = seed
  , offset = 0
  }

-- | Gives access to a raw @'DeepBanana.Device.CuRAND.Generator'@ for call to
-- low-level 
withGenerator :: (Device d)
              => Generator -> (CuRAND.Generator d -> DeviceM d a) -> DeviceM d a
withGenerator gen action = do
  genptr <- DeviceM.unsafeIOToDevice $ malloc
  CuRAND.createGenerator genptr CuRAND.rng_pseudo_default
  rawGen <- DeviceM.unsafeIOToDevice $ peek genptr
  CuRAND.setPseudoRandomGeneratorSeed rawGen $ seed gen
  CuRAND.setGeneratorOffset rawGen $ offset gen
  res <- action rawGen
  CuRAND.destroyGenerator rawGen
  DeviceM.unsafeIOToDevice $ free genptr
  return res

-- | @'DeviceTransfer' t1 t2@ if and only if you can copy the data of t1 from
-- its current device to a different device. Ad-hoc type class to overload data
-- transfers of both individual @'Tensor'@s and @'Weights'@.
class (Device d) => DeviceTransfer t d where
  type Transferred t d :: *
  -- | Transfers data from one device to another.
  transfer :: (MonadError e m, Variant e OutOfMemory) => d -> t -> m (Transferred t d)

-- | Unsafe version of @'transfer'@, does not check that the output device
-- has enough memory.
transfer' :: forall t d . (DeviceTransfer t d) => d -> t -> Transferred t d
transfer' d t = unsafeRunExcept (transfer d t :: Either OutOfMemory (Transferred t d))
