{-# LANGUAGE PolyKinds, MultiParamTypeClasses, TypeFamilies, UndecidableInstances, ScopedTypeVariables, OverloadedStrings, BangPatterns #-}
module DeepBanana.Device (
    module DeepBanana.Device.Monad
  , Generator(..)
  , generator
  , withGenerator
  ) where
import qualified DeepBanana.Device.Monad as DeviceM
import DeepBanana.Device.Monad (
    Device
  , DeviceM
  , runDeviceM
  )
import Control.Monad.Primitive
import Debug.Trace
import Foreign
import Foreign.ForeignPtr

import qualified DeepBanana.Device.Cubits as Cubits
import qualified DeepBanana.Device.CuRAND as CuRAND
import DeepBanana.Prelude

data Generator = Generator {
    seed :: CULLong
  , offset :: CULLong
  } deriving (Eq, Ord, Show)

generator :: CULLong -> Generator
generator seed = Generator {
    seed = seed
  , offset = 0
  }

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
