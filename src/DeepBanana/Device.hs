{-# LANGUAGE PolyKinds, MultiParamTypeClasses, TypeFamilies, UndecidableInstances, ScopedTypeVariables, OverloadedStrings, BangPatterns #-}
module DeepBanana.Device (
    module DeepBanana.Device.Monad
  , module DeepBanana.Device.CuRAND
  , MGenerator(..)
  , Generator(..)
  , generator
  , newMutableGenerator
  , unsafeFreezeGen
  , unsafeThawGen
  , withRawGen
  ) where
import qualified DeepBanana.Device.Monad as DeviceM
import DeepBanana.Device.Monad (
    Device
  , DeviceM
  , runDeviceM
  )
import Control.Monad.Primitive
import Foreign
import Foreign.ForeignPtr

import qualified DeepBanana.Device.Cubits as Cubits
import qualified DeepBanana.Device.CuRAND as CuRAND
import DeepBanana.Device.CuRAND (
    RngType
  , rng_test
  , rng_pseudo_default
  , rng_pseudo_xorwow
  , rng_pseudo_mrg32k3a
  , rng_pseudo_mt19937
  , rng_pseudo_philox4_32_10
  , rng_quasi_default
  , rng_quasi_sobol32
  , rng_quasi_scrambled_sobol32
  , rng_quasi_sobol64
  , rng_quasi_scrambled_sobol64
  )
import DeepBanana.Prelude

newtype MGenerator st d = MGenerator {
  mgenfptr :: ForeignPtr ()
  }

newMutableGenerator :: forall m d . (PrimMonad m, Device d)
                    => CuRAND.RngType -> CULLong -> m (MGenerator (PrimState m) d)
newMutableGenerator rngType seed = unsafePrimToPrim $ do
  genptr <- malloc
  let p = Proxy :: Proxy d
  runDeviceM p $ CuRAND.createGenerator genptr rngType
  gen <- peek genptr
  runDeviceM p $ CuRAND.setPseudoRandomGeneratorSeed gen seed
  let CuRAND.Generator genOpaquePtr = gen
  datafptr <- newForeignPtr Cubits.freeCuRANDGenerator genOpaquePtr
  return $ MGenerator datafptr

newtype Generator d = Generator {
  genfptr :: ForeignPtr ()
  }

unsafeFreezeGen :: (PrimMonad m) => MGenerator (PrimState m) d -> m (Generator d)
unsafeFreezeGen (MGenerator genfptr) = return $ Generator genfptr

unsafeThawGen :: (PrimMonad m) => Generator d -> m (MGenerator (PrimState m) d)
unsafeThawGen (Generator genfptr) = return $ MGenerator genfptr

generator :: (Device d) => CuRAND.RngType -> CULLong -> Generator d
generator rngType seed = runST $ do
  mgen <- newMutableGenerator rngType seed
  unsafeFreezeGen mgen

withRawGen :: (Device d)
           => MGenerator RealWorld d
           -> (CuRAND.Generator d -> IO a)
           -> IO a
withRawGen (MGenerator datafptr) action = withForeignPtr datafptr $ \genOpaquePtr -> do
  action $ CuRAND.Generator genOpaquePtr
