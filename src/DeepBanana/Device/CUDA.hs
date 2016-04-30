{-|
Wrapper for the few functions and datatypes needed from the cuda package, using
the @'DeviceM'@ monad to make them safe to use from pure code.
See @'Foreign.CUDA'@ for documentation.
-}
module DeepBanana.Device.CUDA (
    mallocArray
  , newListArray
  , free
  , pokeArray
  , copyArray
  , peekListArray
  , peekArray
  -- * Re-exports
  , module Foreign.CUDA
  ) where

import qualified Foreign.CUDA as CUDA
import Foreign.CUDA (
    DevicePtr(..)
  , useDevicePtr
  , CUDAException(..)
  , Status(..)
  )

import DeepBanana.Device.Monad
import DeepBanana.Prelude hiding (mallocArray, free, pokeArray, copyArray, peekArray)

-- | See @'Foreign.CUDA.mallocArray'@ for documentation.
mallocArray size =
  unsafeIOToDevice $ CUDA.mallocArray size

-- | See @'Foreign.CUDA.newListArray'@ for documentation.
newListArray xs =
  unsafeIOToDevice $ CUDA.newListArray xs

-- | See @'Foreign.CUDA.free'@ for documentation.
free devptr =
  unsafeIOToDevice $ CUDA.free devptr

-- | See @'Foreign.CUDA.pokeArray'@ for documentation.
pokeArray size inp out =
  unsafeIOToDevice $ CUDA.pokeArray size inp out

-- | See @'Foreign.CUDA.copyArray'@ for documentation.
copyArray size inp out =
  unsafeIOToDevice $ CUDA.copyArray size inp out

-- | See @'Foreign.CUDA.peekListArray'@ for documentation.
peekListArray size devptr =
  unsafeIOToDevice $ CUDA.peekListArray size devptr

-- | See @'Foreign.CUDA.peekArray'@ for documentation.
peekArray size inp out =
  unsafeIOToDevice $ CUDA.peekArray size inp out
