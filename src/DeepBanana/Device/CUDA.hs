module DeepBanana.Device.CUDA (
    module Foreign.CUDA
  , mallocArray
  , newListArray
  , free
  , pokeArray
  , copyArray
  , peekListArray
  , peekArray
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

mallocArray size =
  unsafeIOToDevice $ CUDA.mallocArray size

newListArray xs =
  unsafeIOToDevice $ CUDA.newListArray xs

free devptr =
  unsafeIOToDevice $ CUDA.free devptr

pokeArray size inp out =
  unsafeIOToDevice $ CUDA.pokeArray size inp out

copyArray size inp out =
  unsafeIOToDevice $ CUDA.copyArray size inp out

peekListArray size devptr =
  unsafeIOToDevice $ CUDA.peekListArray size devptr

peekArray size inp out =
  unsafeIOToDevice $ CUDA.peekArray size inp out
