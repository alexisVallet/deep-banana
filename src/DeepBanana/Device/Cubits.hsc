{-# LANGUAGE ForeignFunctionInterface #-}
{-|
FFI wrapper to custom CUDA code provided by the c library hnn_cubits.
-}
module DeepBanana.Device.Cubits where

import Foreign.CUDA.Types

import DeepBanana.Device.Monad
import DeepBanana.Device.CuRAND
import DeepBanana.Prelude

#include <hnn_cubits.h>

foreign import ccall  "thresh"
  thresh :: DevicePtr CFloat -> CSize -> CFloat -> DevicePtr CFloat
         -> DeviceM d ()

foreign import ccall  "threshDouble"
  threshDouble :: DevicePtr CDouble -> CSize -> CDouble
               -> DevicePtr CDouble -> DeviceM d ()

foreign import ccall  "mul"
  mul :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "mulDouble"
  mulDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "add"
  add :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "addDouble"
  addDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "tabs"
  tabs :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "tabsDouble"
  tabsDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "signum"
  tsignum :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "signumDouble"
  tsignumDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "subtract"
  subtract :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "subtractDouble"
  subtractDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "negate"
  tnegate :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "negateDouble"
  tnegateDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "scale"
  scale :: CFloat -> DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "scaleDouble"
  scaleDouble :: CDouble -> DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "logFloat"
  logFloat :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "logDouble"
  logDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall  "inv"
  inv :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall  "invDouble"
  invDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "texp"
  texp :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "texpDouble"
  texpDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tsqrt"
  tsqrt :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tsqrtDouble"
  tsqrtDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tsin"
  tsin :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tsinDouble"
  tsinDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tcos"
  tcos :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tcosDouble"
  tcosDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "ttan"
  ttan :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "ttanDouble"
  ttanDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tasin"
  tasin :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tasinDouble"
  tasinDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tacos"
  tacos :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tacosDouble"
  tacosDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tatan"
  tatan :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tatanDouble"
  tatanDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tsinh"
  tsinh :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tsinhDouble"
  tsinhDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tcosh"
  tcosh :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tcoshDouble"
  tcoshDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "ttanh"
  ttanh :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "ttanhDouble"
  ttanhDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tasinh"
  tasinh :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tasinhDouble"
  tasinhDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tacosh"
  tacosh :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tacoshDouble"
  tacoshDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tatanh"
  tatanh :: DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tatanhDouble"
  tatanhDouble :: DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tpow"
  tpow :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tpowDouble"
  tpowDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "tmax"
  tmax :: DevicePtr CFloat -> DevicePtr CFloat -> CSize -> DeviceM d ()

foreign import ccall "tmaxDouble"
  tmaxDouble :: DevicePtr CDouble -> DevicePtr CDouble -> CSize -> DeviceM d ()

foreign import ccall "broadcast_copy"
  broadcast_copy :: CInt -> CSize -> DevicePtr CFloat -> DevicePtr CInt -> DevicePtr CFloat -> DevicePtr CInt -> DeviceM d ()

foreign import ccall "broadcast_copyDouble"
  broadcast_copyDouble :: CInt -> CSize -> DevicePtr CDouble -> DevicePtr CInt -> DevicePtr CDouble -> DevicePtr CInt -> DeviceM d ()

foreign import ccall "&freeDevicePtr"
  freeDevicePtr :: FunPtr (Ptr a -> IO ())

foreign import ccall "&freeCuRANDGenerator"
  freeCuRANDGenerator :: FunPtr (Ptr () -> IO ())
