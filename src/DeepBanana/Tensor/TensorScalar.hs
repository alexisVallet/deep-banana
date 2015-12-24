{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-|
Defines datatype which can be used as scalar elements for a tensor. Also exports much
needed, although orphan instances for 'CFloat' and 'CDouble': 'Generic', 'NFData' and
'Serialize'.
-}
module DeepBanana.Tensor.TensorScalar (
    TensorScalar(..)
  , CFloat, CDouble
  ) where

import Foreign.C
import Foreign.Storable
import Data.Serialize
import Control.DeepSeq
import GHC.Generics
import Data.VectorSpace
import Data.Proxy
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuRAND as CuRAND
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuDNN as CuDNN

import qualified DeepBanana.Cubits as Cubits

instance Generic CFloat where
  type Rep CFloat = Rep Float
  from cx = from (realToFrac cx :: Float)
  to rep = realToFrac (to rep :: Float)

instance Generic CDouble where
  type Rep CDouble = Rep Double
  from cx = from (realToFrac cx :: Double)
  to rep = realToFrac (to rep :: Double)

instance Serialize CFloat
instance Serialize CDouble

-- | Type class for tensor scalars. Basically requires them to be storable, serializable,
-- deepseqable numbers that can be handled by CUDA, Cublas, CuDNN and Haskell. Basically
-- requires low-level numeric routines.
class (Cublas.Cublas a, Floating a, Storable a, VectorSpace a, a ~ Scalar a, Serialize a, NFData a)
      => TensorScalar a where
  -- | The corresponding CuDNN datatype.
  datatype :: Proxy a -> CuDNN.DataType
  -- | Low-level in-place numeric operations.
  thresh :: CUDA.DevicePtr a -> CSize -> a -> CUDA.DevicePtr a -> IO ()
  rawMul :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawAdd :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawAbs :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSignum :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSubtract :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawNegate :: CUDA.DevicePtr a -> CSize -> IO ()
  rawScale :: a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawLog :: CUDA.DevicePtr a -> CSize -> IO ()
  rawInv :: CUDA.DevicePtr a -> CSize -> IO ()
  rawExp :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSqrt :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSin :: CUDA.DevicePtr a -> CSize -> IO ()
  rawCos :: CUDA.DevicePtr a -> CSize -> IO ()
  rawTan :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAsin :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAcos :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAtan :: CUDA.DevicePtr a -> CSize -> IO ()
  rawSinh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawCosh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawTanh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAsinh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAcosh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawAtanh :: CUDA.DevicePtr a -> CSize -> IO ()
  rawPow :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  rawMax :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> IO ()
  broadcast_copy :: CInt -> CSize -> CUDA.DevicePtr a -> CUDA.DevicePtr CInt -> CUDA.DevicePtr a -> CUDA.DevicePtr CInt -> IO ()
  -- | Low-level CuRAND bindings.
  generateUniform :: CuRAND.Generator
                  -> CUDA.DevicePtr a
                  -> CSize
                  -> IO CuRAND.Status
  generateNormal :: CuRAND.Generator
                 -> CUDA.DevicePtr a
                 -> CSize
                 -> a
                 -> a
                 -> IO CuRAND.Status
  generateLogNormal :: CuRAND.Generator
                    -> CUDA.DevicePtr a
                    -> CSize
                    -> a
                    -> a
                    -> IO CuRAND.Status

instance TensorScalar CFloat where
  datatype = const CuDNN.float
  thresh = Cubits.thresh
  rawMul = Cubits.mul
  rawAdd = Cubits.add
  rawAbs = Cubits.tabs
  rawSignum = Cubits.tsignum
  rawSubtract = Cubits.subtract
  rawNegate = Cubits.tnegate
  rawScale = Cubits.scale
  rawLog = Cubits.logFloat
  rawInv = Cubits.inv
  rawExp = Cubits.texp
  rawSqrt = Cubits.tsqrt
  rawSin = Cubits.tsin
  rawCos = Cubits.tcos
  rawTan = Cubits.ttan
  rawAsin = Cubits.tasin
  rawAcos = Cubits.tacos
  rawAtan = Cubits.tatan
  rawSinh = Cubits.tsinh
  rawCosh = Cubits.tcosh
  rawTanh = Cubits.ttanh
  rawAsinh = Cubits.tasinh
  rawAcosh = Cubits.tacosh
  rawAtanh = Cubits.tatanh
  rawPow = Cubits.tpow
  rawMax = Cubits.tmax
  broadcast_copy = Cubits.broadcast_copy
  generateUniform = CuRAND.generateUniform
  generateNormal = CuRAND.generateNormal
  generateLogNormal = CuRAND.generateLogNormal

instance TensorScalar CDouble where
  datatype = const CuDNN.double
  thresh = Cubits.threshDouble
  rawMul = Cubits.mulDouble
  rawAdd = Cubits.addDouble
  rawAbs = Cubits.tabsDouble
  rawSignum = Cubits.tsignumDouble
  rawSubtract = Cubits.subtractDouble
  rawNegate = Cubits.tnegateDouble
  rawScale = Cubits.scaleDouble
  rawLog = Cubits.logDouble
  rawInv = Cubits.invDouble
  rawExp = Cubits.texpDouble
  rawSqrt = Cubits.tsqrtDouble
  rawSin = Cubits.tsinDouble
  rawCos = Cubits.tcosDouble
  rawTan = Cubits.ttanDouble
  rawAsin = Cubits.tasinDouble
  rawAcos = Cubits.tacosDouble
  rawAtan = Cubits.tatanDouble
  rawSinh = Cubits.tsinhDouble
  rawCosh = Cubits.tcoshDouble
  rawTanh = Cubits.ttanhDouble
  rawAsinh = Cubits.tasinhDouble
  rawAcosh = Cubits.tacoshDouble
  rawAtanh = Cubits.tatanhDouble
  rawPow = Cubits.tpowDouble
  rawMax = Cubits.tmaxDouble
  broadcast_copy = Cubits.broadcast_copyDouble
  generateUniform = CuRAND.generateUniformDouble
  generateNormal = CuRAND.generateNormalDouble
  generateLogNormal = CuRAND.generateLogNormalDouble
