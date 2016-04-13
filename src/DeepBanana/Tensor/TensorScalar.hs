{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-|
Defines datatype which can be used as scalar elements for a tensor. Also exports much
needed, although orphan instances for 'CFloat' and 'CDouble': 'Generic', 'NFData' and
'Serialize'.
-}
module DeepBanana.Tensor.TensorScalar (
    TensorScalar
  , CFloat
  , CDouble
  , mySizeOf
  , datatype
  , thresh
  , rawMul
  , rawAdd
  , rawAbs
  , rawSignum
  , rawSubtract
  , rawNegate
  , rawScale
  , rawLog
  , rawInv
  , rawExp
  , rawSqrt
  , rawSin
  , rawCos
  , rawTan
  , rawAsin
  , rawAcos
  , rawAtan
  , rawSinh
  , rawCosh
  , rawTanh
  , rawAsinh
  , rawAcosh
  , rawAtanh
  , rawPow
  , rawMax
  , broadcast_copy
  , generateUniform
  , generateNormal
  , generateLogNormal
  ) where

import qualified Data.Serialize as S

import DeepBanana.Device
import qualified DeepBanana.Device.CUDA as CUDA
import qualified DeepBanana.Device.CuDNN as CuDNN
import qualified DeepBanana.Device.CuRAND as CuRAND
import qualified DeepBanana.Device.Cubits as Cubits
import DeepBanana.Prelude

instance Generic CFloat where
  type Rep CFloat = Rep Float
  from cx = from (realToFrac cx :: Float)
  to rep = realToFrac (to rep :: Float)

instance Generic CDouble where
  type Rep CDouble = Rep Double
  from cx = from (realToFrac cx :: Double)
  to rep = realToFrac (to rep :: Double)

instance S.Serialize CFloat
instance S.Serialize CDouble

mySizeOf :: forall a . Storable a => Proxy a -> Int
mySizeOf _ = sizeOf (error "mySizeOf: token parameter that shouldn't have been evaluated" :: a)

-- | Type class for tensor scalars. Basically requires them to be storable, serializable,
-- deepseqable numbers that can be handled by CUDA, Cublas, CuDNN and Haskell. Basically
-- requires low-level numeric routines.
class (Floating a, Storable a, VectorSpace a, a ~ Scalar a, S.Serialize a, NFData a)
      => TensorScalar a where
  -- | The corresponding CuDNN datatype.
  datatype :: Proxy a -> CuDNN.DataType
  -- | Low-level in-place numeric operations.
  thresh :: CUDA.DevicePtr a -> CSize -> a -> CUDA.DevicePtr a -> DeviceM d ()
  rawMul :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAdd :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAbs :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawSignum :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawSubtract :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawNegate :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawScale :: a -> CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawLog :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawInv :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawExp :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawSqrt :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawSin :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawCos :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawTan :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAsin :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAcos :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAtan :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawSinh :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawCosh :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawTanh :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAsinh :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAcosh :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawAtanh :: CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawPow :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> DeviceM d ()
  rawMax :: CUDA.DevicePtr a -> CUDA.DevicePtr a -> CSize -> DeviceM d ()
  broadcast_copy :: CInt -> CSize -> CUDA.DevicePtr a -> CUDA.DevicePtr CInt -> CUDA.DevicePtr a -> CUDA.DevicePtr CInt -> DeviceM d ()
  -- | Low-level CuRAND bindings.
  generateUniform :: CuRAND.Generator d
                  -> CUDA.DevicePtr a
                  -> CSize
                  -> DeviceM d CuRAND.Status
  generateNormal :: CuRAND.Generator d
                 -> CUDA.DevicePtr a
                 -> CSize
                 -> a
                 -> a
                 -> DeviceM d CuRAND.Status
  generateLogNormal :: CuRAND.Generator d
                    -> CUDA.DevicePtr a
                    -> CSize
                    -> a
                    -> a
                    -> DeviceM d CuRAND.Status

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
