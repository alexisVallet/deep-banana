{-# LANGUAGE DeriveGeneric, UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-|
GPU multi-dimensional dense numeric arrays.
-}
module DeepBanana.Tensor (
    module DeepBanana.Tensor.Shape
  , module DeepBanana.Tensor.TensorScalar
  , module Data.VectorSpace
  , Tensor(..)
  -- * Basic manipulations
  , dtype
  , reshape
  , broadcast
  , zeros
  , ones
  -- * Converting from/to mutable tensors
  , unsafeFreeze
  , unsafeThaw
  -- * Converting from/to lists
  , fromList
  , toList
  -- * Converting from/to storable vectors
  , fromVector
  , toVector
  -- * Utilities
  , tconcat
  , tsplit
  , elementwiseMax
  , flatten
  ) where
import Foreign
import Foreign.C
import Data.Proxy
import Control.Applicative
import Control.Monad.Primitive
import System.IO.Unsafe
import Data.VectorSpace
import Data.AdditiveGroup
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV
import Data.Traversable
import GHC.Generics
import GHC.TypeLits
import Data.Serialize
import Control.Monad
import Control.DeepSeq
import Data.Ratio
import Unsafe.Coerce

import qualified DeepBanana.Tensor.Mutable as MT
import DeepBanana.Tensor.Mutable (MTensor(..))
import DeepBanana.Tensor.Shape
import DeepBanana.Tensor.TensorScalar
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as CuBlas

-- | An immutable, GPU tensor with a given shape 's', whose scalar type 'a' should be
-- an instance of 'TensorScalar'. Such tensors are instances of 'Num', 'Fractional' and
-- 'Floating', implementing all the numeric operations in an elementwise fashion:
--
-- >>> fromList [1,2,3] + fromList [3,4,5] :: Tensor '[3] CFloat
-- Tensor [3] [4.0,6.0,8.0]
--
-- Scalars are automatically broadcast to whatever shape is necessary:
--
-- >>> 5 * fromList [1,2,3] :: Tensor '[3] CFloat
-- Tensor [3] [5.0,10.0,15.0]
data Tensor (s :: [Nat]) a = Tensor (ForeignPtr a)

instance (Shape s, TensorScalar a, Eq a) => Eq (Tensor s a) where
  t1 == t2 = toList t1 == toList t2

data STensor a = STensor [a]
               deriving Generic

instance (Shape s, Generic a, TensorScalar a) => Generic (Tensor s a) where
  type Rep (Tensor s a) = Rep (STensor a)
  from t = from (STensor (toList t))
  to rep = let STensor listData = to rep in
            fromList listData

instance (Shape s, Serialize a, Generic a, TensorScalar a) => Serialize (Tensor s a)

instance (Shape s, NFData a, Generic a, TensorScalar a) => NFData (Tensor s a)

instance forall s a . (Shape s, TensorScalar a, Show a) => Show (Tensor s a) where
  show t = "Tensor " ++ show  (dimensions (Proxy :: Proxy s))
           ++ " "  ++ show (take 10 $ toList t)

-- | Type safe reshaping.
reshape :: (Shape s1, Shape s2, Size s1 ~ Size s2) => Tensor s1 a -> Tensor s2 a
reshape = unsafeCoerce

-- | Returns the CuDNN datatype of a tensor.
dtype :: forall s a . (TensorScalar a) => Tensor s a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

-- | Converts a mutable tensor into an immutable one in O(1) time. The data is not
-- copied, so one should make sure the input mutable tensor is not modified after the
-- the conversion. Unsafe.
unsafeFreeze :: (PrimMonad m) => MTensor (PrimState m) s a -> m (Tensor s a)
unsafeFreeze (MTensor ptr) = return $ Tensor ptr

-- | Converts an immutable tensor into a mutable one in O(1) time. The data is not
-- copied, so one should make sure the input immutable tensor is not used after the
-- conversion. Unsafe.
unsafeThaw :: (PrimMonad m) => Tensor s a -> m (MTensor (PrimState m) s a)
unsafeThaw (Tensor ptr) = return $ MTensor ptr

-- | Initializes a tensor with all zeros.
zeros :: forall s a . (Shape s, TensorScalar a) => Tensor s a
zeros = unsafePerformIO $ MT.zeros >>= unsafeFreeze

-- | Initializes a tensor with all ones.
ones :: forall s a . (Shape s, TensorScalar a) => Tensor s a
ones = unsafePerformIO $ MT.ones >>= unsafeFreeze

-- | Initializes a tensor from list data. If the length of the list does not correspond
-- to the size of the desired shape, throws an exception at run time.
fromList :: (TensorScalar a, Shape s) => [a] -> Tensor s a
fromList value = unsafePerformIO $ MT.fromList value >>= unsafeFreeze

-- | Converts a tensor to a list.
toList :: (TensorScalar a, Shape s) => Tensor s a -> [a]
toList t = unsafePerformIO $ unsafeThaw t >>= MT.toList

-- | Converts a storable vector to an immutable tensor. Throws an error at runtime
-- when the input's length does not correspond to the desired output size. Runs in
-- O(n) time.
fromVector :: forall s a . (TensorScalar a, Shape s) => SV.Vector a -> Tensor s a
fromVector value = unsafePerformIO $ do
  when (SV.length value /= size (Proxy :: Proxy s))
    $ error $ "fromVector: Incompatible sizes\n\tinput vector: "
    ++ show (SV.length value) ++ "\n\trequired output shape: "
    ++ show (dimensions (Proxy :: Proxy s))
    ++ ", size " ++ show (size (Proxy :: Proxy s))
  res <- MT.emptyTensor
  SV.unsafeWith value $ \vptr -> do
    MT.withDevicePtr res $ \resptr -> do
      CUDA.pokeArray (size (Proxy :: Proxy s)) vptr resptr
  unsafeFreeze res

-- | Converts a tensor to a storable vector. Runs in O(n) time.
toVector :: forall a s . (TensorScalar a, Shape s) => Tensor s a -> SV.Vector a
toVector t = unsafePerformIO $ do
  res <- SMV.new $ size (Proxy :: Proxy s)
  mt <- unsafeThaw t
  SMV.unsafeWith res $ \resptr -> do
    MT.withDevicePtr mt $ \mtptr -> do
      CUDA.peekArray (size (Proxy :: Proxy s)) mtptr resptr
  SV.unsafeFreeze res

-- | Concatenates two 1-dimensional tensors. Inverse of tsplit. Runs in O(n+m) time.
tconcat :: forall n m a . (KnownNat n, KnownNat m, KnownNat (n + m), TensorScalar a)
       => Tensor '[n] a -> Tensor '[m] a -> Tensor '[n + m] a
tconcat t1 t2 = unsafePerformIO $ do
  let t1sz = fromIntegral $ natVal (Proxy :: Proxy n)
      t2sz = fromIntegral $ natVal (Proxy :: Proxy m)
  out <- MT.emptyTensor
  mt1 <- unsafeThaw t1
  mt2 <- unsafeThaw t2
  MT.withDevicePtr mt1 $ \t1ptr -> do
    MT.withDevicePtr mt2 $ \t2ptr -> do
      MT.withDevicePtr out $ \outptr -> do
        CUDA.copyArray t1sz t1ptr outptr
        CUDA.copyArray t2sz t2ptr
          (CUDA.DevicePtr
           $ plusPtr
           (CUDA.useDevicePtr outptr)
           (sizeOf (undefined :: a) * fromIntegral t1sz))
  unsafeFreeze out

-- | Splits a 1-dimensional tensor into 2 parts. Inverse of tconcat. Runs in O(n+m) time.
tsplit :: forall n m a . (KnownNat n, KnownNat m, TensorScalar a)
      => Tensor '[n + m] a -> (Tensor '[n] a, Tensor '[m] a)
tsplit t = unsafePerformIO $ do
  let t1sz = fromIntegral $ natVal (Proxy :: Proxy n)
      t2sz = fromIntegral $ natVal (Proxy :: Proxy m)
  mt1 <- MT.emptyTensor
  mt2 <- MT.emptyTensor
  mt <- unsafeThaw t
  MT.withDevicePtr mt1 $ \t1ptr -> do
    MT.withDevicePtr mt2 $ \t2ptr -> do
      MT.withDevicePtr mt $ \tptr -> do
        CUDA.copyArray t1sz tptr t1ptr
        let offsetPtr =
              CUDA.DevicePtr
              $ plusPtr
              (CUDA.useDevicePtr tptr)
              (sizeOf (undefined :: a) * fromIntegral t1sz)
        CUDA.copyArray t2sz offsetPtr t2ptr
  pure (,) <*> unsafeFreeze mt1 <*> unsafeFreeze mt2

-- | Flattens a tensor into a 1-dimensional vector.
flatten :: (Shape s, 1 <= Size s, KnownNat (Size s)) => Tensor s a -> Tensor '[Size s] a
flatten = reshape

-- | Type-safe broadcasting.
broadcast :: forall s1 s2 a
          . (TensorScalar a, Shape s1, Shape s2, Broadcast s1 s2)
          => Tensor s1 a -> Tensor s2 a
broadcast inp = unsafePerformIO $ do
  let inp_shape =
        fmap fromIntegral
        $ take (nbdim (Proxy :: Proxy s2) - nbdim (Proxy :: Proxy s1)) (repeat 1)
        ++ dimensions (Proxy :: Proxy s1)
      out_shape = fmap fromIntegral $ dimensions (Proxy :: Proxy s2)
      out_size = fromIntegral $ size (Proxy :: Proxy s2)
      out_nbdim = fromIntegral $ nbdim (Proxy :: Proxy s2)
  minp <- unsafeThaw inp
  mout <- MT.emptyTensor
  MT.withDevicePtr minp $ \inpptr -> do
    MT.withDevicePtr mout $ \outptr -> do
      CUDA.withListArray inp_shape $ \inp_shapeptr -> do
        CUDA.withListArray out_shape $ \out_shapeptr -> do
          broadcast_copy out_nbdim out_size inpptr inp_shapeptr outptr out_shapeptr
  unsafeFreeze mout

instance forall a s . (TensorScalar a, Shape s) => Num (Tensor s a) where
  t1 + t2 = unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawAdd t1ptr t3ptr $ fromIntegral $ size (Proxy :: Proxy s)
      unsafeFreeze t3'
  t1 * t2 = unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawMul t1ptr t3ptr $ fromIntegral $ size (Proxy :: Proxy s)
      unsafeFreeze t3'
  t1 - t2 = unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawSubtract t1ptr t3ptr $ fromIntegral $ size (Proxy :: Proxy s)
      unsafeFreeze t3'
  negate t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    MT.withDevicePtr res $ \resptr -> do
      MT.rawNegate resptr $ fromIntegral $ size (Proxy :: Proxy s)
    unsafeFreeze res
  signum t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    MT.withDevicePtr res $ \resptr -> do
      MT.rawSignum resptr $ fromIntegral $ size (Proxy :: Proxy s)
    unsafeFreeze res
  fromInteger i = fromInteger i *^ ones

instance (Shape s, TensorScalar a) => Fractional (Tensor s a) where
  recip x = unsafePerformIO $ do
    res <- unsafeThaw x >>= MT.copy
    MT.inv res
    unsafeFreeze res
  fromRational r = fromInteger (numerator r) / fromInteger (denominator r)

-- | Computes the elementwise maximum of 2 tensors.
elementwiseMax :: forall s a . (Shape s, TensorScalar a)
               => Tensor s a -> Tensor s a -> Tensor s a
elementwiseMax x y = unsafePerformIO $ do
  mx <- unsafeThaw x
  my <- unsafeThaw y >>= MT.copy
  MT.withDevicePtr mx $ \pmx -> do
    MT.withDevicePtr my $ \pmy -> do
      MT.rawMax pmx pmy (fromIntegral $ size (Proxy :: Proxy s))
  unsafeFreeze my

fromRaw :: forall s a . (Shape s, TensorScalar a)
        => (CUDA.DevicePtr a -> CSize -> IO ()) -> Tensor s a -> Tensor s a
fromRaw action x = unsafePerformIO $ do
  res <- unsafeThaw x >>= MT.copy
  MT.withDevicePtr res $ \resptr -> do
    action resptr $ fromIntegral $ size (Proxy :: Proxy s)
  unsafeFreeze res

instance (Shape s, TensorScalar a) => Floating (Tensor s a) where
  pi = pi *^ ones
  exp = fromRaw MT.rawExp
  log = fromRaw MT.rawLog
  sqrt = fromRaw MT.rawSqrt
  sin = fromRaw MT.rawSin
  cos = fromRaw MT.rawCos
  tan = fromRaw MT.rawTan
  asin = fromRaw MT.rawAsin
  acos = fromRaw MT.rawAcos
  atan = fromRaw MT.rawAtan
  sinh = fromRaw MT.rawSinh
  cosh = fromRaw MT.rawCosh
  tanh = fromRaw MT.rawTanh
  asinh = fromRaw MT.rawAsinh
  acosh = fromRaw MT.rawAcosh
  atanh = fromRaw MT.rawAtanh
  x**y = unsafePerformIO $ do
    mx <- unsafeThaw x
    my <- unsafeThaw y >>= MT.copy
    MT.withDevicePtr mx $ \pmx -> do
      MT.withDevicePtr my $ \pmy -> do
        MT.rawPow pmx pmy $ fromIntegral $ size (Proxy :: Proxy s)
    unsafeFreeze my

-- Vector space instance for Tensors.
instance (Shape s, TensorScalar a) => AdditiveGroup (Tensor s a) where
  zeroV = fromInteger 0
  t1 ^+^ t2 = t1 + t2
  negateV t = negate t

instance (Shape s, TensorScalar a) => VectorSpace (Tensor s a) where
  type Scalar (Tensor s a) = a
  x *^ t = unsafePerformIO $ do -- TODO: somehow use Cublas's scal instead
    res <- unsafeThaw t >>= MT.copy
    MT.withDevicePtr res $ \resptr -> do
      MT.rawScale x resptr $ fromIntegral $ size (Proxy :: Proxy s)
    unsafeFreeze res

instance (Shape s, TensorScalar a) => InnerSpace (Tensor s a) where
  t1 <.> t2 = foldr (+) 0 $ fmap (\(x1,x2) -> x1 * x2) $ zip (toList t1) (toList t2)
