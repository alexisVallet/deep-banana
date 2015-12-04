{-# LANGUAGE DeriveGeneric, UndecidableInstances #-}
module DeepBanana.Tensor (
    module DeepBanana.Tensor.Shape
  , module DeepBanana.Tensor.TensorScalar
  , module Data.VectorSpace
  , dtype
  , Tensor(..)
  , reshape
  , unsafeFreeze
  , unsafeThaw
  , fromList
  , toList
  , fromVector
  , toVector
  , elementwiseMax
  , tconcat
  , tsplit
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

data Tensor (s :: [Nat]) a = Tensor (ForeignPtr a)

data STensor a = STensor [a]
               deriving Generic

instance (Shape s, Generic a, TensorScalar a) => Generic (Tensor s a) where
  type Rep (Tensor s a) = Rep (STensor a)
  from t = from (STensor (toList t))
  to rep = let STensor listData = to rep in
            fromList listData

instance (Shape s, Serialize a, Generic a, TensorScalar a) => Serialize (Tensor s a)

instance (NFData a, Generic a, TensorScalar a) => NFData (Tensor s a)

instance forall s a . (Shape s, TensorScalar a, Show a) => Show (Tensor s a) where
  show t = "Tensor " ++ show (shape (Proxy :: Proxy s)) ++ " "  ++ show (take 10 $ toList t)

reshape :: (Shape s1, Shape s2, Size s1 ~ Size s2) => Tensor s1 a -> Tensor s2 a
reshape = unsafeCoerce

dtype :: forall s a . (TensorScalar a) => Tensor s a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

-- mutable/unmutable conversions
unsafeFreeze :: (PrimMonad m) => MTensor (PrimState m) s a -> m (Tensor s a)
unsafeFreeze (MTensor ptr) = return $ Tensor ptr

unsafeThaw :: (PrimMonad m) => Tensor s a -> m (MTensor (PrimState m) s a)
unsafeThaw (Tensor ptr) = return $ MTensor ptr

zeros :: forall s a . (Shape s, TensorScalar a) => Tensor s a
zeros = unsafePerformIO $ MT.zeros >>= unsafeFreeze

ones :: forall s a . (Shape s, TensorScalar a) => Tensor s a
ones = unsafePerformIO $ MT.ones >>= unsafeFreeze

-- conversion to/from lists
fromList :: (TensorScalar a, Shape s) => [a] -> Tensor s a
fromList value = unsafePerformIO $ MT.fromList value >>= unsafeFreeze

toList :: (TensorScalar a, Shape s) => Tensor s a -> [a]
toList t = unsafePerformIO $ unsafeThaw t >>= MT.toList

-- conversion to/from storable based vectors
fromVector :: forall s a . (TensorScalar a, Shape s) => SV.Vector a -> Tensor s a
fromVector value = unsafePerformIO $ do
  res <- MT.emptyTensor
  SV.unsafeWith value $ \vptr -> do
    MT.withDevicePtr res $ \resptr -> do
      CUDA.pokeArray (size (Proxy :: Proxy s)) vptr resptr
  unsafeFreeze res

toVector :: forall a s . (TensorScalar a, Shape s) => Tensor s a -> SV.Vector a
toVector t = unsafePerformIO $ do
  res <- SMV.new $ size (Proxy :: Proxy s)
  mt <- unsafeThaw t
  SMV.unsafeWith res $ \resptr -> do
    MT.withDevicePtr mt $ \mtptr -> do
      CUDA.peekArray (size (Proxy :: Proxy s)) mtptr resptr
  SV.unsafeFreeze res

-- Vector concatenation and splitting
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