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
  , unsafeReshape
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
  , tsplitAt
  , elementwiseMax
  , flatten
  ) where
import Foreign
import Foreign.C
import Data.Proxy
import Control.Applicative
import Control.Monad.Except
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
import DeepBanana.Tensor.Mutable (MTensor)
import DeepBanana.Tensor.Shape
import DeepBanana.Tensor.TensorScalar
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.Cublas as CuBlas

data Tensor (n :: Nat) a = Tensor {
    shape :: Dim n
  , dataptr :: ForeignPtr a
  }

instance (Shape (Dim n), TensorScalar a, Eq a) => Eq (Tensor n a) where
  t1 == t2 = toList t1 == toList t2 && shape t1 == shape t2

data STensor n a = STensor (Dim n) [a]
               deriving Generic

instance forall n a
         . (Shape (Dim n), Generic a, TensorScalar a)
         => Generic (Tensor n a) where
  type Rep (Tensor n a) = Rep (STensor n a)
  from t = from (STensor (shape t) (toList t) :: STensor n a)
  to rep = let STensor shp listData = to rep :: STensor n a in
            fromList shp listData

instance (Shape (Dim n), Serialize a, Generic a, TensorScalar a)
         => Serialize (Tensor n a)

instance (Shape (Dim n), NFData a, Generic a, TensorScalar a) => NFData (Tensor n a)

instance (Shape (Dim n), TensorScalar a, Show a) => Show (Tensor n a) where
  show t = "Tensor " ++ show  (shape t)
           ++ " "  ++ show (take 10 $ toList t)

-- | Reshaping.
reshape :: (Shape (Dim n1), Shape (Dim n2), MonadError String m)
        => Dim n2 -> Tensor n1 a -> m (Tensor n2 a)
reshape newshp (Tensor oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwError $
    "Couldn't reshape tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ Tensor newshp dataptr

unsafeReshape :: (Shape (Dim n1), Shape (Dim n2))
              => Dim n2 -> Tensor n1 a -> Tensor n2 a
unsafeReshape newshp t = case reshape newshp t of
  Left err -> error err
  Right res -> res

-- | Returns the CuDNN datatype of a tensor.
dtype :: forall n a . (TensorScalar a) => Tensor n a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

-- | Converts a mutable tensor into an immutable one in O(1) time. The data is not
-- copied, so one should make sure the input mutable tensor is not modified after the
-- the conversion. Unsafe.
unsafeFreeze :: (PrimMonad m) => MTensor (PrimState m) n a -> m (Tensor n a)
unsafeFreeze (MT.MTensor shp ptr) = return $ Tensor shp ptr

-- | Converts an immutable tensor into a mutable one in O(1) time. The data is not
-- copied, so one should make sure the input immutable tensor is not used after the
-- conversion. Unsafe.
unsafeThaw :: (PrimMonad m) => Tensor n a -> m (MTensor (PrimState m) n a)
unsafeThaw (Tensor shp ptr) = return $ MT.MTensor shp ptr

-- | Initializes a tensor with all zeros.
zeros :: (Shape (Dim n), TensorScalar a) => Dim n -> Tensor n a
zeros shp = unsafePerformIO $ MT.zeros shp >>= unsafeFreeze

-- | Initializes a tensor with all ones.
ones :: (Shape (Dim n), TensorScalar a) => Dim n -> Tensor n a
ones shp = unsafePerformIO $ MT.ones shp >>= unsafeFreeze

-- | Initializes a tensor from list data. If the length of the list does not correspond
-- to the size of the desired shape, throws an exception at run time.
fromList :: (Shape (Dim n), TensorScalar a) => Dim n -> [a] -> Tensor n a
fromList shape value = unsafePerformIO $ MT.fromList shape value >>= unsafeFreeze

-- | Converts a tensor to a list.
toList :: (Shape (Dim n), TensorScalar a) => Tensor n a -> [a]
toList t = unsafePerformIO $ unsafeThaw t >>= MT.toList

-- | Converts a storable vector to an immutable tensor. Throws an error at runtime
-- when the input's length does not correspond to the desired output size. Runs in
-- O(n) time.
fromVector :: forall m n a
           . (MonadError String m, Shape (Dim n), TensorScalar a)
           => Dim n -> SV.Vector a -> m (Tensor n a)
fromVector shp value = do
  when (SV.length value /= size shp) $ throwError $
    "fromVector: Incompatible sizes\n\tinput vector: "
    ++ show (SV.length value) ++ "\n\trequired output shape: "
    ++ show shp
    ++ ", size " ++ show (size shp)
  return $ unsafePerformIO $ do
    res <- MT.emptyTensor shp
    SV.unsafeWith value $ \vptr -> do
      MT.withDevicePtr res $ \resptr -> do
        CUDA.pokeArray (size shp) vptr resptr
    unsafeFreeze res

-- | Converts a tensor to a storable vector. Runs in O(n) time.
toVector :: (Shape (Dim n), TensorScalar a) => Tensor n a -> SV.Vector a
toVector t = unsafePerformIO $ do
  res <- SMV.new $ size $ shape t
  mt <- unsafeThaw t
  SMV.unsafeWith res $ \resptr -> do
    MT.withDevicePtr mt $ \mtptr -> do
      CUDA.peekArray (size $ shape t) mtptr resptr
  SV.unsafeFreeze res

-- | Concatenates two 1-dimensional tensors. Inverse of tsplit. Runs in O(n+m) time.
tconcat :: forall a . (TensorScalar a)
        => Tensor 1 a -> Tensor 1 a -> Tensor 1 a
tconcat t1 t2 = unsafePerformIO $ do
  let t1sz = size $ shape t1
      t2sz = size $ shape t2
  out <- MT.emptyTensor $ (t1sz + t2sz) :. Z
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
tsplitAt :: forall m a . (MonadError String m, TensorScalar a)
         => Int -> Tensor 1 a -> m (Tensor 1 a, Tensor 1 a)
tsplitAt t1sz t = do
  let t2sz = size (shape t) - t1sz
  when (t2sz < 0) $ throwError $
    "Couldn't split the a vector of size " ++ show (size (shape t)) ++ " at point " ++ show t1sz ++ " : the input vector is too small."
  return $ unsafePerformIO $ do
    mt1 <- MT.emptyTensor $ t1sz :. Z
    mt2 <- MT.emptyTensor $ t2sz :. Z
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
flatten :: (Shape (Dim n)) => Tensor n a -> Tensor 1 a
flatten t = case reshape (size (shape t) :. Z) t of
  Left err -> error $ "Couldn't flatten a tensor. This shouldn't happen.\n" ++ err
  Right res -> res

-- | Type-safe broadcasting.
broadcast :: (Shape (Dim n), MonadError String m, TensorScalar a)
          => Dim n -> Tensor n a -> m (Tensor n a)
broadcast outshp inp = do
  when (not $ broadcastable (shape inp) outshp) $ throwError $
    "Couldn't broadcast shape " ++ show (shape inp) ++ " to shape " ++ show outshp
  return $ unsafePerformIO $ do
    let inp_shape =
          fmap fromIntegral
          $ take (nbdim outshp - nbdim (shape inp)) (repeat 1)
          ++ dimensions (shape inp)
        out_shape = fmap fromIntegral $ dimensions outshp
        inp_size = fromIntegral $ size $ shape inp
        out_size = fromIntegral $ size outshp
        out_nbdim = fromIntegral $ nbdim outshp
    -- We avoid copying when the size doesn't change.
    if out_size == inp_size
      then return $ unsafeReshape outshp inp
      else do
      minp <- unsafeThaw inp
      mout <- MT.emptyTensor outshp
      MT.withDevicePtr minp $ \inpptr -> do
        MT.withDevicePtr mout $ \outptr -> do
          CUDA.withListArray inp_shape $ \inp_shapeptr -> do
            CUDA.withListArray out_shape $ \out_shapeptr -> do
              broadcast_copy out_nbdim out_size inpptr inp_shapeptr outptr
                out_shapeptr
      unsafeFreeze mout

bin_broadcast :: (MonadError String m, Shape (Dim n), TensorScalar a)
              => Tensor n a -> Tensor n a -> m (Tensor n a, Tensor n a)
bin_broadcast t1 t2 = do
  case (broadcastable (shape t1) (shape t2), broadcastable (shape t2) (shape t1)) of
   (False,False) -> throwError $ "Incompatible shapes for broadcasting: " ++ show (shape t1) ++ " and " ++ show (shape t2)
   (True,False) -> do
     t1' <- broadcast (shape t2) t1
     return (t1', t2)
   (False,True) -> do
     t2' <- broadcast (shape t1) t2
     return (t1, t2')
   (True,True) -> return (t1,t2) -- in that case the shapes must be equal.

bin_broadcast_err :: (Shape (Dim n), TensorScalar a)
                  => Tensor n a -> Tensor n a -> (Tensor n a, Tensor n a)
bin_broadcast_err t1 t2 = case bin_broadcast t1 t2 of
  Left msg -> error msg
  Right res -> res

instance forall a n . (Shape (Dim n), TensorScalar a)
         => Num (Tensor n a) where
  _t1 + _t2 = let (t1,t2) = bin_broadcast_err t1 t2 in unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawAdd t1ptr t3ptr $ fromIntegral $ size $ shape t1
      unsafeFreeze t3'
  _t1 * _t2 = let (t1,t2) = bin_broadcast_err t1 t2 in unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawMul t1ptr t3ptr $ fromIntegral $ size $ shape t1
      unsafeFreeze t3'
  _t1 - _t2 = let (t1,t2) = bin_broadcast_err t1 t2 in unsafePerformIO $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy t2'
      MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> do
          MT.rawSubtract t1ptr t3ptr $ fromIntegral $ size $ shape t1
      unsafeFreeze t3'
  negate t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    MT.withDevicePtr res $ \resptr -> do
      MT.rawNegate resptr $ fromIntegral $ size $ shape t
    unsafeFreeze res
  signum t = unsafePerformIO $ do
    res <- unsafeThaw t >>= MT.copy
    MT.withDevicePtr res $ \resptr -> do
      MT.rawSignum resptr $ fromIntegral $ size $ shape t
    unsafeFreeze res
  fromInteger i = fromInteger i *^ ones scalarShape

instance (Shape (Dim n), TensorScalar a) => Fractional (Tensor n a) where
  recip x = unsafePerformIO $ do
    res <- unsafeThaw x >>= MT.copy
    MT.inv res
    unsafeFreeze res
  fromRational r = fromInteger (numerator r) / fromInteger (denominator r)

-- | Computes the elementwise maximum of 2 tensors.
elementwiseMax :: (Shape (Dim n), MonadError String m, TensorScalar a)
               => Tensor n a -> Tensor n a -> m (Tensor n a)
elementwiseMax _x _y = do
  (x,y) <- bin_broadcast _x _y
  return $ unsafePerformIO $ do
    mx <- unsafeThaw x
    my <- unsafeThaw y >>= MT.copy
    MT.withDevicePtr mx $ \pmx -> do
      MT.withDevicePtr my $ \pmy -> do
        MT.rawMax pmx pmy (fromIntegral $ size $ shape x)
    unsafeFreeze my

fromRaw :: (Shape (Dim n), TensorScalar a)
        => (CUDA.DevicePtr a -> CSize -> IO ()) -> Tensor n a -> Tensor n a
fromRaw action x = unsafePerformIO $ do
  res <- unsafeThaw x >>= MT.copy
  MT.withDevicePtr res $ \resptr -> do
    action resptr $ fromIntegral $ size $ shape x
  unsafeFreeze res

instance (Shape (Dim n), TensorScalar a) => Floating (Tensor n a) where
  pi = pi *^ ones scalarShape
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
  _x**_y = let (x,y) = bin_broadcast_err _x _y in unsafePerformIO $ do
    mx <- unsafeThaw x
    my <- unsafeThaw y >>= MT.copy
    MT.withDevicePtr mx $ \pmx -> do
      MT.withDevicePtr my $ \pmy -> do
        MT.rawPow pmx pmy $ fromIntegral $ size $ shape x
    unsafeFreeze my

-- Vector space instance for Tensors.
instance (Shape (Dim n), TensorScalar a) => AdditiveGroup (Tensor n a) where
  zeroV = fromInteger 0
  t1 ^+^ t2 = t1 + t2
  negateV t = negate t

instance (Shape (Dim n), TensorScalar a) => VectorSpace (Tensor n a) where
  type Scalar (Tensor n a) = a
  x *^ t = unsafePerformIO $ do -- TODO: somehow use Cublas's scal instead
    res <- unsafeThaw t >>= MT.copy
    MT.withDevicePtr res $ \resptr -> do
      MT.rawScale x resptr $ fromIntegral $ size $ shape t
    unsafeFreeze res

instance (Shape (Dim n), TensorScalar a) => InnerSpace (Tensor n a) where
  t1 <.> t2 = foldr (+) 0 $ fmap (\(x1,x2) -> x1 * x2) $ zip (toList t1) (toList t2)
