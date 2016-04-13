{-# LANGUAGE DeriveGeneric, UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-|
GPU multi-dimensional dense numeric arrays.
-}
module DeepBanana.Tensor (
    module DeepBanana.Tensor.Shape
  , module DeepBanana.Tensor.Exception
  , module DeepBanana.Tensor.TensorScalar
  , Tensor(..)
  -- * Basic manipulations
  , dtype
  , reshape
  , reshape'
  , transfer
  , transfer'
  , broadcast
  , broadcast'
  , zeros
  , zeros'
  , ones
  , ones'
  -- * Converting from/to mutable tensors
  , unsafeFreeze
  , unsafeThaw
  -- * Converting from/to lists
  , tensorFromList
  , tensorFromList'
  , tensorToList
  -- * Converting from/to storable vectors
  , fromVector
  , fromVector'
  , toVector
  -- * Utilities
  , tconcat
  , tconcat'
  , tsplitAt
  , tsplitAt'
  , elementwiseMax
  , flatten
  ) where

import Data.Serialize (Serialize)
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as SMV
import System.IO.Unsafe
import Unsafe.Coerce

import DeepBanana.Device
import qualified DeepBanana.Device.CuDNN as CuDNN
import qualified DeepBanana.Device.CUDA as CUDA
import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Tensor.Exception
import qualified DeepBanana.Tensor.Mutable as MT
import DeepBanana.Tensor.Mutable (MTensor)
import DeepBanana.Tensor.Shape
import DeepBanana.Tensor.TensorScalar

data Tensor d (n :: Nat) a = Tensor {
    shape :: Dim n
  , dataptr :: ForeignPtr a
  }

instance (Shape (Dim n), Device d, TensorScalar a, Eq a) => Eq (Tensor d n a) where
  t1 == t2 = tensorToList t1 == tensorToList t2 && shape t1 == shape t2

data STensor d n a = STensor (Dim n) [a]
               deriving Generic

instance forall d n a
         . (Device d, Shape (Dim n), Generic a, TensorScalar a)
         => Generic (Tensor d n a) where
  type Rep (Tensor d n a) = Rep (STensor d n a)
  from t = from (STensor (shape t) (tensorToList t) :: STensor d n a)
  to rep = let STensor shp listData = to rep :: STensor d n a in
            tensorFromList' shp listData

instance (Device d, Shape (Dim n), Serialize a, Generic a, TensorScalar a)
         => Serialize (Tensor d n a)

instance (Device d, Shape (Dim n), NFData a, Generic a, TensorScalar a)
         => NFData (Tensor d n a)

instance (Device d, Shape (Dim n), TensorScalar a, Show a)
         => Show (Tensor d n a) where
  show t = "Tensor " ++ show  (shape t)
           ++ " "  ++ show (take 10 $ tensorToList t)

-- | Reshaping.
reshape :: (Shape (Dim n1), Shape (Dim n2), TensorScalar a,
            MonadError t m, Variant t IncompatibleSize)
        => Dim n2 -> Tensor d n1 a -> m (Tensor d n2 a)
reshape newshp t@(Tensor oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwVariant $ incompatibleSize $
    "Couldn't reshape tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ Tensor newshp dataptr

reshape' :: forall d n1 n2 a
         . (Shape (Dim n1), Shape (Dim n2), TensorScalar a)
         => Dim n2 -> Tensor d n1 a -> Tensor d n2 a
reshape' newshp t =
  unsafeRunExcept (reshape newshp t :: Either IncompatibleSize (Tensor d n2 a))

transfer :: (MonadError t m, Variant t OutOfMemory, Shape (Dim n),
             Device d1, Device d2, TensorScalar a)
         => Tensor d1 n a -> m (Tensor d2 n a)
transfer t = embedExcept $ runST $ runExceptT $ do
  mt <- unsafeThaw t
  mt' <- MT.copy mt
  unsafeFreeze mt'

transfer' :: forall n d1 d2 a
          . (Shape (Dim n), Device d1, Device d2, TensorScalar a)
          => Tensor d1 n a -> Tensor d2 n a
transfer' t = unsafeRunExcept (transfer t :: Either OutOfMemory (Tensor d2 n a))

-- | Returns the CuDNN datatype of a tensor.
dtype :: forall d n a . (TensorScalar a) => Tensor d n a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

-- | Converts a mutable tensor into an immutable one in O(1) time. The data is not
-- copied, so one should make sure the input mutable tensor is not modified after the
-- the conversion. Unsafe.
unsafeFreeze :: (PrimMonad m) => MTensor (PrimState m) d n a -> m (Tensor d n a)
unsafeFreeze (MT.MTensor shp ptr) = return $ Tensor shp ptr

-- | Converts an immutable tensor into a mutable one in O(1) time. The data is not
-- copied, so one should make sure the input immutable tensor is not used after the
-- conversion. Unsafe.
unsafeThaw :: (PrimMonad m) => Tensor d n a -> m (MTensor (PrimState m) d n a)
unsafeThaw (Tensor shp ptr) = return $ MT.MTensor shp ptr

-- | Initializes a tensor with all zeros.
zeros :: (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
          Variant t OutOfMemory)
      => Dim n -> m (Tensor d n a)
zeros shp = do
  embedExcept $ runST $ runExceptT $ MT.zeros shp >>= unsafeFreeze

zeros' :: forall d n a
       . (Device d, Shape (Dim n), TensorScalar a)
       => Dim n -> Tensor d n a
zeros' shp = unsafeRunExcept (zeros shp :: Either OutOfMemory (Tensor d n a))

-- | Initializes a tensor with all ones.
ones :: (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
         Variant t OutOfMemory)
     => Dim n -> m (Tensor d n a)
ones shp = do
  embedExcept $ runST $ runExceptT $ MT.ones shp >>= unsafeFreeze

ones' :: forall d n a
      . (Device d, Shape (Dim n),  TensorScalar a)
      => Dim n -> Tensor d n a
ones' shp = unsafeRunExcept (ones shp :: Either OutOfMemory (Tensor d n a))

-- | Initializes a tensor from list data. If the length of the list does not correspond
-- to the size of the desired shape, throws an exception at run time.
tensorFromList :: (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
                   Variant t IncompatibleSize, Variant t OutOfMemory)
               => Dim n -> [a] -> m (Tensor d n a)
tensorFromList shape value = do
  embedExcept $ runST $ runExceptT $ MT.tensorFromList shape value >>= unsafeFreeze

tensorFromList' :: forall d n a
                . (Device d, Shape (Dim n), TensorScalar a)
                => Dim n -> [a] -> Tensor d n a
tensorFromList' shape value =
  unsafeRunExcept (tensorFromList shape value
                   :: Either (Coproduct '[IncompatibleSize, OutOfMemory]) (Tensor d n a))

-- | Converts a tensor to a list.
tensorToList :: (Shape (Dim n), Device d, TensorScalar a) => Tensor d n a -> [a]
tensorToList t = runST $ unsafeThaw t >>= MT.tensorToList

-- | Converts a storable vector to an immutable tensor. Throws an error at runtime
-- when the input's length does not correspond to the desired output size. Runs in
-- O(n) time.
fromVector :: forall t m d n a
           . (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
              Variant t OutOfMemory, Variant t IncompatibleSize)
           => Dim n -> SVector a -> m (Tensor d n a)
fromVector shp value = do
  when (length value /= size shp) $ throwVariant $ incompatibleSize $
    "fromVector: Incompatible sizes\n\tinput vector: "
    ++ show (length value) ++ "\n\trequired output shape: "
    ++ show shp
    ++ ", size " ++ show (size shp)
  embedExcept $ unsafePerformIO $ runExceptT $ do
    res <- MT.emptyTensor shp
    liftIO $ SV.unsafeWith value $ \vptr -> do
      MT.withDevicePtr res $ \resptr -> do
        runDeviceM (Proxy :: Proxy d) $ CUDA.pokeArray (size shp) vptr resptr
    unsafeFreeze res

fromVector' :: forall d n a
            . (Device d, Shape (Dim n), TensorScalar a)
            => Dim n -> SVector a -> Tensor d n a
fromVector' shp value =
  unsafeRunExcept (fromVector shp value
                   :: Either (Coproduct '[OutOfMemory, IncompatibleSize]) (Tensor d n a))

-- | Converts a tensor to a storable vector. Runs in O(n) time.
toVector :: forall d n a
         . (Device d, Shape (Dim n), TensorScalar a) => Tensor d n a -> SVector a
toVector t = unsafePerformIO $ do
  res <- SMV.new $ size $ shape t
  mt <- unsafeThaw t
  SMV.unsafeWith res $ \resptr -> do
    MT.withDevicePtr mt $ \mtptr -> do
      runDeviceM (Proxy :: Proxy d) $ CUDA.peekArray (size $ shape t) mtptr resptr
  SV.unsafeFreeze res

-- | Concatenates two 1-dimensional tensors. Inverse of tsplit. Runs in O(n+m) time.
tconcat :: forall t d m a
        . (Device d, TensorScalar a, MonadError t m, Variant t OutOfMemory)
        => Tensor d 1 a -> Tensor d 1 a -> m (Tensor d 1 a)
tconcat t1 t2 = do
  embedExcept $ unsafePerformIO $ runExceptT $ do
    let t1sz = size $ shape t1
        t2sz = size $ shape t2
    out <- MT.emptyTensor $ (t1sz + t2sz) :. Z
    mt1 <- unsafeThaw t1
    mt2 <- unsafeThaw t2
    liftIO $ MT.withDevicePtr mt1 $ \t1ptr -> do
      MT.withDevicePtr mt2 $ \t2ptr -> do
        MT.withDevicePtr out $ \outptr -> runDeviceM (Proxy :: Proxy d) $ do
          CUDA.copyArray t1sz t1ptr outptr
          CUDA.copyArray t2sz t2ptr
            (CUDA.DevicePtr
             $ plusPtr
             (CUDA.useDevicePtr outptr)
             (mySizeOf (Proxy :: Proxy a) * fromIntegral t1sz))
    unsafeFreeze out

tconcat' :: forall d a
         . (Device d, TensorScalar a)
         => Tensor d 1 a -> Tensor d 1 a -> Tensor d 1 a
tconcat' t1 t2 =
  unsafeRunExcept (tconcat t1 t2 :: Either OutOfMemory (Tensor d 1 a))

-- | Splits a 1-dimensional tensor into 2 parts. Inverse of tconcat. Runs in O(n+m) time.
tsplitAt :: forall t d m a
         . (Device d, TensorScalar a, MonadError t m, Variant t OutOfMemory,
            Variant t IncompatibleSize)
         => Int -> Tensor d 1 a -> m (Tensor d 1 a, Tensor d 1 a)
tsplitAt t1sz t = do
  let t2sz = size (shape t) - t1sz
  when (t2sz < 0) $ throwVariant $ incompatibleSize $
    "Couldn't split the a vector of size " ++ show (size (shape t)) ++ " at point " ++ show t1sz ++ " : the input vector is too small."
  embedExcept $ unsafePerformIO $ runExceptT $ do
    mt1 <- MT.emptyTensor $ t1sz :. Z
    mt2 <- MT.emptyTensor $ t2sz :. Z
    mt <- unsafeThaw t
    liftIO $ MT.withDevicePtr mt1 $ \t1ptr -> do
      MT.withDevicePtr mt2 $ \t2ptr -> do
        MT.withDevicePtr mt $ \tptr -> runDeviceM (Proxy :: Proxy d) $ do
          CUDA.copyArray t1sz tptr t1ptr
          let offsetPtr =
                CUDA.DevicePtr
                $ plusPtr
                (CUDA.useDevicePtr tptr)
                (mySizeOf (Proxy :: Proxy a) * fromIntegral t1sz)
          CUDA.copyArray t2sz offsetPtr t2ptr
    pure (,) <*> unsafeFreeze mt1 <*> unsafeFreeze mt2

tsplitAt' :: forall d a
          . (Device d, TensorScalar a)
          => Int -> Tensor d 1 a -> (Tensor d 1 a, Tensor d 1 a)
tsplitAt' t1sz t = unsafeRunExcept (tsplitAt t1sz t :: Either (Coproduct '[OutOfMemory,IncompatibleSize]) (Tensor d 1 a, Tensor d 1 a))

-- | Flattens a tensor into a 1-dimensional vector.
flatten :: forall n d a
        . (Shape (Dim n), TensorScalar a)
        => Tensor d n a -> Tensor d 1 a
flatten t =
  case reshape (size (shape t) :. Z) t :: Either IncompatibleSize (Tensor d 1 a) of
   Left err -> error $ "Couldn't flatten a tensor. This shouldn't happen.\n" ++ show err
   Right res -> res

-- | Type-safe broadcasting.
broadcast :: forall d t n m a
          . (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
             Variant t IncompatibleShape, Variant t OutOfMemory)
          => Dim n -> Tensor d n a -> m (Tensor d n a)
broadcast outshp inp = do
  when (not $ broadcastable (shape inp) outshp) $ throwVariant $ incompatibleShape $
    "Couldn't broadcast shape " ++ show (shape inp) ++ " to shape " ++ show outshp
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
    then return $ reshape' outshp inp
    else embedExcept $ unsafePerformIO $ runExceptT $ do
    minp <- unsafeThaw inp
    mout <- MT.emptyTensor outshp
    liftIO $ MT.withDevicePtr minp $ \inpptr -> do
      MT.withDevicePtr mout $ \outptr -> runDeviceM (Proxy :: Proxy d) $ do
        inp_shapeptr <- CUDA.newListArray inp_shape
        out_shapeptr <- CUDA.newListArray out_shape
        broadcast_copy out_nbdim out_size inpptr inp_shapeptr outptr out_shapeptr
        CUDA.free inp_shapeptr
        CUDA.free out_shapeptr
    unsafeFreeze mout

broadcast' :: forall n d a
           . (Device d, Shape (Dim n), TensorScalar a)
           => Dim n -> Tensor d n a -> Tensor d n a
broadcast' outshp inp =
  unsafeRunExcept (broadcast outshp inp
                   :: Either (Coproduct '[IncompatibleShape, OutOfMemory]) (Tensor d n a))

bin_broadcast :: forall n d t m a
              . (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
                 Variant t IncompatibleShape, Variant t OutOfMemory)
              => Tensor d n a -> Tensor d n a -> m (Tensor d n a, Tensor d n a)
bin_broadcast t1 t2 =
  case (broadcast (shape t2) t1, broadcast (shape t1) t2) of
   (Left err1, Left err2) -> throwError err1 >> throwError err2
   (Right t1', Left err) -> return (t1', t2)
   (Left err, Right t2') -> return (t1, t2')
   _ -> return (t1, t2)

bin_broadcast' :: forall d n a
               . (Device d, Shape (Dim n), TensorScalar a)
               => Tensor d n a -> Tensor d n a -> (Tensor d n a, Tensor d n a)
bin_broadcast' t1 t2 =
  unsafeRunExcept (bin_broadcast t1 t2
                   :: Either (Coproduct '[IncompatibleShape, OutOfMemory]) (Tensor d n a, Tensor d n a))

instance forall d n a . (Device d, Shape (Dim n), TensorScalar a)
         => Num (Tensor d n a) where
  _t1 + _t2 = let (t1,t2) = bin_broadcast' _t1 _t2
                  eres = unsafePerformIO $ runExceptT $ do
                    t1' <- unsafeThaw t1
                    t2' <- unsafeThaw t2
                    t3' <- MT.copy t2'
                    liftIO $ MT.withDevicePtr t1' $ \t1ptr -> do
                      MT.withDevicePtr t3' $ \t3ptr -> runDeviceM (Proxy :: Proxy d) $ do
                        MT.rawAdd t1ptr t3ptr
                          $ fromIntegral $ size $ shape t1
                    unsafeFreeze t3' in
               unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))
  _t1 * _t2 = let (t1,t2) = bin_broadcast' _t1 _t2
                  eres = unsafePerformIO $ runExceptT $ do
                    t1' <- unsafeThaw t1
                    t2' <- unsafeThaw t2
                    t3' <- MT.copy t2'
                    liftIO $ MT.withDevicePtr t1' $ \t1ptr -> do
                      MT.withDevicePtr t3' $ \t3ptr -> runDeviceM (Proxy :: Proxy d) $ do
                        MT.rawMul t1ptr t3ptr
                          $ fromIntegral $ size $ shape t1
                    unsafeFreeze t3' in
               unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))
  _t1 - _t2 = let (t1,t2) = bin_broadcast' _t1 _t2
                  eres = unsafePerformIO $ runExceptT $ do
                    t1' <- unsafeThaw t1
                    t2' <- unsafeThaw t2
                    t3' <- MT.copy t2'
                    liftIO $ MT.withDevicePtr t1' $ \t1ptr -> do
                      MT.withDevicePtr t3' $ \t3ptr -> runDeviceM (Proxy :: Proxy d) $ do
                        MT.rawSubtract t1ptr t3ptr
                          $ fromIntegral $ size $ shape t1
                    unsafeFreeze t3' in
               unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))
  negate t = let eres = unsafePerformIO $ runExceptT $ do
                   res <- unsafeThaw t >>= MT.copy
                   liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (Proxy :: Proxy d) $ do
                     MT.rawNegate resptr
                       $ fromIntegral $ size $ shape t
                   unsafeFreeze res in
              unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))
  signum t = let eres = unsafePerformIO $ runExceptT $ do
                   res <- unsafeThaw t >>= MT.copy
                   liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (Proxy :: Proxy d) $ do
                     MT.rawSignum resptr
                       $ fromIntegral $ size $ shape t
                   unsafeFreeze res in
              unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))
  fromInteger i = fromInteger i *^ ones' scalarShape

instance forall d n a
         . (Device d, Shape (Dim n), TensorScalar a)
         => Fractional (Tensor d n a) where
  recip x = let eres = unsafePerformIO $ runExceptT $ do
                  res <- unsafeThaw x >>= MT.copy
                  liftIO $ MT.inv res
                  unsafeFreeze res in
             unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))
  fromRational r = fromInteger (numerator r) / fromInteger (denominator r)

-- | Computes the elementwise maximum of 2 tensors.
elementwiseMax :: forall d n a t m
               . (Device d, Shape (Dim n), TensorScalar a, MonadError t m,
                  Variant t OutOfMemory, Variant t IncompatibleShape)
               => Tensor d n a -> Tensor d n a -> m (Tensor d n a)
elementwiseMax _x _y = do
  (x,y) <- bin_broadcast _x _y
  embedExcept $ unsafePerformIO $ runExceptT $ do
    mx <- unsafeThaw x
    my <- unsafeThaw y >>= MT.copy
    liftIO $ MT.withDevicePtr mx $ \pmx -> do
      MT.withDevicePtr my $ \pmy -> runDeviceM (Proxy :: Proxy d) $ do
        MT.rawMax pmx pmy (fromIntegral $ size $ shape x)
    unsafeFreeze my

fromRaw :: forall d n a
        . (Device d, Shape (Dim n), TensorScalar a)
        => (CUDA.DevicePtr a -> CSize -> DeviceM d ())
        -> Tensor d n a -> Tensor d n a
fromRaw action x =
  let eres = unsafePerformIO $ runExceptT $ do
        mx <- unsafeThaw x
        res <- MT.copy mx
        liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (Proxy :: Proxy d) $ do
          action resptr $ fromIntegral $ size $ shape x
        unsafeFreeze res
  in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))

instance forall d n a
         . (Device d, Shape (Dim n), TensorScalar a) => Floating (Tensor d n a) where
  pi = pi *^ ones' scalarShape
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
  _x**_y = let (x,y) = bin_broadcast' _x _y
               eres = unsafePerformIO $ runExceptT $ do
                 mx <- unsafeThaw x
                 my <- unsafeThaw y >>= MT.copy
                 liftIO $ MT.withDevicePtr mx $ \pmx -> do
                   MT.withDevicePtr my $ \pmy -> runDeviceM (Proxy :: Proxy d) $ do
                     MT.rawPow pmx pmy
                       $ fromIntegral $ size $ shape x
                 unsafeFreeze my
           in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))

-- Vector space instance for Tensors.
instance (Device d, Shape (Dim n), TensorScalar a)
         => AdditiveGroup (Tensor d n a) where
  zeroV = fromInteger 0
  t1 ^+^ t2 = t1 + t2
  negateV t = negate t

instance forall d n a
         . (Device d, Shape (Dim n), TensorScalar a)
         => VectorSpace (Tensor d n a) where
  type Scalar (Tensor d n a) = a
  x *^ t = let eres = unsafePerformIO $ runExceptT $ do
                 res <- unsafeThaw t >>= MT.copy
                 liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (Proxy :: Proxy d) $ do
                   MT.rawScale x resptr
                     $ fromIntegral $ size $ shape t
                 unsafeFreeze res
           in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d n a))

instance (Device d, Shape (Dim n), TensorScalar a)
         => InnerSpace (Tensor d n a) where
  t1 <.> t2 = foldr (+) 0 $ fmap (\(x1,x2) -> x1 * x2) $ zip (tensorToList t1) (tensorToList t2)
