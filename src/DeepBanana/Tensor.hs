{-# LANGUAGE DeriveGeneric, UndecidableInstances #-}
{-# LANGUAGE TypeFamilies, BangPatterns #-}
{-# LANGUAGE RankNTypes, AllowAmbiguousTypes #-}
{-|
Immutable GPU multi-dimensional dense numeric arrays.
-}
module DeepBanana.Tensor (
    Tensor(..)
  -- * Basic manipulations
  , dtype
  , zeros
  , zeros'
  , ones
  , ones'
  -- * Shape manipulations
  , reshape
  , reshape'
  , shapeConvert
  , liftFixed1
  , liftFixed2
  , broadcast
  , broadcast'
  -- * Device manipulations
  , deviceConvert
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
  -- * Re-exports
  , module DeepBanana.Tensor.Shape
  , module DeepBanana.Tensor.Exception
  , module DeepBanana.Tensor.TensorScalar
  ) where

import Data.Serialize (Serialize)
import qualified Data.Serialize as S
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
import DeepBanana.Weights

-- | An immutable,  dense numerical array stored on GPU device @d@, with shape @s@ and
-- scalar datatype @a@.
-- As long as the shape and device datatypes are fixed at the type level, it is an
-- instance of all standard numeric type classes in an elementwise fashion.
data Tensor d s a = Tensor {
    device :: d
  , shape :: s
  , dataptr :: !(ForeignPtr a)
  }

instance (Shape s, Device d, TensorScalar a, Eq a) => Eq (Tensor d s a) where
  t1 == t2 = device t1 == device t2
             && tensorToList t1 == tensorToList t2
             && shape t1 == shape t2

instance (Device d, Shape s, TensorScalar a)
         => Serialize (Tensor d s a) where
  put t = do
    S.put $ device t
    S.put $ shape t
    S.put $ tensorToList t
  get = do
    dev <- S.get
    shp <- S.get
    tlist <- S.get
    return $ tensorFromList' dev shp tlist

instance (NFData d, NFData s) => NFData (Tensor d s a) where
  rnf (Tensor d s _) = rnf (s, d)

instance (Device d, Shape s, TensorScalar a, Show a)
         => Show (Tensor d s a) where
  show t = "Tensor " ++ show (device t) ++ " " ++ show  (shape t)
           ++ " "  ++ show (take 10 $ tensorToList t)

-- | Reshaping. Throws an @'IncompatibleSize'@ exception when the desired size and the
-- input size do not correspond.
reshape :: (Shape s1, Shape s2, TensorScalar a,
            MonadError t m, Variant t IncompatibleSize)
        => s2 -> Tensor d s1 a -> m (Tensor d s2 a)
reshape newshp t@(Tensor dev oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwVariant $ incompatibleSize $
    "Couldn't reshape tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ Tensor dev newshp dataptr

-- | Unsafe reshaping. Throws an imprecise exception when the desired size and the
-- input size do not correspond.
reshape' :: forall d s1 s2 a
         . (Shape s1, Shape s2, TensorScalar a)
         => s2 -> Tensor d s1 a -> Tensor d s2 a
reshape' newshp t =
  unsafeRunExcept (reshape newshp t :: Either IncompatibleSize (Tensor d s2 a))

-- | Shape conversion. This differs from @'reshape'@ in that the desired shape and
-- input shape must have exactly the same dimensions. This is useful when converting
-- between dynamic and fixed shape datatypes. Throws an @'IncompatibleShape'@ exception
-- when the 2 shapes dimensions differ.
shapeConvert :: forall s1 s2 d l a m t
             . (MonadError t m, Shape s1, Shape s2, Variant t IncompatibleShape)
             => s2 -> Tensor d s1 a -> m (Tensor d s2 a)
shapeConvert outshp (Tensor dev shp datafptr) = do
  when (dimensions outshp /= dimensions shp) $ throwVariant $ incompatibleShape $
    "Cannot convert tensor from shape " ++ show shp ++ " to shape " ++ show outshp
  return $ Tensor dev outshp datafptr

-- | Device conversion. This differs from @'transfer'@ in that the desired device and
-- input device must actually refer to the same device. This is useful when converting
-- between dynamic and fixed device datatypes. Throws an @'IncompatibleDevice'@
-- exception when the 2 devices differ.
deviceConvert :: (MonadError t m, Device d, Device d', Variant t IncompatibleDevice)
              => d' -> Tensor d s a -> m (Tensor d' s a)
deviceConvert outDev (Tensor inpDev shp datafptr) = do
  when (deviceId outDev /= deviceId inpDev) $ throwVariant $ incompatibleDevice $
    "Cannot convert tensor from device " ++ show inpDev ++ " to device " ++ show outDev
  return $ Tensor outDev shp datafptr

-- | Lifts an unary function on fixed shape tensors to work on any tensor. Useful to
-- apply unary numeric functions such as @'log'@ or @'exp'@ to dynamic shape tensors
-- in a type safe fashion.
liftFixed1 :: forall d s a
           . (Shape s, Device d)
           => (forall s' d' . (IsFixed s', ValidUniqueDevice d')
               => Tensor d' s' a -> Tensor d' s' a)
           -> Tensor d s a -> Tensor d s a
liftFixed1 f x = unsafeRunExcept eres
  where
    eres :: Either (Coproduct '[IncompatibleShape, IncompatibleDevice]) (Tensor d s a)
    eres = do
      case toAnyFixed $ shape x of
       AnyFixed fshp -> do
         fx <- shapeConvert fshp x
         withValidUniqueDevice (device x) $ \dev' -> do
           dfx <- deviceConvert dev' fx
           dfy <- shapeConvert (shape x) $ f dfx
           deviceConvert (device x) dfy

instance (Device d, Shape s, TensorScalar a)
         => FixShape (Tensor d s a) where
  type FixScalar (Tensor d s a) = a
  liftVec = liftFixed2

-- | Lifts a binary function on fixed shape tensors to work on any tensor. Throws
-- an @'IncompatibleShape'@ exception when the input tensors do not have the same
-- shape. Useful to apply binary numeric functions such as @'(+)'@ and @'(*)'@ to
-- dynamic shape tensors in a type safe fashion.
liftFixed2 :: (Device d, Shape s, MonadError t m, Variant t IncompatibleShape,
               Variant t IncompatibleDevice)
              => (forall d' s' . (ValidUniqueDevice d', IsFixed s')
               => Tensor d' s' a -> Tensor d' s' a -> Tensor d' s' a)
           -> Tensor d s a -> Tensor d s a -> m (Tensor d s a)
liftFixed2 g x1 x2 = do
  case toAnyFixed $ shape x1 of
   AnyFixed fshp -> do
     fx1 <- shapeConvert fshp x1
     fx2 <- shapeConvert fshp x2
     withValidUniqueDevice (device x1) $ \dev' -> do
       dfx1 <- deviceConvert dev' fx1
       dfx2 <- deviceConvert dev' fx2
       dfy <- shapeConvert (shape x1) $ g dfx1 dfx2
       deviceConvert (device x1) dfy

instance (Device d, Device d', TensorScalar a, Shape s)
         => DeviceTransfer (Tensor d s a) d' where
  type Transferred (Tensor d s a) d' = Tensor d' s a
  transfer dev t = embedExcept $ runST $ runExceptT $ do
    mt <- unsafeThaw t
    mt' <- MT.copy dev mt
    unsafeFreeze mt'

-- | Returns the CuDNN datatype of a tensor.
dtype :: forall d s a . (TensorScalar a) => Tensor d s a -> CuDNN.DataType
dtype _ = datatype (Proxy :: Proxy a)

-- | Converts a mutable tensor into an immutable one in O(1) time. The data is not
-- copied, so one should make sure the input mutable tensor is not modified after the
-- the conversion. Unsafe.
unsafeFreeze :: (PrimMonad m) => MTensor (PrimState m) d s a -> m (Tensor d s a)
unsafeFreeze (MT.MTensor dev shp ptr) = return $ Tensor dev shp ptr

-- | Converts an immutable tensor into a mutable one in O(1) time. The data is not
-- copied, so one should make sure the input immutable tensor is not used after the
-- conversion. Unsafe.
unsafeThaw :: (PrimMonad m) => Tensor d s a -> m (MTensor (PrimState m) d s a)
unsafeThaw (Tensor dev shp ptr) = return $ MT.MTensor dev shp ptr

-- | Returns a tensor with all zero coefficients. Throws an @'OutOfMemory'@ exception
-- when the desired device is out of memory.
zeros :: (Device d, Shape s, TensorScalar a, MonadError t m,
          Variant t OutOfMemory)
      => d -> s -> m (Tensor d s a)
zeros dev shp = do
  embedExcept $ runST $ runExceptT $ MT.zeros dev shp >>= unsafeFreeze

-- | Returns a tensor with all zero coefficients. Throws an imprecise exception when
-- the desired device is out of memory.
zeros' :: forall d s a
       . (Device d, Shape s, TensorScalar a)
       => d -> s -> Tensor d s a
zeros' dev shp = unsafeRunExcept (zeros dev shp :: Either OutOfMemory (Tensor d s a))

-- | Initializes a tensor with all ones. Throws an @'OutOfMemory'@ exception
-- when the desired device is out of memory.
ones :: (Device d, Shape s, TensorScalar a, MonadError t m,
         Variant t OutOfMemory)
     => d -> s -> m (Tensor d s a)
ones dev shp = do
  embedExcept $ runST $ runExceptT $ MT.ones dev shp >>= unsafeFreeze

-- | Returns a tensor with all one coefficients. Throws an imprecise exception when
-- the desired device is out of memory.
ones' :: forall d s a
      . (Device d, Shape s,  TensorScalar a)
      => d -> s -> Tensor d s a
ones' dev shp = unsafeRunExcept (ones dev shp :: Either OutOfMemory (Tensor d s a))

-- | Initializes a tensor from list data. If the length of the list does not correspond
-- to the size of the desired shape, throws an exception at run time.
tensorFromList :: (Device d, Shape s, TensorScalar a, MonadError t m,
                   Variant t IncompatibleSize, Variant t OutOfMemory)
               => d -> s -> [a] -> m (Tensor d s a)
tensorFromList dev shape value = do
  embedExcept $ runST $ runExceptT $ MT.tensorFromList dev shape value >>= unsafeFreeze

-- | Unsafe version of @'tensorFromList'@, throws imprecise exceptions.
tensorFromList' :: forall d s a
                . (Device d, Shape s, TensorScalar a)
                => d -> s -> [a] -> Tensor d s a
tensorFromList' dev shape value =
  unsafeRunExcept (tensorFromList dev shape value
                   :: Either (Coproduct '[IncompatibleSize, OutOfMemory]) (Tensor d s a))

-- | Converts a tensor to a list.
tensorToList :: (Shape s, Device d, TensorScalar a) => Tensor d s a -> [a]
tensorToList t = runST $ unsafeThaw t >>= MT.tensorToList

-- | Converts a storable vector to an immutable tensor. Throws an error at runtime
-- when the input's length does not correspond to the desired output size. Runs in
-- O(n) time.
fromVector :: forall t m d s a
           . (Device d, Shape s, TensorScalar a, MonadError t m,
              Variant t OutOfMemory, Variant t IncompatibleSize)
           => d -> s -> SVector a -> m (Tensor d s a)
fromVector dev shp value = do
  when (length value /= size shp) $ throwVariant $ incompatibleSize $
    "fromVector: Incompatible sizes\n\tinput vector: "
    ++ show (length value) ++ "\n\trequired output shape: "
    ++ show shp
    ++ ", size " ++ show (size shp)
  embedExcept $ unsafePerformIO $ runExceptT $ do
    res <- MT.emptyTensor dev shp
    liftIO $ SV.unsafeWith value $ \vptr -> do
      MT.withDevicePtr res $ \resptr -> do
        runDeviceM dev $ CUDA.pokeArray (size shp) vptr resptr
    unsafeFreeze res

-- | Unsafe version of @'fromVector'@, throws imprecise exceptions.
fromVector' :: forall d s a
            . (Device d, Shape s, TensorScalar a)
            => d -> s -> SVector a -> Tensor d s a
fromVector' dev shp value =
  unsafeRunExcept (fromVector dev shp value
                   :: Either (Coproduct '[OutOfMemory, IncompatibleSize]) (Tensor d s a))

-- | Converts a tensor to a storable vector. Runs in O(n) time.
toVector :: forall d s a
         . (Device d, Shape s, TensorScalar a) => Tensor d s a -> SVector a
toVector t = unsafePerformIO $ do
  res <- SMV.new $ size $ shape t
  mt <- unsafeThaw t
  SMV.unsafeWith res $ \resptr -> do
    MT.withDevicePtr mt $ \mtptr -> do
      runDeviceM (device t) $ CUDA.peekArray (size $ shape t) mtptr resptr
  SV.unsafeFreeze res

-- | Concatenates two 1-dimensional tensors. Inverse of tsplit. Runs in O(n+m) time.
tconcat :: forall t d m a
        . (Device d, TensorScalar a, MonadError t m, Variant t OutOfMemory)
        => Tensor d (Dim 1) a -> Tensor d (Dim 1) a -> m (Tensor d (Dim 1) a)
tconcat t1 t2 = do
  embedExcept $ unsafePerformIO $ runExceptT $ do
    let t1sz = size $ shape t1
        t2sz = size $ shape t2
    out <- MT.emptyTensor (device t1) $ (t1sz + t2sz) :. Z
    mt1 <- unsafeThaw t1
    mt2 <- unsafeThaw t2
    liftIO $ MT.withDevicePtr mt1 $ \t1ptr -> do
      MT.withDevicePtr mt2 $ \t2ptr -> do
        MT.withDevicePtr out $ \outptr -> runDeviceM (device t1) $ do
          CUDA.copyArray t1sz t1ptr outptr
          CUDA.copyArray t2sz t2ptr
            (CUDA.DevicePtr
             $ plusPtr
             (CUDA.useDevicePtr outptr)
             (mySizeOf (Proxy :: Proxy a) * fromIntegral t1sz))
    unsafeFreeze out

-- | Unsafe version of @'tconcat'@, throws imprecise exceptions.
tconcat' :: forall d a
         . (Device d, TensorScalar a)
         => Tensor d (Dim 1) a -> Tensor d (Dim 1) a -> Tensor d (Dim 1) a
tconcat' t1 t2 =
  unsafeRunExcept (tconcat t1 t2 :: Either OutOfMemory (Tensor d (Dim 1) a))

-- | Splits a 1-dimensional tensor into 2 parts. Inverse of tconcat. Runs in O(n+m) time.
tsplitAt :: forall t d m a
         . (Device d, TensorScalar a, MonadError t m, Variant t OutOfMemory,
            Variant t IncompatibleSize)
         => Int -> Tensor d (Dim 1) a -> m (Tensor d (Dim 1) a, Tensor d (Dim 1) a)
tsplitAt t1sz t = do
  let t2sz = size (shape t) - t1sz
  when (t2sz < 0) $ throwVariant $ incompatibleSize $
    "Couldn't split the a vector of size " ++ show (size (shape t)) ++ " at point " ++ show t1sz ++ " : the input vector is too small."
  embedExcept $ unsafePerformIO $ runExceptT $ do
    mt1 <- MT.emptyTensor (device t) $ t1sz :. Z
    mt2 <- MT.emptyTensor (device t) $ t2sz :. Z
    mt <- unsafeThaw t
    liftIO $ MT.withDevicePtr mt1 $ \t1ptr -> do
      MT.withDevicePtr mt2 $ \t2ptr -> do
        MT.withDevicePtr mt $ \tptr -> runDeviceM (device t) $ do
          CUDA.copyArray t1sz tptr t1ptr
          let offsetPtr =
                CUDA.DevicePtr
                $ plusPtr
                (CUDA.useDevicePtr tptr)
                (mySizeOf (Proxy :: Proxy a) * fromIntegral t1sz)
          CUDA.copyArray t2sz offsetPtr t2ptr
    pure (,) <*> unsafeFreeze mt1 <*> unsafeFreeze mt2

-- | Unsafe version of @'tSplit'@, throws imprecise exceptions.
tsplitAt' :: forall d a
          . (Device d, TensorScalar a)
          => Int -> Tensor d (Dim 1) a -> (Tensor d (Dim 1) a, Tensor d (Dim 1) a)
tsplitAt' t1sz t = unsafeRunExcept (tsplitAt t1sz t :: Either (Coproduct '[OutOfMemory,IncompatibleSize]) (Tensor d (Dim 1) a, Tensor d (Dim 1) a))

-- | Flattens a tensor into a 1-dimensional vector.
flatten :: forall s d a
        . (Shape s, TensorScalar a)
        => Tensor d s a -> Tensor d (Dim 1) a
flatten t =
  case reshape (size (shape t) :. Z) t
       :: Either IncompatibleSize (Tensor d (Dim 1) a) of
   Left err -> error $ "Couldn't flatten a tensor. This shouldn't happen.\n" ++ show err
   Right res -> res

-- | Type-safe broadcasting. See the documentation for @'broadcastable'@ for the
-- broadcasting rules. This throws an @'IncompatibleShape'@ exception when the shape
-- of the tensor is not broadcastable to the desired shape. Note that, unlike numpy,
-- this currently makes a full copy of the tensor in memory.
broadcast :: forall d t s1 s2 m a
          . (Device d, Shape s1, Shape s2, TensorScalar a, MonadError t m,
             Variant t IncompatibleShape, Variant t OutOfMemory)
          => s2 -> Tensor d s1 a -> m (Tensor d s2 a)
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
    mout <- MT.emptyTensor (device inp) outshp
    liftIO $ MT.withDevicePtr minp $ \inpptr -> do
      MT.withDevicePtr mout $ \outptr -> runDeviceM (device inp) $ do
        inp_shapeptr <- CUDA.newListArray inp_shape
        out_shapeptr <- CUDA.newListArray out_shape
        broadcast_copy out_nbdim out_size inpptr inp_shapeptr outptr out_shapeptr
        CUDA.free inp_shapeptr
        CUDA.free out_shapeptr
    unsafeFreeze mout

-- | Unsafe version of @'broadcast'@, throws imprecise exceptions.
broadcast' :: forall s1 s2 d a
           . (Device d, Shape s1, Shape s2, TensorScalar a)
           => s2 -> Tensor d s1 a -> Tensor d s2 a
broadcast' outshp inp =
  unsafeRunExcept (broadcast outshp inp
                   :: Either (Coproduct '[IncompatibleShape, OutOfMemory]) (Tensor d s2 a))

-- | Broadcast two tensors to the same shape, if either of them is broadcastable to
-- to the other. Throws an @'IncompatibleShape'@ exception otherwise.
bin_broadcast :: forall s d t m a
              . (Device d, Shape s, TensorScalar a, MonadError t m,
                 Variant t IncompatibleShape, Variant t OutOfMemory)
              => Tensor d s a -> Tensor d s a -> m (Tensor d s a, Tensor d s a)
bin_broadcast t1 t2 =
  case (broadcast (shape t2) t1, broadcast (shape t1) t2) of
   (Left err1, Left err2) -> throwError err1 >> throwError err2
   (Right t1', Left err) -> return (t1', t2)
   (Left err, Right t2') -> return (t1, t2')
   _ -> return (t1, t2)

-- | Unsafe version of @'bin_broadcast'@, throws imprecise exceptions.
bin_broadcast' :: forall d s a
               . (Device d, Shape s, TensorScalar a)
               => Tensor d s a -> Tensor d s a -> (Tensor d s a, Tensor d s a)
bin_broadcast' t1 t2 =
  unsafeRunExcept (bin_broadcast t1 t2
                   :: Either (Coproduct '[IncompatibleShape, OutOfMemory]) (Tensor d s a, Tensor d s a))

instance forall d s a
         . (ValidUniqueDevice d, IsFixed s, TensorScalar a)
         => Num (Tensor d s a) where
  t1 + t2 = let
    eres = unsafePerformIO $ runExceptT $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy (device t1) t2'
      liftIO $ MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> runDeviceM (device t1) $ do
          MT.rawAdd t1ptr t3ptr
            $ fromIntegral $ size $ shape t1
      unsafeFreeze t3' in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))

  t1 * t2 = let 
    eres = unsafePerformIO $ runExceptT $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy (device t1) t2'
      liftIO $ MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> runDeviceM (device t1) $ do
          MT.rawMul t1ptr t3ptr
            $ fromIntegral $ size $ shape t1
      unsafeFreeze t3' in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))

  t1 - t2 = let
    eres = unsafePerformIO $ runExceptT $ do
      t1' <- unsafeThaw t1
      t2' <- unsafeThaw t2
      t3' <- MT.copy (device t1) t2'
      liftIO $ MT.withDevicePtr t1' $ \t1ptr -> do
        MT.withDevicePtr t3' $ \t3ptr -> runDeviceM (device t1) $ do
          MT.rawSubtract t1ptr t3ptr
            $ fromIntegral $ size $ shape t1
      unsafeFreeze t3' in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))
  negate t = let eres = unsafePerformIO $ runExceptT $ do
                   res <- unsafeThaw t >>= MT.copy (device t)
                   liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (device t) $ do
                     MT.rawNegate resptr
                       $ fromIntegral $ size $ shape t
                   unsafeFreeze res in
              unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))
  signum t = let eres = unsafePerformIO $ runExceptT $ do
                   res <- unsafeThaw t >>= MT.copy (device t)
                   liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (device t) $ do
                     MT.rawSignum resptr
                       $ fromIntegral $ size $ shape t
                   unsafeFreeze res in
              unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))
  fromInteger i = fromInteger i *^ ones' validUniqueDevice scalarShape

instance forall d s a
         . (ValidUniqueDevice d, IsFixed s, TensorScalar a)
         => Fractional (Tensor d s a) where
  recip x = let eres = unsafePerformIO $ runExceptT $ do
                  res <- unsafeThaw x >>= MT.copy (device x)
                  liftIO $ MT.inv res
                  unsafeFreeze res in
             unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))
  fromRational r = fromInteger (numerator r) / fromInteger (denominator r)

-- | Computes the elementwise maximum of 2 tensors.
elementwiseMax :: forall d s a t m
               . (ValidUniqueDevice d, IsFixed s, TensorScalar a, MonadError t m,
                  Variant t OutOfMemory, Variant t IncompatibleShape)
               => Tensor d s a -> Tensor d s a
               -> m (Tensor d s a)
elementwiseMax _x _y = do
  (x,y) <- bin_broadcast _x _y
  embedExcept $ unsafePerformIO $ runExceptT $ do
    mx <- unsafeThaw x
    my <- unsafeThaw y >>= MT.copy (device x)
    liftIO $ MT.withDevicePtr mx $ \pmx -> do
      MT.withDevicePtr my $ \pmy -> runDeviceM (device x) $ do
        MT.rawMax pmx pmy (fromIntegral $ size $ shape x)
    unsafeFreeze my

fromRaw :: forall d s a
        . (Device d, IsFixed s, TensorScalar a)
        => (CUDA.DevicePtr a -> CSize -> DeviceM d ())
        -> Tensor d s a -> Tensor d s a
fromRaw action x =
  let eres = unsafePerformIO $ runExceptT $ do
        mx <- unsafeThaw x
        res <- MT.copy (device x) mx
        liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (device x) $ do
          action resptr $ fromIntegral $ size $ shape x
        unsafeFreeze res
  in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))

instance forall d s a
         . (ValidUniqueDevice d, IsFixed s, TensorScalar a)
         => Floating (Tensor d s a) where
  pi = pi *^ ones' validUniqueDevice scalarShape
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
  x**y = let
    eres = unsafePerformIO $ runExceptT $ do
      mx <- unsafeThaw x
      my <- unsafeThaw y >>= MT.copy (device x)
      liftIO $ MT.withDevicePtr mx $ \pmx -> do
        MT.withDevicePtr my $ \pmy -> runDeviceM (device x) $ do
          MT.rawPow pmx pmy
            $ fromIntegral $ size $ shape x
      unsafeFreeze my
    in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))

-- Vector space instance for Tensors.
instance (ValidUniqueDevice d, IsFixed s, TensorScalar a)
         => AdditiveGroup (Tensor d s a) where
  zeroV = fromInteger 0
  t1 ^+^ t2 = t1 + t2
  negateV t = negate t

instance forall d s a
         . (ValidUniqueDevice d, IsFixed s, TensorScalar a)
         => VectorSpace (Tensor d s a) where
  type Scalar (Tensor d s a) = a
  x *^ t = let eres = unsafePerformIO $ runExceptT $ do
                 res <- unsafeThaw t >>= MT.copy (device t)
                 liftIO $ MT.withDevicePtr res $ \resptr -> runDeviceM (device t) $ do
                   MT.rawScale x resptr
                     $ fromIntegral $ size $ shape t
                 unsafeFreeze res
           in unsafeRunExcept (eres :: Either OutOfMemory (Tensor d s a))

instance (ValidUniqueDevice d, IsFixed s, TensorScalar a)
         => InnerSpace (Tensor d s a) where
  t1 <.> t2 = foldr (+) 0 $ fmap (\(x1,x2) -> x1 * x2) $ zip (tensorToList t1) (tensorToList t2)
