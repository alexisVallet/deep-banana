{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-|
Mutable GPU multi-dimensional dense numeric arrays, parametrized by the state token of
some primitive state monad.
-}
module DeepBanana.Tensor.Mutable (
  MTensor(..)
  , IOTensor(..)
  , dtype
  -- * Creating mutable tensors
  , emptyTensor
  , tensorFromList
  , zeros
  , ones
  -- * Basic manipulations
  , withDevicePtr
  , reshape
  , reshape'
  , copy
  , tensorToList
  -- * In place numeric functions
  , threshInplace
  , tlog
  , texp
  , inv
  -- * Re-exports
  , module DeepBanana.Tensor.Shape
  , module DeepBanana.Tensor.TensorScalar
  ) where

import Control.Monad.Primitive (unsafePrimToPrim)
import Foreign.ForeignPtr
import Foreign.Marshal
import System.IO.Unsafe
import System.Mem
import Unsafe.Coerce

import DeepBanana.Device
import DeepBanana.Device.Cubits (freeDevicePtr)
import qualified DeepBanana.Device.CUDA as CUDA
import qualified DeepBanana.Device.CuDNN as CuDNN
import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Tensor.Exception
import DeepBanana.Tensor.Shape
import DeepBanana.Tensor.TensorScalar

-- | Mutable tensor, parametrized by some state token @st@, stored on a device @d@, with
-- shape @s@ and scalar datatype @a@.
data MTensor st d s a = MTensor {
    device :: d
  , shape :: s
  , dataptr :: ForeignPtr a
  }

-- | Convenient synonym for mutable tensors in the @IO@ monad.
type IOTensor = MTensor RealWorld

-- | Returns the CuDNN datatype identifier for the tensor.
dtype :: forall m d n a . (PrimMonad m, TensorScalar a)
      => MTensor (PrimState m) d n a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

-- | Creates an uninitialized mutable tensor with a given shape. Uninitialized means the
-- contents of the tensor are defined by whatever was lying around in memory at the
-- time. In case there is not enough space on the device, will throw and @'OutOfMemory'@
-- exception after attempting garbage collection first.
emptyTensor :: forall s d m t a
            . (Shape s, Device d, TensorScalar a, PrimMonad m, MonadError t m,
               Variant t OutOfMemory)
            => d -> s -> m (MTensor (PrimState m) d s a)
emptyTensor dev shp = do
  let action = do
        let handleOutOfMem e = case (e :: CUDA.CUDAException) of
              CUDA.ExitCode CUDA.MemoryAllocation ->
                return $ Left $ outOfMemory $
                "emptyTensor: failed to allocate a tensor of shape " ++ show shp
              err -> error $ "emptyTensor: unhandled exception: " ++ show err
        edvcptr <- liftIO (fmap Right (runDeviceM dev
                                       $ CUDA.mallocArray (size shp)))
                   `catch` handleOutOfMem
        case edvcptr of
         Left err -> throwError err
         Right dvcptr -> do
           datafptr <- liftIO $ newForeignPtr freeDevicePtr
                       (CUDA.useDevicePtr dvcptr)
           return $ MTensor dev shp datafptr
  eres <- unsafePrimToPrim
          $ runExceptT
          $ attemptGCThenRetryOn (Proxy :: Proxy OutOfMemory)
          $ (action :: ExceptT OutOfMemory IO (MTensor (PrimState m) d s a))
  case eres of
   Left err -> throwVariant err
   Right res -> return res

-- | Runs an 'IO' action with the internal device pointer of a given mutable tensor.
withDevicePtr :: (Storable a)
              => IOTensor d s a
              -> (CUDA.DevicePtr a -> IO b)
              -> IO b
withDevicePtr (MTensor _ _ datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)

-- | Creates a mutable tensor from a list. Throws an exception at runtime when the size
-- of the list does not correspond to the desired shape.
tensorFromList :: forall m d s a t
               . (PrimMonad m, Device d, Shape s, TensorScalar a,
                  MonadError t m, Variant t OutOfMemory, Variant t IncompatibleSize)
               => d -> s -> [a] -> m (MTensor (PrimState m) d s a)
tensorFromList dev shp content = do
  when (size shp /= length content) $ throwVariant $ incompatibleSize $
    "tensorFromList: input list and desired shape should have the same size.\nInput list length: " ++ show (length content) ++ "\nDesired shape: " ++ show shp ++ ", size " ++ show (size shp)
  unsafeFromList dev shp content

-- | Creates a mutable tensor from a list, not checking that the provided list length
-- and shape size correspond.
unsafeFromList :: forall m d s a t
               . (PrimMonad m, Device d, Shape s, TensorScalar a,
                  MonadError t m, Variant t OutOfMemory)
               => d -> s -> [a] -> m (MTensor (PrimState m) d s a)
unsafeFromList dev shp content = do
  tensor <- emptyTensor dev shp :: m (MTensor (PrimState m) d s a)
  unsafePrimToPrim $ do
    withArray content $ \dataptr -> do
      let tensor' = unsafeCoerce tensor :: IOTensor d s a
      withDevicePtr tensor' $ \dvcptr -> do
        runDeviceM dev $ CUDA.pokeArray (size shp) dataptr dvcptr
  return tensor

-- | Converts a mutable tensor to a list of elements.
tensorToList :: forall m d s a
             . (PrimMonad m, Device d, Shape s, TensorScalar a)
             => MTensor (PrimState m) d s a -> m [a]
tensorToList tensor = unsafePrimToPrim $ do
  withDevicePtr (unsafeCoerce tensor :: IOTensor d s a)
    $ runDeviceM (device tensor) . CUDA.peekListArray (size $ shape tensor)

-- | Creates a mutable tensor filled with zeros.
zeros :: (PrimMonad m, Device d, Shape s, TensorScalar a, MonadError t m,
          Variant t OutOfMemory)
      => d -> s -> m (MTensor (PrimState m) d s a)
zeros dev shp = unsafeFromList dev shp $ take (size shp) $ repeat 0

-- | Creates a mutable tensor filled with ones.
ones :: (PrimMonad m, Device d, Shape s, TensorScalar a, MonadError t m,
         Variant t OutOfMemory)
      => d -> s -> m (MTensor (PrimState m) d s a)
ones dev shp = unsafeFromList dev shp $ take (size shp) $ repeat 1

-- | Copies the data of a mutable tensor into a new one.
copy :: forall m d1 d2 s a t
     . (PrimMonad m, Device d2, Shape s, TensorScalar a,
        MonadError t m, Variant t OutOfMemory)
     => d2 -> MTensor (PrimState m) d1 s a -> m (MTensor (PrimState m) d2 s a)
copy dev2 tensor = do
  out <- emptyTensor dev2 $ shape tensor :: m (MTensor (PrimState m) d2 s a)
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor d1 s a) $ \tensorptr -> do
      withDevicePtr (unsafeCoerce out :: IOTensor d2 s a) $ \outptr -> do
        runDeviceM dev2
          $ CUDA.copyArray (size $ shape tensor) tensorptr outptr
  return out

-- | Thresholds a mutable tensor in place. Used to implement dropout.
threshInplace :: forall m d s a
              . (PrimMonad m, Device d, Shape s, TensorScalar a)
              => MTensor (PrimState m) d s a -> a -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor d s a) $ \tensorptr -> do
      runDeviceM (device tensor)
        $ thresh tensorptr (fromIntegral $ size $ shape tensor) threshold tensorptr

-- | In place logarithm.
tlog :: forall m d s a . (PrimMonad m, Device d, Shape s, TensorScalar a)
     => MTensor (PrimState m) d s a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor d s a
  withDevicePtr iotensor $ \tptr -> do
    runDeviceM (device tensor)
      $ rawLog tptr (fromIntegral $ size $ shape tensor)

-- | In place exponential.
texp :: forall m d s a . (PrimMonad m, Device d, Shape s, TensorScalar a)
     => MTensor (PrimState m) d s a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor d s a
  withDevicePtr iot $ \tptr -> do
    runDeviceM (device t)
      $ rawExp tptr (fromIntegral $ size $ shape t)

-- | In place inverse.
inv :: forall m d s a . (PrimMonad m, Device d, Shape s, TensorScalar a)
    => MTensor (PrimState m) d s a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor d s a
  withDevicePtr iotensor $ \tptr -> do
    runDeviceM (device tensor)
      $ rawInv tptr $ fromIntegral $ size $ shape tensor

-- | Tensor reshaping. Will throw and @'IncompatibleSize'@ exception if the desired
-- shape size does not correspond to the input tensor size.
reshape :: (Shape s1, Shape s2, MonadError t m, Variant t IncompatibleSize)
        => s2 -> MTensor st d s1 a -> m (MTensor st d s2 a)
reshape newshp (MTensor dev oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwVariant $ incompatibleSize $
    "reshape: couldn't reshape mutable tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ MTensor dev newshp dataptr

-- | Unsafe pure tensor reshaping, which throws an imprecise exception when the desired
-- shape size does not correspond to the input tensor size.
reshape' :: forall d s1 s2 st a . (Shape s1, Shape s2)
         => s2 -> MTensor st d s1 a -> MTensor st d s2 a
reshape' newshp tensor =
  case (reshape newshp tensor :: Either IncompatibleSize (MTensor st d s2 a)) of
  Left err -> throw err
  Right out -> out
