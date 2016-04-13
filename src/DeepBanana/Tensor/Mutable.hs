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
  , module DeepBanana.Tensor.Shape
  , module DeepBanana.Tensor.TensorScalar
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

-- | Mutable tensor, parametrized by some state token 'st'.
data MTensor st d n a = MTensor {
    shape :: Dim n
  , dataptr :: ForeignPtr a
  }

-- | Convenient synonym for mutable tensors in the 'IO' monad.
type IOTensor = MTensor RealWorld

-- | Returns the CuDNN datatype identifier for the tensor.
dtype :: forall m d n a . (PrimMonad m, TensorScalar a)
      => MTensor (PrimState m) d n a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

-- | Creates an uninitialized mutable tensor with a given shape. Uninitialized means the
-- contents of the tensor are defined by whatever was lying around in memory at the
-- time.
emptyTensor :: forall n d m t a
            . (Shape (Dim n), Device d, TensorScalar a, PrimMonad m, MonadError t m,
               Variant t OutOfMemory)
            => Dim n -> m (MTensor (PrimState m) d n a)
emptyTensor shp = do
  let action = do
        let handleOutOfMem e = case (e :: CUDA.CUDAException) of
              CUDA.ExitCode CUDA.MemoryAllocation ->
                return $ Left $ outOfMemory $
                "emptyTensor: failed to allocate a tensor of shape " ++ show shp
              err -> error $ "emptyTensor: unhandled exception: " ++ show err
        edvcptr <- liftIO (fmap Right (runDeviceM (Proxy :: Proxy d)
                                       $ CUDA.mallocArray (size shp)))
                   `catch` handleOutOfMem
        case edvcptr of
         Left err -> throwError err
         Right dvcptr -> do
           datafptr <- liftIO $ newForeignPtr freeDevicePtr
                       (CUDA.useDevicePtr dvcptr)
           return $ MTensor shp datafptr
  eres <- unsafePrimToPrim
          $ runExceptT
          $ attemptGCThenRetryOn (Proxy :: Proxy OutOfMemory)
          $ (action :: ExceptT OutOfMemory IO (MTensor (PrimState m) d n a))
  case eres of
   Left err -> throwVariant err
   Right res -> return res

-- | Runs an 'IO' action with the internal device pointer of a given mutable tensor.
withDevicePtr :: (Storable a)
              => IOTensor d n a
              -> (CUDA.DevicePtr a -> IO b)
              -> IO b
withDevicePtr (MTensor _ datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)

-- | Creates a mutable tensor from a list. Throws an exception at runtime when the size
-- of the list does not correspond to the desired shape.
tensorFromList :: forall m d n a t
               . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a,
                  MonadError t m, Variant t OutOfMemory, Variant t IncompatibleSize)
               =>  Dim n -> [a] -> m (MTensor (PrimState m) d n a)
tensorFromList shp content = do
  when (size shp /= length content) $ throwVariant $ incompatibleSize $
    "tensorFromList: input list and desired shape should have the same size.\nInput list length: " ++ show (length content) ++ "\nDesired shape: " ++ show shp ++ ", size " ++ show (size shp)
  unsafeFromList shp content

unsafeFromList :: forall m d n a t
               . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a,
                  MonadError t m, Variant t OutOfMemory)
               => Dim n -> [a] -> m (MTensor (PrimState m) d n a)
unsafeFromList shp content = do
  tensor <- emptyTensor shp :: m (MTensor (PrimState m) d n a)
  unsafePrimToPrim $ do
    withArray content $ \dataptr -> do
      let tensor' = unsafeCoerce tensor :: IOTensor d n a
      withDevicePtr tensor' $ \dvcptr -> do
        runDeviceM (Proxy :: Proxy d) $ CUDA.pokeArray (size shp) dataptr dvcptr
  return tensor

-- | Converts a mutable tensor to a list of elements.
tensorToList :: forall m d n a
             . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a)
             => MTensor (PrimState m) d n a -> m [a]
tensorToList tensor = unsafePrimToPrim $ do
  withDevicePtr (unsafeCoerce tensor :: IOTensor d n a)
    $ runDeviceM (Proxy :: Proxy d) . CUDA.peekListArray (size $ shape tensor)

-- | Creates a mutable tensor filled with zeros.
zeros :: (PrimMonad m, Device d, Shape (Dim n), TensorScalar a, MonadError t m,
          Variant t OutOfMemory)
      => Dim n -> m (MTensor (PrimState m) d n a)
zeros shp = unsafeFromList shp $ take (size shp) $ repeat 0

-- | Creates a mutable tensor filled with ones.
ones :: (PrimMonad m, Device d, Shape (Dim n), TensorScalar a, MonadError t m,
         Variant t OutOfMemory)
      => Dim n -> m (MTensor (PrimState m) d n a)
ones shp = unsafeFromList shp $ take (size shp) $ repeat 1

-- | Copies the data of a mutable tensor into a new one.
copy :: forall m d1 d2 n a t
     . (PrimMonad m, Device d2, Shape (Dim n), TensorScalar a,
        MonadError t m, Variant t OutOfMemory)
     => MTensor (PrimState m) d1 n a -> m (MTensor (PrimState m) d2 n a)
copy tensor = do
  out <- emptyTensor $ shape tensor :: m (MTensor (PrimState m) d2 n a)
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor d1 n a) $ \tensorptr -> do
      withDevicePtr (unsafeCoerce out :: IOTensor d2 n a) $ \outptr -> do
        runDeviceM (Proxy :: Proxy d2)
          $ CUDA.copyArray (size $ shape tensor) tensorptr outptr
  return out

-- | Thresholds a mutable tensor in place. Used to implement dropout.
threshInplace :: forall m d n a
              . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a)
              => MTensor (PrimState m) d n a -> a -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor d n a) $ \tensorptr -> do
      runDeviceM (Proxy :: Proxy d)
        $ thresh tensorptr (fromIntegral $ size $ shape tensor) threshold tensorptr

-- | In place logarithm.
tlog :: forall m d n a . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a)
     => MTensor (PrimState m) d n a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor d n a
  withDevicePtr iotensor $ \tptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ rawLog tptr (fromIntegral $ size $ shape tensor)

-- | In place exponential.
texp :: forall m d n a . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a)
     => MTensor (PrimState m) d n a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor d n a
  withDevicePtr iot $ \tptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ rawExp tptr (fromIntegral $ size $ shape t)

-- | In place inverse.
inv :: forall m d n a . (PrimMonad m, Device d, Shape (Dim n), TensorScalar a)
    => MTensor (PrimState m) d n a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor d n a
  withDevicePtr iotensor $ \tptr -> do
    runDeviceM (Proxy :: Proxy d)
      $ rawInv tptr $ fromIntegral $ size $ shape tensor

-- | Tensor reshaping.
reshape :: (Shape (Dim n), Shape (Dim k), MonadError t m, Variant t IncompatibleSize)
        => Dim k -> MTensor st d n a -> m (MTensor st d k a)
reshape newshp (MTensor oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwVariant $ incompatibleSize $
    "reshape: couldn't reshape mutable tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ MTensor newshp dataptr

reshape' :: forall n d k st a . (Shape (Dim n), Shape (Dim k))
              => Dim k -> MTensor st d n a -> MTensor st d k a
reshape' newshp tensor =
  case (reshape newshp tensor :: Either IncompatibleSize (MTensor st d k a)) of
  Left err -> throw err
  Right out -> out
