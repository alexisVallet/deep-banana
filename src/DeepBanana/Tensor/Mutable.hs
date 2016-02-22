{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-|
Mutable GPU multi-dimensional dense numeric arrays, parametrized by the state token of
some primitive state monad.
-}
module DeepBanana.Tensor.Mutable (
  MTensor(..)
  , module DeepBanana.Tensor.Shape
  , IOTensor(..)
  , TensorScalar(..)
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
import Foreign.Concurrent
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import Foreign.Marshal
import System.IO.Unsafe
import Unsafe.Coerce

import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Tensor.Exception
import DeepBanana.Tensor.Shape
import DeepBanana.Tensor.TensorScalar

-- | Mutable tensor, parametrized by some state token 'st'.
data MTensor st n a = MTensor {
    shape :: Dim n
  , dataptr :: ForeignPtr a
  }

-- | Convenient synonym for mutable tensors in the 'IO' monad.
type IOTensor = MTensor RealWorld

-- | Returns the CuDNN datatype identifier for the tensor.
dtype :: forall m n a . (PrimMonad m, TensorScalar a)
      => MTensor (PrimState m) n a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

-- | Creates an uninitialized mutable tensor with a given shape. Uninitialized means the
-- contents of the tensor are defined by whatever was lying around in memory at the
-- time.
emptyTensor :: forall n m t a
            . (Shape (Dim n), TensorScalar a, PrimMonad m, MonadError t m,
               Variant t OutOfMemory)
            => Dim n -> m (MTensor (PrimState m) n a)
emptyTensor shp = do
  let action = do
        let handleOutOfMem e = case (e :: CUDA.CUDAException) of
              CUDA.ExitCode CUDA.MemoryAllocation ->
                return $ Left $ outOfMemory $
                "emptyTensor: failed to allocate a tensor of shape " ++ show shp
              err -> error $ "emptyTensor: unhandled exception: " ++ show err
        edvcptr <- liftIO $ (fmap Right $ CUDA.mallocArray (size shp)) `catch` handleOutOfMem
        case edvcptr of
         Left err -> throwError err
         Right dvcptr -> do
           let finalizer = CUDA.free dvcptr
           datafptr <- liftIO $ Foreign.Concurrent.newForeignPtr
                       (CUDA.useDevicePtr dvcptr)
                       finalizer
           return $ MTensor shp datafptr
  eres <- unsafePrimToPrim $ runExceptT $ (action :: ExceptT OutOfMemory IO (MTensor (PrimState m) n a))
  case eres of
   Left err -> throwVariant err
   Right res -> return res

-- | Runs an 'IO' action with the internal device pointer of a given mutable tensor.
withDevicePtr :: (Storable a)
              => IOTensor n a
              -> (CUDA.DevicePtr a -> IO b)
              -> IO b
withDevicePtr (MTensor _ datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)

-- | Creates a mutable tensor from a list. Throws an exception at runtime when the size
-- of the list does not correspond to the desired shape.
tensorFromList :: forall m n a t
          . (PrimMonad m, Shape (Dim n), TensorScalar a, MonadError t m,
             Variant t OutOfMemory, Variant t IncompatibleSize)
          =>  Dim n -> [a] -> m (MTensor (PrimState m) n a)
tensorFromList shp content = do
  when (size shp /= length content) $ throwVariant $ incompatibleSize $
    "tensorFromList: input list and desired shape should have the same size.\nInput list length: " ++ show (length content) ++ "\nDesired shape: " ++ show shp ++ ", size " ++ show (size shp)
  unsafeFromList shp content

unsafeFromList :: forall m n a t
               . (PrimMonad m, Shape (Dim n), TensorScalar a, MonadError t m,
                  Variant t OutOfMemory)
               => Dim n -> [a] -> m (MTensor (PrimState m) n a)
unsafeFromList shp content = do
  tensor <- emptyTensor shp :: m (MTensor (PrimState m) n a)
  unsafePrimToPrim $ do
    withArray content $ \dataptr -> do
      let tensor' = unsafeCoerce tensor :: IOTensor n a
      withDevicePtr tensor' $ \dvcptr -> do
        CUDA.pokeArray (size shp) dataptr dvcptr
  return tensor

-- | Converts a mutable tensor to a list of elements.
tensorToList :: forall m n a . (PrimMonad m, Shape (Dim n), TensorScalar a)
       => MTensor (PrimState m) n a -> m [a]
tensorToList tensor = unsafePrimToPrim $ do
  withDevicePtr (unsafeCoerce tensor :: IOTensor n a)
    $ CUDA.peekListArray (size $ shape tensor)

-- | Creates a mutable tensor filled with zeros.
zeros :: (PrimMonad m, Shape (Dim n), TensorScalar a, MonadError t m,
          Variant t OutOfMemory)
      => Dim n -> m (MTensor (PrimState m) n a)
zeros shp = unsafeFromList shp $ take (size shp) $ repeat 0

-- | Creates a mutable tensor filled with ones.
ones :: (PrimMonad m, Shape (Dim n), TensorScalar a, MonadError t m,
         Variant t OutOfMemory)
      => Dim n -> m (MTensor (PrimState m) n a)
ones shp = unsafeFromList shp $ take (size shp) $ repeat 1

-- | Copies the data of a mutable tensor into a new one.
copy :: forall m n a t
     . (PrimMonad m, Shape (Dim n), TensorScalar a, MonadError t m,
        Variant t OutOfMemory)
     => MTensor (PrimState m) n a -> m (MTensor (PrimState m) n a)
copy tensor = do
  out <- emptyTensor $ shape tensor :: m (MTensor (PrimState m) n a)
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor n a) $ \tensorptr -> do
      withDevicePtr (unsafeCoerce out :: IOTensor n a) $ \outptr -> do
        CUDA.copyArray (size $ shape tensor) tensorptr outptr
  return out

-- | Thresholds a mutable tensor in place. Used to implement dropout.
threshInplace :: forall m n . (PrimMonad m, Shape (Dim n))
              => MTensor (PrimState m) n CFloat -> CFloat -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor n a) $ \tensorptr -> do
      thresh tensorptr (fromIntegral $ size $ shape tensor) threshold tensorptr

-- | In place logarithm.
tlog :: forall m n a . (PrimMonad m, Shape (Dim n), TensorScalar a)
     => MTensor (PrimState m) n a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor n a
  withDevicePtr iotensor $ \tptr -> do
    rawLog tptr (fromIntegral $ size $ shape tensor)

-- | In place exponential.
texp :: forall m n a . (PrimMonad m, Shape (Dim n), TensorScalar a)
     => MTensor (PrimState m) n a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor n a
  withDevicePtr iot $ \tptr -> do
    rawExp tptr (fromIntegral $ size $ shape t)

-- | In place inverse.
inv :: forall m n a . (PrimMonad m, Shape (Dim n), TensorScalar a)
    => MTensor (PrimState m) n a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor n a
  withDevicePtr iotensor $ \tptr -> do
    rawInv tptr $ fromIntegral $ size $ shape tensor

-- | Tensor reshaping.
reshape :: (Shape (Dim n), Shape (Dim k), MonadError t m, Variant t IncompatibleSize)
        => Dim k -> MTensor st n a -> m (MTensor st k a)
reshape newshp (MTensor oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwVariant $ incompatibleSize $
    "reshape: couldn't reshape mutable tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ MTensor newshp dataptr

reshape' :: forall n k st a . (Shape (Dim n), Shape (Dim k))
              => Dim k -> MTensor st n a -> MTensor st k a
reshape' newshp tensor =
  case (reshape newshp tensor :: Either IncompatibleSize (MTensor st k a)) of
  Left err -> throw err
  Right out -> out
