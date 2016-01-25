{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-|
Mutable GPU multi-dimensional dense numeric arrays, parametrized by the state token of
some primitive state monad.
-}
module DeepBanana.Tensor.Mutable (
  MTensor(..)
  , Shape(..)
  , IOTensor(..)
  , TensorScalar(..)
  , dtype
  , shaped
  -- * Creating mutable tensors
  , emptyTensor
  , fromList
  , zeros
  , ones
  -- * Basic manipulations
  , withDevicePtr
  , reshape
  , copy
  , toList
  -- * In place numeric functions
  , threshInplace
  , tlog
  , texp
  , inv
  ) where
import Foreign
import Foreign.Marshal
import Foreign.Concurrent
import System.IO.Unsafe
import Data.List
import System.IO.Error
import Control.Monad
import Control.Monad.Except
import Control.Monad.Primitive
import Unsafe.Coerce
import Data.VectorSpace
import Data.Serialize
import Data.Proxy
import Control.DeepSeq
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import GHC.TypeLits

import DeepBanana.Tensor.Shape
import DeepBanana.Tensor.TensorScalar

-- | Mutable tensor, parametrized by some state token 'st'.
data MTensor st (n :: Nat) a = MTensor {
    shape :: Shape n
  , dataptr :: ForeignPtr a
  }

-- | Convenient synonym for mutable tensors in the 'IO' monad.
type IOTensor = MTensor RealWorld

-- | Returns the CuDNN datatype identifier for the tensor.
dtype :: (PrimMonad m, TensorScalar a)
      => MTensor (PrimState m) n a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

-- | Creates an uninitialized mutable tensor with a given shape. Uninitialized means the
-- contents of the tensor are defined by whatever was lying around in memory at the
-- time.
emptyTensor :: (TensorScalar a, PrimMonad m)
            => Shape n -> m (MTensor (PrimState m) n a)
emptyTensor shp = unsafePrimToPrim $ do
  dvcptr <- CUDA.mallocArray $ size shp
  let finalizer = CUDA.free dvcptr
  datafptr <- Foreign.Concurrent.newForeignPtr
              (CUDA.useDevicePtr dvcptr)
              finalizer
  return $ MTensor shp datafptr

-- | Runs an 'IO' action with the internal device pointer of a given mutable tensor.
withDevicePtr :: (Storable a) => IOTensor n a -> (CUDA.DevicePtr a -> IO b) -> IO b
withDevicePtr (MTensor _ datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)  

-- | Creates a mutable tensor from a list. Throws an exception at runtime when the size
-- of the list does not correspond to the desired shape.
fromList :: forall m n a . (PrimMonad m, TensorScalar a)
         =>  Shape n -> [a] -> m (MTensor (PrimState m) n a)
fromList shp content = unsafePrimToPrim $ do
    when (size shp /= length content) $ ioError $ userError
      "Shape is incompatible with provided data."
    withArray content $ \dataptr -> do
      tensor <- emptyTensor shp :: IOTensor n a
      withDevicePtr tensor $ \dvcptr -> do
        CUDA.pokeArray (size shp) dataptr dvcptr
        return $ unsafeCoerce tensor

-- | Converts a mutable tensor to a list of elements.
toList :: forall m n a . (PrimMonad m, TensorScalar a)
       => MTensor (PrimState m) n a -> m [a]
toList tensor = unsafePrimToPrim $ do
  withDevicePtr (unsafeCoerce tensor :: IOTensor n a)
    $ CUDA.peekListArray (size $ shape tensor)

-- | Creates a mutable tensor filled with zeros.
zeros :: (PrimMonad m, TensorScalar a)
      => Shape n -> m (MTensor (PrimState m) n a)
zeros shp = fromList shp $ take (size shp) $ repeat 0

-- | Creates a mutable tensor filled with ones.
ones :: (PrimMonad m, TensorScalar a)
      => Shape n -> m (MTensor (PrimState m) n a)
ones shp = fromList shp $ take (size shp) $ repeat 1

-- | Copies the data of a mutable tensor into a new one.
copy :: forall m n a . (PrimMonad m, TensorScalar a)
     => MTensor (PrimState m) n a -> m (MTensor (PrimState m) n a)
copy tensor = unsafePrimToPrim $ do
  out <- emptyTensor $ shape tensor
  withDevicePtr (unsafeCoerce tensor :: IOTensor n a) $ \tensorptr -> do
    withDevicePtr out $ \outptr -> do
      CUDA.copyArray (size $ shape tensor) tensorptr outptr
      return $ unsafeCoerce out

-- | Thresholds a mutable tensor in place. Used to implement dropout.
threshInplace :: forall m n . (PrimMonad m)
              => MTensor (PrimState m) n CFloat -> CFloat -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor n a) $ \tensorptr -> do
      thresh tensorptr (fromIntegral $ size $ shape tensor) threshold tensorptr

-- | In place logarithm.
tlog :: forall m n a . (PrimMonad m, TensorScalar a)
     => MTensor (PrimState m) n a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor n a
  withDevicePtr iotensor $ \tptr -> do
    rawLog tptr (fromIntegral $ size $ shape tensor)

-- | In place exponential.
texp :: forall m n a . (PrimMonad m, Shape n, TensorScalar a)
     => MTensor (PrimState m) n a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor n a
  withDevicePtr iot $ \tptr -> do
    rawExp tptr (fromIntegral $ size $ shape tensor)

-- | In place inverse.
inv :: forall m n a . (PrimMonad m, TensorScalar a)
    => MTensor (PrimState m) n a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor n a
  withDevicePtr iotensor $ \tptr -> do
    rawInv tptr $ fromIntegral $ size $ shape tensor

-- | Tensor reshaping.
reshape :: (MonadError String m) => Shape k -> MTensor st n a -> m (MTensor st k a)
reshape newshp (MTensor oldshp dataptr) = do
  when (size newshp /= size oldshp) $ throwError $
    "Couldn't reshape mutable tensor from shape " ++ show oldshp ++ " to shape " ++ show newshp ++ ": incompatible sizes " ++ show (size oldshp) ++ " and " ++ show (size newshp)
  return $ MTensor newshp dataptr
