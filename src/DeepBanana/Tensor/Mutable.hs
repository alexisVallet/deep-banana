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
data MTensor st (shp :: [Nat]) a = MTensor (ForeignPtr a)

-- | Convenient synonym for mutable tensors in the 'IO' monad.
type IOTensor = MTensor RealWorld

-- | Returns the CuDNN datatype identifier for the tensor.
dtype :: forall m s a . (PrimMonad m, TensorScalar a)
      => MTensor (PrimState m) s a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

-- | Utility function to insert shape information at the type level. Useful
-- to desambiguate shape for the compiler when putting an explicit type signature,
-- which happens often in monadic code with mutable tensors.
shaped :: Proxy s -> m (MTensor st s a) -> m (MTensor st s a)
shaped _ mt = mt

-- | Creates an uninitialized mutable tensor with a given shape. Uninitialized means the
-- contents of the tensor are defined by whatever was lying around in memory at the
-- time.
emptyTensor :: forall m s a . (TensorScalar a, Shape s, PrimMonad m)
            => m (MTensor (PrimState m) s a)
emptyTensor = unsafePrimToPrim $ do
  dvcptr <- CUDA.mallocArray $ size (Proxy :: Proxy s)
  let finalizer = CUDA.free dvcptr
  datafptr <- Foreign.Concurrent.newForeignPtr
              (CUDA.useDevicePtr dvcptr)
              finalizer
  return $ MTensor datafptr

-- | Runs an 'IO' action with the internal device pointer of a given mutable tensor.
withDevicePtr :: (Storable a) => IOTensor s a -> (CUDA.DevicePtr a -> IO b) -> IO b
withDevicePtr (MTensor datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)  

-- | Creates a mutable tensor from a list. Throws an exception at runtime when the size
-- of the list does not correspond to the desired shape.
fromList :: forall m s a . (PrimMonad m, TensorScalar a, Shape s)
         =>  [a] -> m (MTensor (PrimState m) s a)
fromList content = unsafePrimToPrim $ do
    when (size (Proxy :: Proxy s) /= length content) $ ioError $ userError
      "Shape is incompatible with provided data."
    withArray content $ \dataptr -> do
      tensor <- emptyTensor :: IO (IOTensor s a)
      withDevicePtr tensor $ \dvcptr -> do
        CUDA.pokeArray (size (Proxy :: Proxy s)) dataptr dvcptr
        return $ unsafeCoerce $ tensor

-- | Converts a mutable tensor to a list of elements.
toList :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
       => MTensor (PrimState m) s a -> m [a]
toList tensor = unsafePrimToPrim $ do
  withDevicePtr (unsafeCoerce tensor :: IOTensor s a)
    $ CUDA.peekListArray (size (Proxy :: Proxy s))

-- | Creates a mutable tensor filled with zeros.
zeros :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
      => m (MTensor (PrimState m) s a)
zeros = fromList $ take (size (Proxy :: Proxy s)) $ repeat 0

-- | Creates a mutable tensor filled with ones.
ones :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
      => m (MTensor (PrimState m) s a)
ones = fromList $ take (size (Proxy :: Proxy s)) $ repeat 1

-- | Copies the data of a mutable tensor into a new one.
copy :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
     => MTensor (PrimState m) s a -> m (MTensor (PrimState m) s a)
copy tensor = unsafePrimToPrim $ do
  out <- emptyTensor :: IO (IOTensor s a)
  withDevicePtr (unsafeCoerce tensor :: IOTensor s a) $ \tensorptr -> do
    withDevicePtr out $ \outptr -> do
      CUDA.copyArray (size (Proxy :: Proxy s)) tensorptr outptr
      return $ unsafeCoerce out

-- | Thresholds a mutable tensor in place. Used to implement dropout.
threshInplace :: forall m s . (PrimMonad m, Shape s)
              => MTensor (PrimState m) s CFloat -> CFloat -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor s a) $ \tensorptr -> do
      thresh tensorptr (fromIntegral $ size (Proxy :: Proxy s)) threshold tensorptr

-- | In place logarithm.
tlog :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
     => MTensor (PrimState m) s a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor s a
  withDevicePtr iotensor $ \tptr -> do
    rawLog tptr (fromIntegral $ size (Proxy :: Proxy s))

-- | In place exponential.
texp :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
     => MTensor (PrimState m) s a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor s a
  withDevicePtr iot $ \tptr -> do
    rawExp tptr (fromIntegral $ size (Proxy :: Proxy s))

-- | In place inverse.
inv :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
    => MTensor (PrimState m) s a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor s a
  withDevicePtr iotensor $ \tptr -> do
    rawInv tptr $ fromIntegral $ size (Proxy :: Proxy s)

-- | Type-safe tensor reshaping.
reshape :: (Shape s1, Shape s2, Size s1 ~ Size s2)
        => MTensor st s1 a -> MTensor st s2 a
reshape (MTensor dataptr) = MTensor dataptr
