{-# LANGUAGE UndecidableInstances #-}
module DeepBanana.Tensor.Mutable (
  MTensor(..)
  , Shape(..)
  , IOTensor(..)
  , TensorScalar(..)
  , dtype
  , shaped
  , emptyTensor
  , withDevicePtr
  , fromList
  , toList
  , zeros
  , ones
  , copy
  , threshInplace
  , tlog
  , texp
  , inv
  , reshape
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

-- mutable tensor with type-level fixed shape.
data MTensor st (shp :: [Nat]) a = MTensor (ForeignPtr a)

type IOTensor = MTensor RealWorld

dtype :: forall m s a . (PrimMonad m, TensorScalar a)
      => MTensor (PrimState m) s a
      -> m CuDNN.DataType
dtype _ = return $ datatype (Proxy :: Proxy a)

shaped :: Proxy s -> m (MTensor st s a) -> m (MTensor st s a)
shaped _ mt = mt

emptyTensor :: forall m s a . (TensorScalar a, Shape s, PrimMonad m)
            => m (MTensor (PrimState m) s a)
emptyTensor = unsafePrimToPrim $ do
  dvcptr <- CUDA.mallocArray $ size (Proxy :: Proxy s)
  let finalizer = CUDA.free dvcptr
  datafptr <- Foreign.Concurrent.newForeignPtr
              (CUDA.useDevicePtr dvcptr)
              finalizer
  return $ MTensor datafptr

withDevicePtr :: (Storable a) => IOTensor s a -> (CUDA.DevicePtr a -> IO b) -> IO b
withDevicePtr (MTensor datafptr) action = do
  withForeignPtr datafptr $ \dataptr -> action (CUDA.DevicePtr dataptr)  

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

zeros :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
      => m (MTensor (PrimState m) s a)
zeros = fromList $ take (size (Proxy :: Proxy s)) $ repeat 0

ones :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
      => m (MTensor (PrimState m) s a)
ones = fromList $ take (size (Proxy :: Proxy s)) $ repeat 1

toList :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
       => MTensor (PrimState m) s a -> m [a]
toList tensor = unsafePrimToPrim $ do
  withDevicePtr (unsafeCoerce tensor :: IOTensor s a)
    $ CUDA.peekListArray (size (Proxy :: Proxy s))

copy :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
     => MTensor (PrimState m) s a -> m (MTensor (PrimState m) s a)
copy tensor = unsafePrimToPrim $ do
  out <- emptyTensor :: IO (IOTensor s a)
  withDevicePtr (unsafeCoerce tensor :: IOTensor s a) $ \tensorptr -> do
    withDevicePtr out $ \outptr -> do
      CUDA.copyArray (size (Proxy :: Proxy s)) tensorptr outptr
      return $ unsafeCoerce out

threshInplace :: forall m s . (PrimMonad m, Shape s)
              => MTensor (PrimState m) s CFloat -> CFloat -> m ()
threshInplace tensor threshold =
  unsafePrimToPrim $ do
    withDevicePtr (unsafeCoerce tensor :: IOTensor s a) $ \tensorptr -> do
      thresh tensorptr (fromIntegral $ size (Proxy :: Proxy s)) threshold tensorptr

tlog :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
     => MTensor (PrimState m) s a -> m ()
tlog tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor s a
  withDevicePtr iotensor $ \tptr -> do
    rawLog tptr (fromIntegral $ size (Proxy :: Proxy s))

texp :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
     => MTensor (PrimState m) s a -> m ()
texp t = unsafePrimToPrim $ do
  let iot = unsafeCoerce t :: IOTensor s a
  withDevicePtr iot $ \tptr -> do
    rawExp tptr (fromIntegral $ size (Proxy :: Proxy s))

inv :: forall m s a . (PrimMonad m, Shape s, TensorScalar a)
    => MTensor (PrimState m) s a -> m ()
inv tensor = unsafePrimToPrim $ do
  let iotensor = unsafeCoerce tensor :: IOTensor s a
  withDevicePtr iotensor $ \tptr -> do
    rawInv tptr $ fromIntegral $ size (Proxy :: Proxy s)

reshape :: (Shape s1, Shape s2, Size s1 ~ Size s2)
        => MTensor st s1 a -> MTensor st s2 a
reshape = unsafeCoerce
