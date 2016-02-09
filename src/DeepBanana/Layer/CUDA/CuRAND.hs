module DeepBanana.Layer.CUDA.CuRAND (
    Generator
  , CuRANDT(..)
  , runCuRANDT
  , uniform
  , CULLong
  , normal
  , logNormal
  , dropout
  ) where

import Control.Monad.ST
import Control.Monad.State
import Foreign.C
import qualified Foreign.CUDA.CuRAND as CuRAND
import Foreign.Marshal
import Foreign.Storable
import System.IO.Unsafe

import DeepBanana.Layer
import DeepBanana.Tensor
import qualified DeepBanana.Tensor.Mutable as MT

newtype Generator = G CuRAND.Generator

type CuRANDT = StateT Generator

runCuRANDT :: (Monad m) => CULLong -> CuRANDT m a -> m a
runCuRANDT seed action =
  evalStateT action $ createGenerator CuRAND.rng_pseudo_default seed

-- The trick to make CuRAND work in pure code is that all the code that updates
-- the internal PRNG state works with this.
unsafeCuRandWrap :: (MonadState Generator m) => (CuRAND.Generator -> IO a) -> m a
unsafeCuRandWrap f = do
  G g <- get
  let res = unsafePerformIO $ f g
  put $ G g
  return res

createGenerator :: CuRAND.RngType -> CULLong -> Generator
createGenerator rngType seed = unsafePerformIO $ do
  g <- alloca $ \genptr -> do
    CuRAND.createGenerator genptr rngType
    peek genptr
  CuRAND.setPseudoRandomGeneratorSeed g seed
  return $ G g

uniform :: (MonadState Generator m, TensorScalar a, Shape (Dim n))
        => Dim n -> m (Tensor n a)
uniform shp = unsafeCuRandWrap $ \gen -> do
  res <- MT.emptyTensor shp
  MT.withDevicePtr res $ \resptr -> do
    generateUniform gen resptr $ fromIntegral $ size shp
  unsafeFreeze res

normal :: (MonadState Generator m, TensorScalar a, Shape (Dim n))
       =>  Dim n -> a -> a -> m (Tensor n a)
normal shp mean std = unsafeCuRandWrap $ \gen -> do
  res <- MT.emptyTensor shp
  MT.withDevicePtr res $ \resptr -> do
    generateNormal gen resptr (fromIntegral $ size shp) mean std
  unsafeFreeze res

logNormal :: (MonadState Generator m, TensorScalar a, Shape (Dim n))
          => Dim n -> a -> a -> m (Tensor n a)
logNormal shp mean std = unsafeCuRandWrap $ \gen -> do
  res <- MT.emptyTensor shp
  MT.withDevicePtr res $ \resptr -> do
    generateLogNormal gen resptr (fromIntegral $ size shp) mean std
  unsafeFreeze res

-- dropout
dropout :: (MonadState Generator m, TensorScalar a, Shape (Dim n))
        => a
        -> Layer m a '[] (Tensor n a) (Tensor n a)
dropout drop_proba = noWeights fwdbwd
  -- Simple dropout algo: generate a random tensor of 0 and 1s,
  -- elementwise multiply with the same random tensor on both forward
  -- and backward pass.
  where fwdbwd inp = do
          pure_mask <- unsafeCuRandWrap $ \gen -> do
            mask <- dropoutMaskIO gen (shape inp) drop_proba
            unsafeFreeze mask
          return $ runST $ do
            return (inp * pure_mask, \upgrad -> upgrad * pure_mask)

-- dropout
-- compute a random mask of ones and zeros
-- to apply elementwise to the input tensor.
dropoutMaskIO :: (TensorScalar a, Shape (Dim n))
              => CuRAND.Generator
              -> Dim n
              -> a
              -> IO (MT.IOTensor n a)
dropoutMaskIO gen shp drop_proba = do
  -- Simple algo for dropout of activations:
  -- 1- generate an array of random values between 0 and 1
  -- 2- threshold that array with the dropout probability
  -- 3- elementwise multiply the input array with it
  rand_array <- MT.emptyTensor shp
  MT.withDevicePtr rand_array $ \randarrayptr -> do
    -- generate random array
    generateUniform gen randarrayptr $ fromIntegral $ size shp
    -- threshold it
    thresh randarrayptr (fromIntegral $ size shp) drop_proba randarrayptr
    return rand_array
