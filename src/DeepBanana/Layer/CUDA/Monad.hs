{-# LANGUAGE UndecidableInstances, RankNTypes, GADTs #-}
module DeepBanana.Layer.CUDA.Monad (
    CudaErrorT
  , CudaError
  , CudaT
  , Cuda
  , CudaExceptions
  , Generator
  , createGenerator
  , CuRAND.RngType
  , CuRAND.rng_test
  , CuRAND.rng_pseudo_default
  , CuRAND.rng_pseudo_xorwow
  , CuRAND.rng_pseudo_mrg32k3a
  , CuRAND.rng_pseudo_mt19937
  , CuRAND.rng_pseudo_philox4_32_10
  , CuRAND.rng_quasi_default
  , CuRAND.rng_quasi_sobol32
  , CuRAND.rng_quasi_scrambled_sobol32
  , CuRAND.rng_quasi_sobol64
  , CuRAND.rng_quasi_scrambled_sobol64
  , MonadCudaError(..)
  , runCudaErrorT
  , runCudaError
  , unsafeRunCudaError
  , cudaErrorHoist
  , MonadCuda(..)
  , unsafeCuRandWrap
  , runCudaT
  , runCudaTEx
  , runCuda
  , runCudaEx
  , cudaHoist
  , unsafeIOToPrim
  , embedCuda
  , embedCudaFromST
  , embedCudaError
  , embedCudaErrorFromST
  ) where

import Control.Lens
import qualified Foreign.CUDA.CuRAND as CuRAND
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad.Primitive (unsafePrimToPrim)
import qualified Control.Monad.State.Strict as SState
import qualified Control.Monad.Writer.Strict as SWriter
import qualified Control.Monad.RWS.Strict as SRWS
import Control.Monad.Error (ErrorT, Error)

import DeepBanana.Prelude
import DeepBanana.Exception
import DeepBanana.Layer.CUDA.Exception
import DeepBanana.Layer.CUDA.CuDNN.Exception
import DeepBanana.Data.Exception
import DeepBanana.Tensor.Exception

type CudaErrorT = ExceptT CudaExceptions

type CudaError = CudaErrorT Identity

type CudaRandT = StateT Generator

type CudaRand = CudaRandT Identity

type CudaT m = CudaErrorT (CudaRandT m)

type Cuda = CudaT Identity

type CudaExceptions = Coproduct '[
                        AllocFailed
                      , BadParam
                      , NotSupported
                      , MemoryAllocation
                      , MappingError
                      , ExecutionFailed
                      , OutOfMemory
                      , IncompatibleShape
                      , IncompatibleSize
                      , EmptyBatch
                      ]

newtype Generator = G CuRAND.Generator

class (MonadError CudaExceptions m)
      => MonadCudaError m
instance (MonadError CudaExceptions m) => MonadCudaError m

runCudaErrorT :: (Monad m)
              => CudaErrorT m a -> m (Either CudaExceptions a)
runCudaErrorT = runExceptT

runCudaError :: CudaError a -> Either CudaExceptions a
runCudaError = runIdentity . runCudaErrorT

unsafeRunCudaError :: CudaError a -> a
unsafeRunCudaError action =
  case runCudaError action of
   Left err -> throw err
   Right res -> res

cudaErrorHoist :: (Monad m, Monad m')
               => (forall a . m a -> m' a) -> CudaErrorT m a -> CudaErrorT m' a
cudaErrorHoist = hoist

class (MonadState Generator m)
      => MonadCudaRand m
instance (MonadState Generator m) => MonadCudaRand m

class (MonadCudaError m, MonadCudaRand m)
      => MonadCuda m
instance (MonadCuda m, MonadCudaRand m) => MonadCuda m

createGenerator :: CuRAND.RngType -> CULLong -> Generator
createGenerator rngType seed = unsafePerformIO $ do
  g <- alloca $ \genptr -> do
    CuRAND.createGenerator genptr rngType
    peek genptr
  CuRAND.setPseudoRandomGeneratorSeed g seed
  return $ G g
  
-- The trick to make CuRAND work in pure code is that all the code that updates
-- the internal PRNG state works with this.
unsafeCuRandWrap :: (MonadCuda m)
                 => (CuRAND.Generator -> IO a) -> m a
unsafeCuRandWrap f = do
  G g <- get
  let res = unsafePerformIO $ f g
  put $ G g
  return res

runCudaT :: (Monad m)
         => Generator -> CudaT m a -> m (Either CudaExceptions a, Generator)
runCudaT gen action = flip runStateT gen $ runExceptT action

runCudaTEx :: (MonadThrow m) => Generator -> CudaT m a -> m (a, Generator)
runCudaTEx seed action = do
  (eres, gen) <- runCudaT seed action
  case eres of
   Left err -> throwM err
   Right res -> return (res, gen)

runCuda :: Generator -> Cuda a -> (Either CudaExceptions a, Generator)
runCuda gen action = runIdentity $ runCudaT gen action

runCudaEx :: Generator -> Cuda a -> (a, Generator)
runCudaEx gen action =
  let (eres, gen) = runCuda gen action
  in (unsafeRunExcept eres, gen)

cudaHoist :: (Monad m, Monad n) => (forall a . m a -> n a) -> CudaT m a -> CudaT n a
cudaHoist morph = hoist (hoist morph)

unsafeIOToPrim :: (PrimMonad m) => IO a -> m a
unsafeIOToPrim = unsafePrimToPrim

embedCuda :: (Monad m, MonadCuda m')
          => (forall a . m a -> m' a) -> CudaT m b -> m' b
embedCuda morph action = do
  gen <- get
  (eres, gen') <- morph $ runCudaT gen action
  put gen'
  case eres of
   Left err -> throwError err
   Right res -> return res

embedCudaFromST :: (MonadCuda m) => (forall s . CudaT (ST s) a) -> m a
embedCudaFromST action = do
  gen <- get
  let (eres, gen') = runST $ runCudaT gen action
  put gen'
  case eres of
   Left err -> throwError err
   Right res -> return res

embedCudaError :: (Monad m, MonadCudaError m')
               => (forall a . m a -> m' a) -> CudaErrorT m a -> m' a
embedCudaError morph action = do
  eres <- morph $ runExceptT action
  case eres of
   Left err -> throwError err
   Right res -> return res

embedCudaErrorFromST :: (MonadCudaError m) => (forall s . CudaErrorT (ST s) a) -> m a
embedCudaErrorFromST action = do
  let eres = runST $ runExceptT action
  case eres of
   Left err -> throwError err
   Right res -> return res
