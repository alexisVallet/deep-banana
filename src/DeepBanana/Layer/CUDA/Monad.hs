{-# LANGUAGE UndecidableInstances, RankNTypes, GADTs #-}
module DeepBanana.Layer.CUDA.Monad (
    CudaErrorT
  , CudaError
  , CudaT
  , Cuda
  , CudaExceptions
  , MonadCudaError(..)
  , runCudaErrorT
  , runCudaError
  , unsafeRunCudaError
  , cudaErrorHoist
  , MonadCuda(..)
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
import qualified Control.Monad.State.Strict as SState
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad.Primitive (unsafePrimToPrim)

import DeepBanana.Device
import DeepBanana.Prelude
import DeepBanana.Exception
import DeepBanana.Layer.CUDA.Exception
import DeepBanana.Layer.CUDA.CuDNN.Exception
import DeepBanana.Data.Exception
import DeepBanana.Tensor.Exception

type CudaErrorT = ExceptT CudaExceptions

type CudaError = CudaErrorT Identity

type CudaRandT d = SState.StateT (Generator d)

type CudaRand d = CudaRandT d Identity

type CudaT d m = CudaErrorT (CudaRandT d m)

type Cuda d = CudaT d Identity

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

class (Device d, MonadState (Generator d) m)
      => MonadCudaRand d m
instance (Device d, MonadState (Generator d) m) => MonadCudaRand d m

class (MonadCudaError m, MonadCudaRand d m)
      => MonadCuda d m
instance (MonadCudaError m, MonadCudaRand d m) => MonadCuda d m

runCudaT :: (Monad m)
         => Generator d -> CudaT d m a -> m (Either CudaExceptions a, Generator d)
runCudaT gen action = flip SState.runStateT gen $ runExceptT action

runCudaTEx :: (MonadThrow m) => Generator d -> CudaT d m a -> m (a, Generator d)
runCudaTEx seed action = do
  (eres, gen) <- runCudaT seed action
  case eres of
   Left err -> throwM err
   Right res -> return (res, gen)

runCuda :: Generator d -> Cuda d a -> (Either CudaExceptions a, Generator d)
runCuda gen action = runIdentity $ runCudaT gen action

runCudaEx :: Generator d -> Cuda d a -> (a, Generator d)
runCudaEx gen action =
  let (eres, gen) = runCuda gen action
  in (unsafeRunExcept eres, gen)

cudaHoist :: (Monad m, Monad n) => (forall a . m a -> n a) -> CudaT d m a -> CudaT d n a
cudaHoist morph = hoist (hoist morph)

unsafeIOToPrim :: (PrimMonad m) => IO a -> m a
unsafeIOToPrim = unsafePrimToPrim

embedCuda :: (Monad m, MonadCuda d m')
          => (forall a . m a -> m' a) -> CudaT d m b -> m' b
embedCuda morph action = do
  gen <- get
  (eres, gen') <- morph $ runCudaT gen action
  put gen'
  case eres of
   Left err -> throwError err
   Right res -> return res

embedCudaFromST :: (MonadCuda d m) => (forall s . CudaT d (ST s) a) -> m a
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
