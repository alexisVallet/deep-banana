{-# LANGUAGE RankNTypes #-}
module DeepBanana.Layer.CUDA.Monad (
    module Control.Monad.Reader
  , CUDA
  , CUDAReader(..)
  , runCUDA
  , stToCUDA
  , stToPure
  ) where

import Control.Monad.Except
import Control.Monad.Morph
import Control.Monad.Primitive
import Control.Monad.Reader
import Control.Monad.ST
import Foreign.C
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.CuRAND as CuRAND
import Foreign.Marshal
import Foreign.Storable

data CUDAReader = CUDAReader {
  cublasHandle :: Cublas.Handle,
  cudnnHandle :: CuDNN.Handle,
  generator :: CuRAND.Generator
  }

type CUDA = ExceptT String (ReaderT CUDAReader IO)

createHandle :: IO (CuDNN.Handle)
createHandle = alloca $ \hptr -> CuDNN.createHandle hptr >> peek hptr

createGenerator :: CuRAND.RngType -> IO (CuRAND.Generator)
createGenerator rngtype = do
  alloca $ \genptr -> do
    CuRAND.createGenerator genptr rngtype
    peek genptr

-- Actually running the thing.
runCUDA :: CULLong -> CUDA a -> IO a
runCUDA rngSeed action = do
  eErrRes <- do
    cublas <- Cublas.create
    cudnn <- createHandle
    curand <- createGenerator CuRAND.rng_pseudo_default
    CuRAND.setPseudoRandomGeneratorSeed curand rngSeed
    runReaderT (runExceptT action) $ CUDAReader cublas cudnn curand
  case eErrRes of
   Left err -> throwError $ userError $ err
   Right res -> return res

-- Utilities to run ST actions w/ an additional exception monad on top.
stToCUDA :: (forall s . ExceptT String (ST s) a) -> CUDA a
stToCUDA action = do
  case runST $ runExceptT action of
   Left err -> throwError err
   Right out -> return out

stToPure :: (forall s . ExceptT String (ST s) a) -> a
stToPure action =
  case runST $ runExceptT action of
   Left err -> error err
   Right out -> out
