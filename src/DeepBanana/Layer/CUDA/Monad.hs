module DeepBanana.Layer.CUDA.Monad (
    module Control.Monad.Reader
  , CUDA
  , CUDAReader(..)
  , runCUDA
  ) where

import Foreign.C
import Foreign.Marshal
import Foreign.Storable

import Control.Monad.Reader
import Control.Monad.Except
import qualified Foreign.CUDA.Cublas as Cublas
import qualified Foreign.CUDA.CuDNN as CuDNN
import qualified Foreign.CUDA.CuRAND as CuRAND

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
  eErrRes <- runExceptT $ do
    cublas <- liftIO $ Cublas.create
    cudnn <- liftIO $ createHandle
    curand <- liftIO $ createGenerator CuRAND.rng_pseudo_default
    liftIO $ CuRAND.setPseudoRandomGeneratorSeed curand rngSeed
    runReaderT action $ CUDAReader cublas cudnn curand
  case eErrRes of
   Left err -> throwError $ userError $ err
   Right res -> return res
