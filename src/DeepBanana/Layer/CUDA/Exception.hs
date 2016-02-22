{-# LANGUAGE GeneralizedNewtypeDeriving, StandaloneDeriving #-}
module DeepBanana.Layer.CUDA.Exception (
    handleCUDAException
  , CUDAException
  , MemoryAllocation
  ) where

import qualified Foreign.CUDA as CUDA

import DeepBanana.Exception
import DeepBanana.Prelude

class (Eq e, Exception e) => CUDAException e where
  cudaException :: Proxy e -> CUDA.CUDAException
  exception :: e

-- Mysteriously missing from the CUDA package
deriving instance Eq CUDA.CUDAException

handleCUDAException :: forall m t e a
                    . (MonadIO m, MonadError t m, Variant t e, CUDAException e)
                    => Proxy e -> IO a -> m a
handleCUDAException p action = do
  let mexcept e = if e == cudaException p then Just e else Nothing
      handle _ = return $ Left $ setVariant (exception :: e)
  eres <- liftIO $ catchJust mexcept (fmap Right action) handle
  embedExcept eres

newtype MemoryAllocation = MemoryAllocation (WithStack CUDA.Status)
                           deriving (Eq, Show, Typeable, Exception)

instance CUDAException MemoryAllocation where
  cudaException _ = CUDA.ExitCode CUDA.MemoryAllocation
  exception = MemoryAllocation $ withStack $ CUDA.MemoryAllocation
