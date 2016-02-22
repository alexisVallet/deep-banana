{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module DeepBanana.Layer.CUDA.CuDNN.Exception (
    BadParam
  , badParam
  , NotInitialized
  , notInitialized
  , AllocFailed
  , allocFailed
  , InternalError
  , internalError
  , InvalidValue
  , invalidValue
  , ArchMismatch
  , archMismatch
  , MappingError
  , mappingError
  , ExecutionFailed
  , executionFailed
  , NotSupported
  , notSupported
  , LicenseError
  , licenseError
  , StatusException
  , handleStatus
  ) where

import qualified Foreign.CUDA.CuDNN as CuDNN
import System.IO.Unsafe

import DeepBanana.Prelude
import DeepBanana.Exception
import DeepBanana.Layer.CUDA.Exception

statusToStackString :: CuDNN.Status -> WithStack String
statusToStackString status =
  withStack $ unsafePerformIO $ CuDNN.getErrorString status >>= peekCString

newtype BadParam = BadParam (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

badParam :: BadParam
badParam = BadParam $ statusToStackString CuDNN.bad_param

newtype NotInitialized = NotInitialized (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

notInitialized :: NotInitialized
notInitialized = NotInitialized $ statusToStackString CuDNN.not_initialized

newtype AllocFailed = AllocFailed (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

allocFailed :: AllocFailed
allocFailed = AllocFailed $ statusToStackString CuDNN.alloc_failed

newtype InternalError = InternalError (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

internalError :: InternalError
internalError = InternalError $ statusToStackString CuDNN.internal_error

newtype InvalidValue = InvalidValue (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

invalidValue :: InvalidValue
invalidValue = InvalidValue $ statusToStackString CuDNN.invalid_value

newtype ArchMismatch = ArchMismatch (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

archMismatch :: ArchMismatch
archMismatch = ArchMismatch $ statusToStackString CuDNN.arch_mismatch

newtype MappingError = MappingError (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

mappingError :: MappingError
mappingError = MappingError $ statusToStackString CuDNN.mapping_error

newtype ExecutionFailed = ExecutionFailed (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

executionFailed :: ExecutionFailed
executionFailed = ExecutionFailed $ statusToStackString CuDNN.execution_failed

newtype NotSupported = NotSupported (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

notSupported :: NotSupported
notSupported = NotSupported $ statusToStackString CuDNN.not_supported

newtype LicenseError = LicenseError (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

licenseError :: LicenseError
licenseError = LicenseError $ statusToStackString CuDNN.license_error

class StatusException e where
  status :: Proxy e -> CuDNN.Status
  exception :: e

instance StatusException BadParam where
  status _ = CuDNN.bad_param
  exception = badParam

instance StatusException NotInitialized where
  status _ = CuDNN.not_initialized
  exception = notInitialized

instance StatusException AllocFailed where
  status _ = CuDNN.alloc_failed
  exception = allocFailed

instance StatusException InternalError where
  status _ = CuDNN.internal_error
  exception = internalError

instance StatusException InvalidValue where
  status _ = CuDNN.invalid_value
  exception = invalidValue

instance StatusException ArchMismatch where
  status _ = CuDNN.arch_mismatch
  exception = archMismatch

instance StatusException MappingError where
  status _ = CuDNN.mapping_error
  exception = mappingError

instance StatusException ExecutionFailed where
  status _ = CuDNN.execution_failed
  exception = executionFailed

instance StatusException NotSupported where
  status _ = CuDNN.not_supported
  exception = notSupported

instance StatusException LicenseError where
  status _ = CuDNN.license_error
  exception = licenseError

handleStatus :: forall t m e
             . (MonadError t m, Variant t e, StatusException e)
             => Proxy e -> m CuDNN.Status -> m CuDNN.Status
handleStatus p action = do
  status' <- action
  when (status' == status p) $ throwVariant (exception :: e)
  return status'
