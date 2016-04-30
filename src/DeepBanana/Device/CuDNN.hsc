{-|
FFI wrapper for CuDNN. Naming simply removes the cudnn or cudnn_ prefixes of wrapped functions. Pointers referring to device arrays are wrapped as @'DevicePtr'@, while those referring to host arrays as regular @'Ptr'@. Output types are wrapped as @'DeviceM'@ to ensure they are executed with a specific device. See <https://developer.nvidia.com/cudnn CuDNN's documentation> for documentation of each individual function.
|-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module DeepBanana.Device.CuDNN where

import Data.MemoTrie
import Foreign
import Foreign.C
import Foreign.CUDA.Types
import System.IO.Unsafe

import DeepBanana.Device.Monad
import DeepBanana.Prelude hiding (Handle)

#include <cudnn.h>

-- Version number.
foreign import ccall safe "cudnnGetVersion"
  getVersion :: IO CSize

-- CuDNN return codes.
newtype Status = Status {
  unStatus :: CInt
  } deriving (Show, Eq, Storable)

#{enum Status, Status
 , success = CUDNN_STATUS_SUCCESS
 , not_initialized = CUDNN_STATUS_NOT_INITIALIZED
 , alloc_failed = CUDNN_STATUS_ALLOC_FAILED
 , bad_param = CUDNN_STATUS_BAD_PARAM
 , internal_error = CUDNN_STATUS_INTERNAL_ERROR
 , invalid_value = CUDNN_STATUS_INVALID_VALUE
 , arch_mismatch = CUDNN_STATUS_ARCH_MISMATCH
 , mapping_error = CUDNN_STATUS_MAPPING_ERROR
 , execution_failed = CUDNN_STATUS_EXECUTION_FAILED
 , not_supported = CUDNN_STATUS_NOT_SUPPORTED
 , license_error = CUDNN_STATUS_LICENSE_ERROR
 }

foreign import ccall safe "cudnnGetErrorString"
  getErrorString :: Status -> IO CString

-- Initializing and destroying CuDNN handles.

-- CuDNN handle is an opaque structure.
newtype Handle d = Handle {
  unHandle :: Ptr ()
  } deriving Storable

foreign import ccall safe "cudnnCreate"
  createHandle :: Ptr (Ptr ()) -> IO Status

foreign import ccall safe "cudnnDestroy"
  destroyHandle :: Ptr () -> IO Status

getRawHandle :: Integer -> Ptr ()
{-# NOINLINE getRawHandle #-}
getRawHandle = memo $ \p -> unsafePerformIO $ do
  h <- alloca $ \hptr -> do
    createHandle hptr
    peek hptr
  return h

-- | Returns a CuDNN handle for any specific device. This function is memoized.
handle :: (Device d) => d -> Handle d
handle dev = Handle $ getRawHandle $ fromIntegral $ deviceId dev

-- Getting and setting stream.
foreign import ccall safe "cudnnSetStream"
  setStream :: Handle d -> Stream -> DeviceM d Status

foreign import ccall safe "cudnnGetStream"
  getStream :: Handle d -> Ptr Stream -> DeviceM d Status

-- Data structures for tensors, convolutions, poolings and filters
-- are also opaque.
newtype TensorDescriptor = TensorDescriptor {
  unTensorDescriptor :: Ptr ()
  } deriving Storable
newtype ConvolutionDescriptor = ConvolutionDescriptor {
  unConvolutionDescriptor :: Ptr ()
  } deriving Storable
newtype PoolingDescriptor = PoolingDescriptor {
  unPoolingDescriptor :: Ptr ()
  } deriving Storable
newtype FilterDescriptor = FilterDescriptor {
  unFilterDescriptor :: Ptr ()
  } deriving Storable

-- Floating point datatypes.
newtype DataType = DataType {
  unDataType :: CInt
  } deriving (Show, Eq, Storable)

#{enum DataType, DataType
 , float = CUDNN_DATA_FLOAT
 , double = CUDNN_DATA_DOUBLE
 }

-- Generic tensor descriptor initialization.
foreign import ccall safe "cudnnCreateTensorDescriptor"
  createTensorDescriptor :: Ptr TensorDescriptor -> DeviceM d Status

-- Tensor format.
newtype TensorFormat = TensorFormat {
  unTensorFormat :: CInt
  } deriving (Show, Eq, Storable)

#{enum TensorFormat, TensorFormat
 , nchw = CUDNN_TENSOR_NCHW
 , nhwc = CUDNN_TENSOR_NHWC
 }

-- 4d tensor descriptors.
foreign import ccall safe "cudnnSetTensor4dDescriptor"
  setTensor4dDescriptor :: TensorDescriptor
                        -> TensorFormat
                        -> DataType
                        -> CInt -- n, batch size
                        -> CInt -- c, number of input feature maps
                        -> CInt -- h, height or rows
                        -> CInt -- w, width or columns
                        -> DeviceM d Status

foreign import ccall safe "cudnnSetTensor4dDescriptorEx"
  setTensor4dDescriptorEx :: TensorDescriptor
                          -> DataType
                          -> CInt -- n, batch size
                          -> CInt -- c, number of input feature maps
                          -> CInt -- h, height or rows
                          -> CInt -- w, width or columns
                          -> CInt -- nStride
                          -> CInt -- cStride
                          -> CInt -- hStride
                          -> CInt -- wStride
                          -> DeviceM d Status

foreign import ccall safe "cudnnGetTensor4dDescriptor"
  getTensor4dDescriptor :: TensorDescriptor
                        -> Ptr DataType
                        -> Ptr CInt -- n
                        -> Ptr CInt -- c
                        -> Ptr CInt -- h
                        -> Ptr CInt -- w
                        -> Ptr CInt -- nStride
                        -> Ptr CInt -- cStride
                        -> Ptr CInt -- hStride
                        -> Ptr CInt -- wStride
                        -> DeviceM d Status

foreign import ccall safe "cudnnSetTensorNdDescriptor"
  setTensorNdDescriptor :: TensorDescriptor
                        -> DataType
                        -> CInt -- nbDims
                        -> Ptr CInt -- dimensions array
                        -> Ptr CInt -- strides array
                        -> DeviceM d Status

foreign import ccall safe "cudnnGetTensorNdDescriptor"
  getTensorNdDescriptor :: TensorDescriptor
                        -> CInt -- nbDimsRequested
                        -> Ptr DataType
                        -> Ptr CInt -- nbDims
                        -> Ptr CInt -- dimensions array
                        -> Ptr CInt -- strides array
                        -> DeviceM d Status

foreign import ccall safe "cudnnDestroyTensorDescriptor"
  destroyTensorDescriptor :: TensorDescriptor -> DeviceM d Status

-- Apparently a tensor layout conversion helper?
foreign import ccall safe "cudnnTransformTensor"
  transformTensor :: Handle d
                  -> Ptr a -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> DevicePtr a -- srcData
                  -> Ptr a -- beta
                  -> TensorDescriptor -- destDesc
                  -> DevicePtr a -- destData
                  -> DeviceM d Status

-- Tensor in place bias addition.
newtype AddMode = AddMode {
  unAddMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum AddMode, AddMode
 , add_image = CUDNN_ADD_IMAGE
 , add_same_hw = CUDNN_ADD_SAME_HW
 , add_feature_map = CUDNN_ADD_FEATURE_MAP
 , add_same_chw = CUDNN_ADD_SAME_CHW
 , add_same_c = CUDNN_ADD_SAME_C
 , add_full_tensor = CUDNN_ADD_FULL_TENSOR
 }

foreign import ccall safe "cudnnAddTensor"
  addTensor :: Handle d
            -> Ptr a -- alpha
            -> TensorDescriptor -- biasDesc
            -> DevicePtr a -- biasData
            -> Ptr a -- beta
            -> TensorDescriptor -- srcDestDesc
            -> DevicePtr a -- srcDestData
            -> DeviceM d Status

-- Fills tensor with value.
foreign import ccall safe "cudnnSetTensor"
  setTensor :: Handle d
            -> TensorDescriptor -- srcDestDesc
            -> DevicePtr a -- srcDestData
            -> DevicePtr a -- value
            -> DeviceM d Status

-- Convolution mode.
newtype ConvolutionMode = ConvolutionMode {
  unConvolutionMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum ConvolutionMode, ConvolutionMode
 , convolution = CUDNN_CONVOLUTION
 , cross_correlation = CUDNN_CROSS_CORRELATION
 }

-- Filter struct manipulation.
foreign import ccall safe "cudnnCreateFilterDescriptor"
  createFilterDescriptor :: Ptr FilterDescriptor -> DeviceM d Status

foreign import ccall safe "cudnnSetFilter4dDescriptor"
  setFilter4dDescriptor :: FilterDescriptor
                        -> DataType
                        -> CInt -- k number of filters
                        -> CInt -- c number of input channels
                        -> CInt -- h height (number of rows) of each filter
                        -> CInt -- w width (number of columns) of each filter
                        -> DeviceM d Status

foreign import ccall safe "cudnnGetFilter4dDescriptor"
  getFilter4dDescriptor :: FilterDescriptor
                        -> Ptr DataType
                        -> Ptr CInt -- k number of filters
                        -> Ptr CInt -- c number of input channels
                        -> Ptr CInt -- h height (number of rows) of each filter
                        -> Ptr CInt -- w width (number of columns) of each filter
                        -> DeviceM d Status

foreign import ccall safe "cudnnSetFilterNdDescriptor"
  setFilterNdDescriptor :: FilterDescriptor
                        -> DataType
                        -> CInt -- nbdims
                        -> Ptr CInt -- filter tensor dimensions array
                        -> DeviceM d Status

foreign import ccall safe "cudnnGetFilterNdDescriptor"
  getFilterNdDescriptor :: FilterDescriptor
                        -> CInt -- number of requested dimensions
                        -> Ptr DataType
                        -> Ptr CInt -- nbdims
                        -> Ptr CInt -- filter tensor dimensions array
                        -> DeviceM d Status

foreign import ccall safe "cudnnDestroyFilterDescriptor"
  destroyFilterDescriptor :: FilterDescriptor -> DeviceM d Status

-- Convolution descriptor manipulations.
foreign import ccall safe "cudnnCreateConvolutionDescriptor"
  createConvolutionDescriptor :: Ptr ConvolutionDescriptor
                              -> DeviceM d Status

foreign import ccall safe "cudnnSetConvolution2dDescriptor"
  setConvolution2dDescriptor :: ConvolutionDescriptor
                             -> CInt -- pad_h
                             -> CInt -- pad_w
                             -> CInt -- u vertical stride
                             -> CInt -- v horizontal stride
                             -> CInt -- upscalex
                             -> CInt -- upscaley
                             -> ConvolutionMode
                             -> DeviceM d Status

foreign import ccall safe "cudnnGetConvolution2dDescriptor"
  getConvolution2dDescriptor :: ConvolutionDescriptor
                             -> Ptr CInt -- pad_h
                             -> Ptr CInt -- pad_w
                             -> Ptr CInt -- u
                             -> Ptr CInt -- v
                             -> Ptr CInt -- upscalex
                             -> Ptr CInt -- upscaley
                             -> Ptr ConvolutionMode
                             -> DeviceM d Status

foreign import ccall safe "cudnnGetConvolution2dForwardOutputDim"
  getConvolution2dForwardOutputDim :: ConvolutionDescriptor
                                   -> TensorDescriptor
                                   -> FilterDescriptor
                                   -> Ptr CInt -- n
                                   -> Ptr CInt -- c
                                   -> Ptr CInt -- h
                                   -> Ptr CInt -- w
                                   -> DeviceM d Status

foreign import ccall safe "cudnnSetConvolutionNdDescriptor"
  setConvolutionNdDescriptor :: ConvolutionDescriptor
                             -> CInt -- array length nbDims-2 size
                             -> Ptr CInt -- paddings array
                             -> Ptr CInt -- filter strides array
                             -> Ptr CInt -- upscales array
                             -> ConvolutionMode
                             -> DeviceM d Status

foreign import ccall safe "cudnnGetConvolutionNdDescriptor"
  getConvolutionNdDescriptor :: ConvolutionDescriptor
                             -> CInt -- requested array length
                             -> Ptr CInt -- array length
                             -> Ptr CInt -- paddings array
                             -> Ptr CInt -- strides array
                             -> Ptr CInt -- upscales array
                             -> Ptr ConvolutionMode
                             -> DeviceM d Status

foreign import ccall safe "cudnnGetConvolutionNdForwardOutputDim"
  getConvolutionNdForwardOutputDim :: ConvolutionDescriptor
                                   -> TensorDescriptor
                                   -> FilterDescriptor
                                   -> CInt -- nbDims
                                   -> Ptr CInt -- tensor output dims
                                   -> DeviceM d Status

foreign import ccall safe "cudnnDestroyConvolutionDescriptor"
  destroyConvolutionDescriptor :: ConvolutionDescriptor
                               -> DeviceM d Status

newtype ConvolutionFwdPreference = ConvolutionFwdPreference {
  unConvolutionForwardPreference :: CInt
  } deriving (Show, Eq, Storable)

#{enum ConvolutionFwdPreference, ConvolutionFwdPreference
 , convolution_fwd_no_workspace = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
 , convolution_fwd_prefer_fastest = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, convolution_fwd_specify_workspace_limit = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
 }

newtype ConvolutionFwdAlgo = ConvolutionFwdAlgo {
  unConvolutionFwdAlgo :: CInt
  } deriving (Show, Eq, Storable)

#{enum ConvolutionFwdAlgo, ConvolutionFwdAlgo
 , convolution_fwd_algo_implicit_gemm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
 , convolution_fwd_algo_implicit_precomp_gemm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
 , convolution_fwd_algo_gemm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM
 , convolution_fwd_algo_direct = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
 , convolution_fwd_algo_fft = CUDNN_CONVOLUTION_FWD_ALGO_FFT
 , convolution_fwd_algo_fft_tiling = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
 }

foreign import ccall safe "cudnnGetConvolutionForwardAlgorithm"
  getConvolutionForwardAlgorithm :: Handle d
                                 -> TensorDescriptor -- srcDesc
                                 -> FilterDescriptor
                                 -> ConvolutionDescriptor
                                 -> TensorDescriptor -- destDesc
                                 -> ConvolutionFwdPreference
                                 -> CSize -- memory limit in bytes
                                 -> Ptr ConvolutionFwdAlgo
                                 -> DeviceM d Status

foreign import ccall safe "cudnnGetConvolutionForwardWorkspaceSize"
  getConvolutionForwardWorkspaceSize :: Handle d
                                     -> TensorDescriptor -- srcDesc
                                     -> FilterDescriptor
                                     -> ConvolutionDescriptor
                                     -> TensorDescriptor -- dstDesc
                                     -> ConvolutionFwdAlgo
                                     -> Ptr CSize -- size in bytes
                                     -> DeviceM d Status

foreign import ccall safe "cudnnConvolutionForward"
  convolutionForward :: Handle d
                     -> Ptr a -- alpha
                     -> TensorDescriptor -- srcDesc
                     -> DevicePtr a -- srcData
                     -> FilterDescriptor
                     -> DevicePtr a -- filterData
                     -> ConvolutionDescriptor
                     -> ConvolutionFwdAlgo
                     -> DevicePtr Int8 -- workspace
                     -> CSize -- workspace size in bytes
                     -> Ptr a -- beta
                     -> TensorDescriptor -- destDesc
                     -> DevicePtr a -- destData
                     -> DeviceM d Status

-- Convolution gradient with regards to the bias.
foreign import ccall safe "cudnnConvolutionBackwardBias"
  convolutionBackwardBias :: Handle d
                          -> Ptr a -- alpha
                          -> TensorDescriptor -- srcDesc
                          -> DevicePtr a -- srcData
                          -> Ptr a -- beta
                          -> TensorDescriptor -- destDesc
                          -> DevicePtr a -- destData
                          -> DeviceM d Status

newtype ConvolutionBwdFilterAlgo = ConvolutionBwdFilterAlgo {
  unConvolutionBwdFilterAlgo :: CInt
  } deriving (Eq, Ord, Show, Storable)

#{enum ConvolutionBwdFilterAlgo, ConvolutionBwdFilterAlgo
 , convolution_bwd_filter_algo_0 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
 , convolution_bwd_filter_algo_1 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
 , convolution_bwd_filter_algo_fft = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
 , convolution_bwd_filter_algo_3 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3}

foreign import ccall safe "cudnnGetConvolutionBackwardFilterWorkspaceSize"
  getConvolutionBackwardFilterWorkspaceSize :: Handle d
                                            -> TensorDescriptor
                                            -> TensorDescriptor
                                            -> ConvolutionDescriptor
                                            -> FilterDescriptor
                                            -> ConvolutionBwdFilterAlgo
                                            -> Ptr CSize
                                            -> DeviceM d Status

-- Computes gradient with regards to the filters.
foreign import ccall safe "cudnnConvolutionBackwardFilter"
  convolutionBackwardFilter :: Handle d
                            -> Ptr a -- alpha
                            -> TensorDescriptor -- srcDesc
                            -> DevicePtr a -- srcData
                            -> TensorDescriptor -- diffDesc
                            -> DevicePtr a -- diffData
                            -> ConvolutionDescriptor
                            -> ConvolutionBwdFilterAlgo
                            -> DevicePtr Int8
                            -> CSize
                            -> Ptr a -- beta
                            -> FilterDescriptor -- gradDesc
                            -> DevicePtr a -- gradData
                            -> DeviceM d Status

newtype ConvolutionBwdDataAlgo = ConvolutionBwdDataAlgo {
  unConvoltuionBwdDataAlgo :: CInt
  } deriving (Eq, Ord, Show, Storable)

#{enum ConvolutionBwdDataAlgo, ConvolutionBwdDataAlgo
 , convolution_bwd_data_algo_0 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
 , convolution_bwd_data_algo_1 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
 , convolution_bwd_data_algo_fft = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
 , convolution_bwd_data_algo_fft_tiling = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
  }


foreign import ccall safe "cudnnGetConvolutionBackwardDataWorkspaceSize"
  getConvolutionBackwardDataWorkspaceSize :: Handle d
                                       -> FilterDescriptor
                                       -> TensorDescriptor
                                       -> ConvolutionDescriptor
                                       -> TensorDescriptor
                                       -> ConvolutionBwdDataAlgo
                                       -> Ptr CSize
                                       -> DeviceM d Status
                                       
-- Computes gradient with regards to the data.
foreign import ccall safe "cudnnConvolutionBackwardData"
  convolutionBackwardData :: Handle d
                          -> Ptr a -- alpha
                          -> FilterDescriptor
                          -> DevicePtr a -- filterData
                          -> TensorDescriptor -- diffDesc
                          -> DevicePtr a -- diffData
                          -> ConvolutionDescriptor
                          -> ConvolutionBwdDataAlgo
                          -> DevicePtr Int8 -- workspace
                          -> CSize -- workspace size
                          -> Ptr a -- beta
                          -> TensorDescriptor -- gradDesc
                          -> DevicePtr a -- gradData
                          -> DeviceM d Status

foreign import ccall safe "cudnnIm2Col"
  im2Col :: Handle d
         -> Ptr a -- alpha
         -> TensorDescriptor -- srcDesc
         -> DevicePtr a -- srcData
         -> FilterDescriptor
         -> ConvolutionDescriptor
         -> DevicePtr a -- colBuffer
         -> DeviceM d Status

-- Softmax
newtype SoftmaxAlgorithm = SoftmaxAlgorithm {
  unSoftmaxAlgorithm :: CInt
  } deriving (Show, Eq, Storable)

#{enum SoftmaxAlgorithm, SoftmaxAlgorithm
 , softmax_fast = CUDNN_SOFTMAX_FAST
 , softmax_accurate = CUDNN_SOFTMAX_ACCURATE
 }

newtype SoftmaxMode = SoftmaxMode {
  unSoftmaxMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum SoftmaxMode, SoftmaxMode
 , softmax_mode_instance = CUDNN_SOFTMAX_MODE_INSTANCE
 , softmax_mode_channel = CUDNN_SOFTMAX_MODE_CHANNEL
 }

foreign import ccall safe "cudnnSoftmaxForward"
  softmaxForward :: Handle d
                 -> SoftmaxAlgorithm
                 -> SoftmaxMode
                 -> Ptr a -- alpha
                 -> TensorDescriptor -- srcDesc
                 -> DevicePtr a -- srcData
                 -> Ptr a -- beta
                 -> TensorDescriptor -- destDesc
                 -> DevicePtr a -- destData
                 -> DeviceM d Status

foreign import ccall safe "cudnnSoftmaxBackward"
  softmaxBackward :: Handle d
                  -> SoftmaxAlgorithm
                  -> SoftmaxMode
                  -> Ptr a -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> DevicePtr a -- srcData
                  -> TensorDescriptor -- srcDiffDesc
                  -> DevicePtr a  -- srcDiffData
                  -> Ptr a -- beta
                  -> TensorDescriptor -- destDiffDesc
                  -> DevicePtr a -- destDiffData
                  -> DeviceM d Status

-- Pooling.
newtype PoolingMode = PoolingMode {
  unPoolingMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum PoolingMode, PoolingMode
 , pooling_max = CUDNN_POOLING_MAX
 , pooling_average_count_include_padding = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
 , pooling_average_count_exclude_padding = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
 }

foreign import ccall safe "cudnnCreatePoolingDescriptor"
  createPoolingDescriptor :: Ptr PoolingDescriptor -> DeviceM d Status

foreign import ccall safe "cudnnSetPooling2dDescriptor"
  setPooling2dDescriptor :: PoolingDescriptor
                         -> PoolingMode
                         -> CInt -- window height
                         -> CInt -- window width
                         -> CInt -- vertical padding
                         -> CInt -- horizontal padding
                         -> CInt -- vertical stride
                         -> CInt -- horizontal stride
                         -> DeviceM d Status

foreign import ccall safe "cudnnGetPooling2dDescriptor"
  getPooling2dDescriptor :: PoolingDescriptor
                         -> Ptr PoolingMode
                         -> Ptr CInt -- window height
                         -> Ptr CInt -- window width
                         -> Ptr CInt -- vertical padding
                         -> Ptr CInt -- horizontal padding
                         -> Ptr CInt -- vertical stride
                         -> Ptr CInt -- horizontal stride
                         -> DeviceM d Status

foreign import ccall safe "cudnnSetPoolingNdDescriptor"
  setPoolingNdDescriptor :: PoolingDescriptor
                         -> PoolingMode
                         -> CInt -- nbDims
                         -> Ptr CInt -- window dimensions array
                         -> Ptr CInt -- paddings array
                         -> Ptr CInt -- strides array
                         -> DeviceM d Status

foreign import ccall safe "cudnnGetPoolingNdDescriptor"
  getPoolingNdDescriptor :: PoolingDescriptor
                         -> CInt -- nbDimsRequested
                         -> Ptr PoolingMode
                         -> Ptr CInt -- nbDims
                         -> Ptr CInt -- window dimensons array
                         -> Ptr CInt -- paddings array
                         -> Ptr CInt -- strides array
                         -> DeviceM d Status

-- These 2 functions, although they have headers in cudnn.h, do not
-- have corresponding symbols in libcudnn.so. Bug in CuDNN?
-- foreign import ccall safe "cudnnGetPoolingNdForwardOutputDim"
--   getPoolingNdForwardOutputDim :: PoolingDescriptor
--                                -> TensorDescriptor
--                                -> CInt -- nbDims
--                                -> Ptr CInt -- output tensor dimensions
--                                -> DeviceM d Status

-- foreign import ccall safe "cudnnGetPooling2dForwardOutputDim"
--   getPooling2dForwardOutputDim :: PoolingDescriptor
--                                -> TensorDescriptor
--                                -> Ptr CInt -- outN
--                                -> Ptr CInt -- outC
--                                -> Ptr CInt -- outH
--                                -> Ptr CInt -- outW
--                                -> DeviceM d Status

foreign import ccall safe "cudnnDestroyPoolingDescriptor"
  destroyPoolingDescriptor :: PoolingDescriptor -> DeviceM d Status

foreign import ccall safe "cudnnPoolingForward"
  poolingForward :: Handle d
                 -> PoolingDescriptor
                 -> Ptr a -- alpha
                 -> TensorDescriptor -- srcDesc
                 -> DevicePtr a -- srcData
                 -> Ptr a -- beta
                 -> TensorDescriptor -- destDesc
                 -> DevicePtr a -- destData
                 -> DeviceM d Status

foreign import ccall safe "cudnnPoolingBackward"
  poolingBackward :: Handle d
                  -> PoolingDescriptor
                  -> Ptr a -- alpha
                  -> TensorDescriptor -- srcDesc
                  -> DevicePtr a -- srcData
                  -> TensorDescriptor -- srcDiffDesc
                  -> DevicePtr a -- srcDiffData
                  -> TensorDescriptor -- destDesc
                  -> DevicePtr a -- destData
                  -> Ptr a -- beta
                  -> TensorDescriptor -- destDiffDesc
                  -> DevicePtr a -- destDiffData
                  -> DeviceM d Status

-- Activation functions.
newtype ActivationMode = ActivationMode {
  unActivationMode :: CInt
  } deriving (Show, Eq, Storable)

#{enum ActivationMode, ActivationMode
 , activation_sigmoid = CUDNN_ACTIVATION_SIGMOID
 , activation_relu = CUDNN_ACTIVATION_RELU
 , activation_tanh = CUDNN_ACTIVATION_TANH
 }

foreign import ccall "cudnnActivationForward"
  activationForward :: Handle d
                    -> ActivationMode
                    -> Ptr a -- alpha
                    -> TensorDescriptor -- srcDesc
                    -> DevicePtr a -- srcData
                    -> Ptr a -- beta
                    -> TensorDescriptor -- destDesc
                    -> DevicePtr a -- destData
                    -> DeviceM d Status

foreign import ccall "cudnnActivationBackward"
  activationBackward :: Handle d
                     -> ActivationMode
                     -> Ptr a -- alpha
                     -> TensorDescriptor -- srcDesc
                     -> DevicePtr a -- srcData
                     -> TensorDescriptor -- srcDiffDesc
                     -> DevicePtr a -- srcDiffData
                     -> TensorDescriptor -- destDesc
                     -> DevicePtr a -- destData
                     -> Ptr a --beta
                     -> TensorDescriptor -- destDiffDesc
                     -> DevicePtr a -- destDiffData
                     -> DeviceM d Status

-- Batch normalization
newtype BatchNormMode = BatchNormMode {
  unBatchNormMode :: CInt
  } deriving (Eq, Ord, Show, Storable)

#{enum BatchNormMode, BatchNormMode
 , batchnorm_per_activation = CUDNN_BATCHNORM_PER_ACTIVATION
 , batchnorm_spatial = CUDNN_BATCHNORM_SPATIAL
 }

foreign import ccall safe "cudnnBatchNormalizationForwardInference"
  batchNormalizationForwardInference :: Handle d
                                     -> BatchNormMode
                                     -> Ptr a
                                     -> Ptr a
                                     -> TensorDescriptor
                                     -> DevicePtr a
                                     -> TensorDescriptor
                                     -> DevicePtr a
                                     -> TensorDescriptor
                                     -> DevicePtr a
                                     -> DevicePtr a
                                     -> DevicePtr a
                                     -> DevicePtr a
                                     -> CDouble
                                     -> DeviceM d Status

foreign import ccall safe "cudnnBatchNormalizationForwardTraining"
  batchNormalizationForwardTraining :: Handle d
                                    -> BatchNormMode
                                    -> Ptr a
                                    -> Ptr a
                                    -> TensorDescriptor
                                    -> DevicePtr a
                                    -> TensorDescriptor
                                    -> DevicePtr a
                                    -> TensorDescriptor
                                    -> DevicePtr a
                                    -> DevicePtr a
                                    -> CDouble
                                    -> DevicePtr a
                                    -> DevicePtr a
                                    -> CDouble
                                    -> DevicePtr a
                                    -> DevicePtr a
                                    -> DeviceM d Status

foreign import ccall safe "cudnnBatchNormalizationBackward"
  batchNormalizationBackward :: Handle d
                             -> BatchNormMode
                             -> Ptr a
                             -> Ptr a
                             -> Ptr a
                             -> Ptr a
                             -> TensorDescriptor
                             -> DevicePtr a
                             -> TensorDescriptor
                             -> DevicePtr a
                             -> TensorDescriptor
                             -> DevicePtr a
                             -> TensorDescriptor
                             -> DevicePtr a
                             -> DevicePtr a
                             -> DevicePtr a
                             -> Double
                             -> DevicePtr a
                             -> DevicePtr a
                             -> DeviceM d Status
