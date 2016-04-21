{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
module DeepBanana.Layer.CUDA.CuDNN (
    convolution2d
  , CuDNN.ConvolutionFwdAlgo
  , CuDNN.convolution_fwd_algo_implicit_gemm
  , CuDNN.convolution_fwd_algo_implicit_precomp_gemm
  , CuDNN.convolution_fwd_algo_gemm
  , CuDNN.convolution_fwd_algo_direct
  , CuDNN.convolution_fwd_algo_fft
  , CuDNN.convolution_fwd_algo_fft_tiling
  , CuDNN.ConvolutionBwdDataAlgo
  , CuDNN.convolution_bwd_data_algo_0
  , CuDNN.convolution_bwd_data_algo_1
  , CuDNN.convolution_bwd_data_algo_fft
  , CuDNN.convolution_bwd_data_algo_fft_tiling
  , CuDNN.ConvolutionBwdFilterAlgo
  , CuDNN.convolution_bwd_filter_algo_0
  , CuDNN.convolution_bwd_filter_algo_1
  , CuDNN.convolution_bwd_filter_algo_fft
  , CuDNN.convolution_bwd_filter_algo_3
  , bias
  , activation
  , CuDNN.activation_sigmoid
  , CuDNN.activation_relu
  , CuDNN.activation_tanh
  , pooling2d
  , CuDNN.pooling_max
  , CuDNN.pooling_average_count_include_padding
  , CuDNN.pooling_average_count_exclude_padding
  , softmax
  , CuDNN.softmax_fast
  , CuDNN.softmax_accurate
  , CuDNN.softmax_mode_instance
  , CuDNN.softmax_mode_channel
  , nhwc_to_nchw
  , nchw_to_nhwc
  ) where

import Control.Monad.Primitive (unsafePrimToPrim)
import Foreign.Marshal
import Foreign.Marshal.Array
import Unsafe.Coerce
import System.IO.Unsafe

import DeepBanana.Device
import qualified DeepBanana.Device.CUDA as CUDA
import qualified DeepBanana.Device.CuDNN as CuDNN
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Exception
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Layer.CUDA.CuDNN.Exception
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception
import qualified DeepBanana.Tensor.Mutable as MT
import DeepBanana.Tensor.Mutable (MTensor, IOTensor, withDevicePtr, emptyTensor)

convolution2d :: forall d m a . (Device d, MonadCuda m, TensorScalar a)
              => (Int,Int)
              -> (Int,Int)
              -> CuDNN.ConvolutionFwdAlgo
              -> CuDNN.ConvolutionBwdDataAlgo
              -> CuDNN.ConvolutionBwdFilterAlgo
              -> Layer m a '[Tensor d 4 a] (Tensor d 4 a) (Tensor d 4 a)
convolution2d padding stride fwdAlgo bwdDataAlgo bwdFilterAlgo =
  combinePasses convfwd convbwd
  where convfwd (W ((:.) filters Z)) fmaps = do
          embedCudaFromST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd CuDNN.handle padding stride fwdAlgo fmaps' filters'
            unsafeFreeze convres
        convbwd (W ((:.) filters Z)) fmaps out = do
          let bwdfilters upgrad =
                unsafeRunCudaError $ embedCudaErrorFromST $ do
                  filters' <- unsafeThaw filters
                  fmaps' <- unsafeThaw fmaps
                  upgrad' <- unsafeThaw upgrad
                  filtersgrad <- convolution2dBwdFilters CuDNN.handle bwdFilterAlgo padding stride fmaps' filters' upgrad'
                  unsafeFreeze filtersgrad
              bwdinputs upgrad =
                unsafeRunCudaError $ embedCudaErrorFromST $ do
                  filters' <- unsafeThaw filters
                  fmaps' <- unsafeThaw fmaps
                  upgrad' <- unsafeThaw upgrad
                  inputsgrad <- convolution2dBwdInputs CuDNN.handle bwdDataAlgo padding stride fmaps' filters' upgrad'
                  unsafeFreeze inputsgrad
          return $ broadcast' (shape out)
            >>> \upgrad -> (W $ bwdfilters upgrad :. Z, bwdinputs upgrad)

bias :: forall m d a . (Device d, MonadCuda m, TensorScalar a)
     => Layer m a '[Tensor d 1 a] (Tensor d 4 a) (Tensor d 4 a)
bias = combinePasses biasFwd biasBwd
  where biasFwd (W ((:.) bias_w Z)) batch = embedCudaFromST $ do
          _:.c:._ <- return $ shape batch
          bias_w' <- unsafeThaw bias_w >>= MT.reshape (1:.c:.1:.1:.Z)
          batch' <- unsafeThaw batch
          out <- biasForward CuDNN.handle bias_w' batch'
          unsafeFreeze out
        biasBwd _ _ out =
          case shape out of
           _:.c:._ -> return
                      $ broadcast' (shape out)
                      >>> \upgrad ->
                           let biasgrad = unsafeRunCudaError $ embedCudaErrorFromST $ do
                                 upgrad' <- unsafeThaw upgrad
                                 grad <- biasBackward CuDNN.handle upgrad' >>= MT.reshape (c:.Z)
                                 unsafeFreeze grad
                           in (W ((:.) biasgrad Z), upgrad)

activation :: forall m d a n . (Device d, MonadCuda m, TensorScalar a, Shape (Dim n))
           => CuDNN.ActivationMode
           -> Layer m a '[] (Tensor d n a) (Tensor d n a)
activation mode =
  combinePasses' actfwd actbwd
  where to4 t = reshape' (size (shape t):.1:.1:.1:.Z) t
        actfwd fmaps = do
          embedCudaFromST $ do
            fmaps' <- unsafeThaw $ to4 fmaps
            activations <- activationFwd CuDNN.handle mode fmaps'
            fmap (reshape' (shape fmaps)) $ unsafeFreeze activations
        actbwd inp out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              inp' <- unsafeThaw $ to4 inp
              out' <- unsafeThaw $ to4 out
              upgrad' <- unsafeThaw $ to4 upgrad
              grad <- activationBwd CuDNN.handle mode inp' out' upgrad'
              fmap (reshape' (shape inp)) $ unsafeFreeze grad

-- pooling
pooling2d :: forall m d a . (Device d, MonadCuda m, TensorScalar a)
          => (Int,Int)
          -> (Int,Int)
          -> (Int,Int)
          -> CuDNN.PoolingMode
          -> Layer m a '[] (Tensor d 4 a) (Tensor d 4 a)
pooling2d psize padding stride mode =
  combinePasses' poolfwd poolbwd
  where poolfwd fmaps = do
          embedCudaFromST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd CuDNN.handle psize padding stride mode fmaps'
            unsafeFreeze poolres
        poolbwd inp out = do
          return $ broadcast' (shape out) >>> \
            upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              inp' <- unsafeThaw inp
              out' <- unsafeThaw out
              upgrad' <- unsafeThaw upgrad
              grad <- pooling2dBwd CuDNN.handle psize padding stride mode inp' out' upgrad'
              unsafeFreeze grad

softmax :: forall m d a . (Device d, MonadCuda m, TensorScalar a)
        => CuDNN.SoftmaxAlgorithm
        -> CuDNN.SoftmaxMode
        -> Layer m a '[] (Tensor d 2 a) (Tensor d 2 a)
softmax algorithm mode = combinePasses' softmaxFwd softmaxBwd
  where softmaxFwd input = embedCudaFromST $ do
          let rows:.cols:.Z = shape input
          input' <- unsafeThaw input >>= MT.reshape (rows:.cols:.1:.1:.Z)
          out <- softmaxForward CuDNN.handle algorithm mode input'
          unsafeFreeze out >>= reshape (rows:.cols:.Z)
        softmaxBwd inp out =
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              let rows:.cols:.Z = shape inp
              mu <- unsafeThaw upgrad >>= MT.reshape (rows:.cols:.1:.1:.Z)
              mout <- unsafeThaw out >>= MT.reshape (rows:.cols:.1:.1:.Z)
              mgrad <- softmaxBackward CuDNN.handle algorithm mode mout mu
              unsafeFreeze mgrad >>= reshape (rows:.cols:.Z)

nchw_to_nhwc :: forall d m a . (Device d, MonadCuda m, TensorScalar a)
             => Layer m a '[] (Tensor d 4 a) (Tensor d 4 a)
nchw_to_nhwc = combinePasses' fwdTrans bwdTrans
  where fwdTrans t = do
          embedCudaFromST $ do
            mt <- unsafeThaw t
            mt' <- nchw_to_nhwc' CuDNN.handle mt
            unsafeFreeze mt'
        bwdTrans _ out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              mu <- unsafeThaw upgrad
              mu' <- nhwc_to_nchw' CuDNN.handle mu
              unsafeFreeze mu'

nhwc_to_nchw :: forall d m a . (Device d, MonadCuda m, TensorScalar a)
             => Layer m a '[] (Tensor d 4 a) (Tensor d 4 a)
nhwc_to_nchw = combinePasses' fwdTrans bwdTrans
  where fwdTrans t = do
          embedCudaFromST $ do
            mt <- unsafeThaw t
            mt' <- nhwc_to_nchw' CuDNN.handle mt
            unsafeFreeze mt'
        bwdTrans _ out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              mu <- unsafeThaw upgrad
              mu' <- nchw_to_nhwc' CuDNN.handle mu
              unsafeFreeze mu'

-- Helper functions to deal with low-level boilerplate of CuDNN.
withDescriptor :: (MonadIO m, MonadError t m, Variant t AllocFailed, Storable desc)
               => (Ptr desc -> m CuDNN.Status) -- creation fct
               -> (desc -> m CuDNN.Status) -- set fct
               -> (desc -> m CuDNN.Status) -- destroy fct
               -> (desc -> m a) -- action to perform
               -> m a
withDescriptor create set destroy action = do
  descptr <- liftIO $ malloc
  attemptGCThenRetryOn (Proxy :: Proxy AllocFailed) $ create descptr
  desc <- liftIO $ peek descptr
  liftIO $ free descptr
  set desc
  x <- action desc
  destroy desc
  return x

withTensor4d :: forall m t d a b
             . (MonadIO m, PrimMonad m, PrimState m ~ RealWorld, MonadError t m,
                Variant t AllocFailed, Variant t BadParam,
                Variant t NotSupported, TensorScalar a, Device d)
             => IOTensor d 4 a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> ExceptT t IO b)
             -> m b
withTensor4d tensor action = do
  datatype <- MT.dtype tensor
  withTensorDesc (Proxy :: Proxy d) CuDNN.nchw datatype (MT.shape tensor) $
    \tensordesc -> do
      eres <- liftIO $ withDevicePtr tensor
              $ \dvcptr -> runExceptT $ action tensordesc dvcptr
      embedExcept eres

withTensorDesc :: (Device d, MonadIO m, MonadError t m, Variant t AllocFailed,
                   Variant t BadParam, Variant t NotSupported)
               => Proxy d
               -> CuDNN.TensorFormat
               -> CuDNN.DataType
               -> Dim 4
               -> (CuDNN.TensorDescriptor -> ExceptT t IO a)
               -> m a
withTensorDesc p format dtype (n:.c:.h:.w:.Z) action = do
  let [n',c',h',w'] = fmap fromIntegral [n,c,h,w]
      create4d descptr = handleStatus (Proxy :: Proxy AllocFailed)
                         $ liftIO $ runDeviceM p
                         $ CuDNN.createTensorDescriptor descptr
      set4d d = handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy NotSupported)
                $ liftIO $ runDeviceM p
                $ CuDNN.setTensor4dDescriptor d CuDNN.nchw dtype n' c' h' w'
      destroy4d desc = liftIO $ runDeviceM p $ CuDNN.destroyTensorDescriptor desc
  withDescriptor create4d set4d destroy4d
    $ \descptr -> do
      eres <- liftIO $ runExceptT $ action descptr
      embedExcept eres

withFilter4d :: forall d m t a b
             . (Device d, MonadIO m, PrimMonad m, PrimState m ~ RealWorld,
                MonadError t m, Variant t AllocFailed,
                Variant t BadParam, TensorScalar a)
             => IOTensor d 4 a
             -> (CuDNN.FilterDescriptor -> CUDA.DevicePtr a -> ExceptT t IO b)
             -> m b
withFilter4d tensor action = do
  datatype <- MT.dtype tensor
  let p = Proxy :: Proxy d
      [n,c,h,w] = fmap fromIntegral $ dimensions $ MT.shape tensor
      createFilter = handleStatus (Proxy :: Proxy AllocFailed)
                     . liftIO . runDeviceM p . CuDNN.createFilterDescriptor
      setFilter d = handleStatus (Proxy :: Proxy BadParam)
                    $ liftIO $ runDeviceM p
                    $ CuDNN.setFilter4dDescriptor d datatype n c h w
      destroyFilter = liftIO . runDeviceM p . CuDNN.destroyFilterDescriptor
  withDescriptor createFilter setFilter destroyFilter
    $ \filtdesc -> do
      eres <- liftIO $ withDevicePtr tensor
              $ \dvcptr -> runExceptT $ action filtdesc dvcptr
      embedExcept eres

withConvDesc :: (Device d, MonadIO m, MonadError t m, Variant t AllocFailed,
                 Variant t BadParam, Variant t NotSupported)
             => Proxy d -> (Int,Int) -> (Int,Int) -> (Int,Int)
             -> (CuDNN.ConvolutionDescriptor -> m a) -> m a
withConvDesc p (padh,padw) (strh,strw) (uph,upw) action = do
  let [cpadh,cpadw,cstrh,cstrw,cupw,cuph] =
        map fromIntegral [padh,padw,strh,strw,uph,upw]
      createConv = handleStatus (Proxy :: Proxy AllocFailed)
                   . liftIO . runDeviceM p . CuDNN.createConvolutionDescriptor
      setConv d = handleStatus (Proxy :: Proxy BadParam)
                  $ handleStatus (Proxy :: Proxy NotSupported)
                  $ liftIO $ runDeviceM p
                  $ CuDNN.setConvolution2dDescriptor d cpadh cpadw cstrh cstrw cupw cuph
                  CuDNN.convolution
      destroyConv = liftIO . runDeviceM p . CuDNN.destroyConvolutionDescriptor
  withDescriptor createConv setConv destroyConv action

withPoolDesc :: (Device d, MonadIO m, MonadError t m, Variant t AllocFailed,
                 Variant t BadParam)
             => Proxy d
             -> (Int, Int)
             -> (Int, Int)
             -> (Int, Int)
             -> CuDNN.PoolingMode
             -> (CuDNN.PoolingDescriptor -> m a)
             -> m a
withPoolDesc p (wh,ww) (padh,padw) (strh,strw) mode action = do
  let [cwh,cww,cpadh,cpadw,cstrh,cstrw] =
        fmap fromIntegral [wh,ww,padh,padw,strh,strw]
      createPool = handleStatus (Proxy :: Proxy AllocFailed)
                   . liftIO . runDeviceM p . CuDNN.createPoolingDescriptor
      setPool d = handleStatus (Proxy :: Proxy BadParam)
                  $ liftIO $ runDeviceM p
                  $ CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw
                  cstrh cstrw
      destroyPool = liftIO . runDeviceM p . CuDNN.destroyPoolingDescriptor
  withDescriptor createPool setPool destroyPool action

convOutShape :: Dim 4 -> Dim 4 -> (Int,Int) -> (Int,Int) -> Dim 4
convOutShape (n1:.c1:.h1:.w1:.Z) (n2:.c2:.h2:.w2:.Z) (padh,padw) (strh,strw) =
  n1:.n2:.convOutDim h1 h2 padh strh:.convOutDim w1 w2 padw strw:.Z
  where convOutDim input_dim filter_dim padding stride =
          1 + (input_dim + (2 * padding) - filter_dim) `div` stride

-- convolution
convolution2dFwd :: forall m d a
                 . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
                 => CuDNN.Handle d
                 -> (Int,Int)
                 -> (Int,Int)
                 -> CuDNN.ConvolutionFwdAlgo
                 -> MTensor (PrimState m) d 4 a
                 -> MTensor (PrimState m) d 4 a
                 -> m (MTensor (PrimState m) d 4 a)
convolution2dFwd handle (padh,padw) (strh,strw) algo fmaps filters = do
  -- make the descriptors
  let outshp = convOutShape (MT.shape fmaps) (MT.shape filters) (padh,padw) (strh,strw)
      p = Proxy :: Proxy d
  embedCudaError unsafeIOToPrim $ do
    withConvDesc p (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
      withTensor4d (unsafeCoerce fmaps :: IOTensor d 4 a)
        $ \inputdesc inputptr -> do
        withFilter4d (unsafeCoerce filters :: IOTensor d 4 a)
          $ \filtersdesc filtersptr -> do
          output <- emptyTensor outshp :: CudaErrorT IO (IOTensor d 4 a)
          withTensor4d output $ \outputdesc outputptr -> do
            -- allocate workspace
            wkspcsizeptr <- liftIO $ malloc
            workspacesize <- do
              handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy NotSupported)
                $ liftIO $ runDeviceM p $ CuDNN.getConvolutionForwardWorkspaceSize
                handle inputdesc filtersdesc convdesc outputdesc algo
                wkspcsizeptr
              liftIO $ peek wkspcsizeptr
            liftIO $ free wkspcsizeptr
            workspace <- attemptGCThenRetryOn (Proxy :: Proxy MemoryAllocation)
                         $ handleCUDAException (Proxy :: Proxy MemoryAllocation)
                         $ runDeviceM p
                         $ CUDA.mallocArray (fromIntegral workspacesize)
            -- allocate alpha and beta
            alpha <- liftIO $ newArray [1]
            beta <- liftIO $ newArray [0]
            -- finally run the damn thing
            handleStatus (Proxy :: Proxy BadParam)
              $ handleStatus (Proxy :: Proxy NotSupported)
              $ handleStatus (Proxy :: Proxy MappingError)
              $ handleStatus (Proxy :: Proxy ExecutionFailed)
              $ liftIO $ runDeviceM p
              $ CuDNN.convolutionForward
              handle alpha inputdesc inputptr filtersdesc filtersptr
              convdesc algo workspace workspacesize beta outputdesc
              outputptr
            liftIO $ free alpha
            liftIO $ free beta
            liftIO $ runDeviceM p $ CUDA.free workspace
          return $ unsafeCoerce output

convolution2dBwdFilters :: forall m d a
                        . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
                        => CuDNN.Handle d
                        -> CuDNN.ConvolutionBwdFilterAlgo
                        -> (Int,Int)
                        -> (Int,Int)
                        -> MTensor (PrimState m) d 4 a
                        -> MTensor (PrimState m) d 4 a
                        -> MTensor (PrimState m) d 4 a
                        -> m (MTensor (PrimState m) d 4 a)
convolution2dBwdFilters handle algo (padh,padw) (strh,strw) fmaps filters upgrad = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withConvDesc p (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
    withTensor4d (unsafeCoerce fmaps :: IOTensor d 4 a)
      $ \inputdesc inputptr -> do
      withTensor4d (unsafeCoerce upgrad :: IOTensor d 4 a)
        $ \upgraddesc upgradptr -> do
        alpha <- liftIO $ newArray [1]
        beta <- liftIO $ newArray [0]
        -- compute gradient with regards to the filters
        filtersgrad <- emptyTensor $ MT.shape filters :: CudaErrorT IO (IOTensor d 4 a)
        withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr -> do
          workspacesizeptr <- liftIO $ malloc
          handleStatus (Proxy :: Proxy BadParam)
            $ handleStatus (Proxy :: Proxy NotSupported)
            $ liftIO $ runDeviceM p
            $ CuDNN.getConvolutionBackwardFilterWorkspaceSize handle inputdesc upgraddesc
            convdesc filtersgraddesc algo workspacesizeptr
          workspacesize <- liftIO $ peek workspacesizeptr
          liftIO $ free workspacesizeptr
          workspace <- attemptGCThenRetryOn (Proxy :: Proxy MemoryAllocation)
                       $ handleCUDAException (Proxy :: Proxy MemoryAllocation)
                       $ runDeviceM p
                       $ CUDA.mallocArray (fromIntegral workspacesize)
          handleStatus (Proxy :: Proxy BadParam)
            $ handleStatus (Proxy :: Proxy NotSupported)
            $ handleStatus (Proxy :: Proxy MappingError)
            $ handleStatus (Proxy :: Proxy ExecutionFailed)
            $ liftIO $ runDeviceM p
            $ CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
            upgraddesc upgradptr convdesc algo workspace workspacesize beta filtersgraddesc filtersgradptr
          liftIO $ runDeviceM p $ CUDA.free workspace
        liftIO $ free alpha
        liftIO $ free beta
        return $ unsafeCoerce filtersgrad

convolution2dBwdInputs :: forall m d a
                       . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
                       => CuDNN.Handle d
                       -> CuDNN.ConvolutionBwdDataAlgo
                       -> (Int,Int)
                       -> (Int,Int)
                       -> MTensor (PrimState m) d 4 a
                       -> MTensor (PrimState m) d 4 a
                       -> MTensor (PrimState m) d 4 a
                       -> m (MTensor (PrimState m) d 4 a)
convolution2dBwdInputs handle algo (padh,padw) (strh,strw) fmaps filters upgrad = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withConvDesc p (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
    withFilter4d (unsafeCoerce filters :: IOTensor d 4 a)
      $ \filtersdesc filtersptr -> do
      withTensor4d (unsafeCoerce upgrad :: IOTensor d 4 a)
        $ \upgraddesc upgradptr -> do
        alpha <- liftIO $ newArray [1]
        beta <- liftIO $ newArray [0]
        -- compute gradient with regards to the input feature maps
        inputsgrad <- emptyTensor $ MT.shape fmaps :: CudaErrorT IO (IOTensor d 4 a)
        withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr -> do
          workspacesizeptr <- liftIO $ malloc
          handleStatus (Proxy :: Proxy BadParam)
            $ handleStatus (Proxy :: Proxy NotSupported)
            $ liftIO $ runDeviceM p
            $ CuDNN.getConvolutionBackwardDataWorkspaceSize handle filtersdesc
            upgraddesc convdesc inputsgraddesc algo workspacesizeptr
          workspacesize <- liftIO $ peek workspacesizeptr
          liftIO $ free workspacesizeptr
          workspace <- attemptGCThenRetryOn (Proxy :: Proxy MemoryAllocation)
                       $ handleCUDAException (Proxy :: Proxy MemoryAllocation)
                       $ runDeviceM p
                       $ CUDA.mallocArray (fromIntegral workspacesize)
          handleStatus (Proxy :: Proxy BadParam)
            $ handleStatus (Proxy :: Proxy NotSupported)
            $ handleStatus (Proxy :: Proxy MappingError)
            $ handleStatus (Proxy :: Proxy ExecutionFailed)
            $ liftIO $ runDeviceM p
            $ CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
            upgraddesc upgradptr convdesc algo workspace workspacesize beta inputsgraddesc inputsgradptr
          liftIO $ runDeviceM p $ CUDA.free workspace
        liftIO $ free alpha
        liftIO $ free beta
        return $ unsafeCoerce inputsgrad

-- bias
biasForward :: forall m d a
            . (MonadCudaError m, PrimMonad m, Device d, TensorScalar a)
            => CuDNN.Handle d
            -> MTensor (PrimState m) d 4 a
            -> MTensor (PrimState m) d 4 a
            -> m (MTensor (PrimState m) d 4 a)
biasForward handle bias batch = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withTensor4d (unsafeCoerce bias :: IOTensor d 4 a) $ \biasdesc biasptr -> do
    out <- MT.copy (unsafeCoerce batch :: IOTensor d 4 a) :: CudaErrorT IO (IOTensor d 4 a)
    withTensor4d out $ \outdesc outptr -> do
      (alpha, beta) <- liftIO $ pure (,) <*> newArray [1] <*> newArray [1]
      handleStatus (Proxy :: Proxy NotSupported)
        $ handleStatus (Proxy :: Proxy BadParam)
        $ handleStatus (Proxy :: Proxy ExecutionFailed)
        $ liftIO $ runDeviceM p
        $ CuDNN.addTensor handle alpha biasdesc biasptr beta outdesc outptr
      liftIO $ free alpha >> free beta
    return $ unsafeCoerce out

biasBackward :: forall m d a
             . (MonadCudaError m, PrimMonad m, Device d, TensorScalar a)
             => CuDNN.Handle d
             -> MTensor (PrimState m) d 4 a
             -> m (MTensor (PrimState m) d 4 a)
biasBackward handle upgrad = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withTensor4d (unsafeCoerce upgrad :: IOTensor d 4 a)$ \upgraddesc upgradptr -> do
    let _:.c:._ = MT.shape upgrad
    biasgrad <- emptyTensor $ 1:.c:.1:.1:.Z :: CudaErrorT IO (IOTensor d 4 a)
    withTensor4d biasgrad $ \biasgraddesc biasgradptr -> do
      (alpha, beta) <- liftIO $ pure (,) <*> newArray [1] <*> newArray [0]
      handleStatus (Proxy :: Proxy BadParam)
        $ liftIO $ runDeviceM p
        $ CuDNN.convolutionBackwardBias handle alpha upgraddesc upgradptr
        beta biasgraddesc biasgradptr
      liftIO $ free alpha >> free beta
      return $ unsafeCoerce biasgrad

-- activations
activationFwd :: forall m d a
              . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
              => CuDNN.Handle d
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) d 4 a
              -> m (MTensor (PrimState m) d 4 a)
activationFwd handle mode input = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withTensor4d (unsafeCoerce input :: IOTensor d 4 a) $ \inputdesc inputptr -> do
    output <- emptyTensor $ MT.shape input :: CudaErrorT IO (IOTensor d 4 a)
    withTensor4d output $ \outputdesc outputptr -> do
      alpha <- liftIO $ newArray [1]
      beta <- liftIO $ newArray [0]
      handleStatus (Proxy :: Proxy BadParam)
        $ handleStatus (Proxy :: Proxy ExecutionFailed)
        $ liftIO $ runDeviceM p
        $ CuDNN.activationForward handle mode alpha inputdesc
        inputptr beta outputdesc outputptr
      liftIO $ free alpha
      liftIO $ free beta
    return $ unsafeCoerce output

activationBwd :: forall m d a
              . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
              => CuDNN.Handle d
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) d 4 a
              -> MTensor (PrimState m) d 4 a
              -> MTensor (PrimState m) d 4 a
              -> m (MTensor (PrimState m) d 4 a)
activationBwd handle mode input output upgrad = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withTensor4d (unsafeCoerce input :: IOTensor d 4 a) $ \inputdesc inputptr -> do
    withTensor4d (unsafeCoerce upgrad :: IOTensor d 4 a) $ \upgraddesc upgradptr -> do
      grad <- emptyTensor $ MT.shape input :: CudaErrorT IO (IOTensor d 4 a)
      withTensor4d (unsafeCoerce output :: IOTensor d 4 a) $ \outputdesc outputptr -> do
        withTensor4d grad $ \graddesc gradptr -> do
          alpha <- liftIO $ newArray [1]
          beta <- liftIO $ newArray [0]
          handleStatus (Proxy :: Proxy BadParam)
            $ handleStatus (Proxy :: Proxy NotSupported)
            $ handleStatus (Proxy :: Proxy ExecutionFailed)
            $ liftIO $ runDeviceM p
            $ CuDNN.activationBackward handle mode alpha inputdesc inputptr
            upgraddesc upgradptr outputdesc outputptr beta graddesc
            gradptr
          liftIO $ free alpha
          liftIO $ free beta
      return $ unsafeCoerce grad

-- 2d pooling
-- Helper to compute output MT.shape.
pooling2dOutputShape :: (Int,Int) -> (Int,Int) -> (Int,Int) -> Dim 4 -> Dim 4
pooling2dOutputShape (szr,szc) (padr,padc) (strr, strc) (n:.ch:.r:.c:.Z) =
  (n:.ch:.outr:.outc:.Z)
  where
    inr = r + padr
    inc = c + padc
    outr = (inr - overlap_r) `div` strr
    outc = (inc - overlap_c) `div` strc
    overlap_r = max 0 (szr - strr)
    overlap_c = max 0 (szc - strc)

pooling2dFwd :: forall m d a
             . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
             => CuDNN.Handle d
             -> (Int,Int)
             -> (Int,Int)
             -> (Int,Int)
             -> CuDNN.PoolingMode
             -> MTensor (PrimState m) d 4 a
             -> m (MTensor (PrimState m) d 4 a)
pooling2dFwd handle size padding stride mode input = do
  let outshp = pooling2dOutputShape size padding stride $ MT.shape input
      p = Proxy :: Proxy d
  embedCudaError unsafeIOToPrim $ do
    withPoolDesc p size padding stride mode $ \pooldesc -> do
      withTensor4d (unsafeCoerce input :: IOTensor d 4 a)
        $ \inputdesc inputptr -> do
        output <- emptyTensor outshp :: CudaErrorT IO (IOTensor d 4 a)
        -- actual pooling
        withTensor4d output $ \outputdesc outputptr -> do
          alpha <- liftIO $ newArray [1]
          beta <- liftIO $ newArray [0]
          handleStatus (Proxy :: Proxy BadParam)
            $ handleStatus (Proxy :: Proxy NotSupported)
            $ handleStatus (Proxy :: Proxy ExecutionFailed)
            $ liftIO $ runDeviceM p
            $ CuDNN.poolingForward handle pooldesc alpha inputdesc
            inputptr beta outputdesc outputptr
          liftIO $ free alpha
          liftIO $ free beta
        return $ unsafeCoerce output

pooling2dBwd :: forall m d a
             . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
             => CuDNN.Handle d
             -> (Int,Int)
             -> (Int,Int)
             -> (Int,Int)
             -> CuDNN.PoolingMode
             -> MTensor (PrimState m) d 4 a
             -> MTensor (PrimState m) d 4 a
             -> MTensor (PrimState m) d 4 a
             -> m (MTensor (PrimState m) d 4 a)
pooling2dBwd handle size padding stride mode inp out upgrad = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withPoolDesc p size padding stride mode $ \pooldesc -> do
    withTensor4d (unsafeCoerce inp :: IOTensor d 4 a)
      $ \inpdesc inpptr -> do
      withTensor4d (unsafeCoerce out :: IOTensor d 4 a)
        $ \outdesc outptr -> do
        withTensor4d (unsafeCoerce upgrad :: IOTensor d 4 a)
          $ \upgraddesc upgradptr -> do
          grad <- emptyTensor $ MT.shape inp :: CudaErrorT IO (IOTensor d 4 a)
          withTensor4d grad $ \graddesc gradptr -> do
            alpha <- liftIO $ newArray [1]
            beta <- liftIO $ newArray [0]
            handleStatus (Proxy :: Proxy BadParam)
              $ handleStatus (Proxy :: Proxy NotSupported)
              $ handleStatus (Proxy :: Proxy ExecutionFailed)
              $ liftIO $ runDeviceM p
              $ CuDNN.poolingBackward handle pooldesc alpha outdesc outptr
              upgraddesc upgradptr inpdesc inpptr beta graddesc gradptr
          return $ unsafeCoerce grad

-- softmax
softmaxForward :: forall m d a
               . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
               => CuDNN.Handle d
               -> CuDNN.SoftmaxAlgorithm
               -> CuDNN.SoftmaxMode
               -> MTensor (PrimState m) d 4 a
               -> m (MTensor (PrimState m) d 4 a)
softmaxForward handle algorithm mode input = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withTensor4d (unsafeCoerce input :: IOTensor d 4 a) $ \inpdesc inpptr -> do
    out <- MT.emptyTensor $ MT.shape input :: CudaErrorT IO (IOTensor d 4 a)
    withTensor4d out $ \outdesc outptr -> do
      (alpha, beta) <- liftIO $ pure (,) <*> newArray [1] <*> newArray [0]
      handleStatus (Proxy :: Proxy BadParam)
        $ handleStatus (Proxy :: Proxy ExecutionFailed)
        $ liftIO $ runDeviceM p
        $ CuDNN.softmaxForward handle algorithm mode alpha inpdesc inpptr
        beta outdesc outptr
      liftIO $ free alpha >> free beta
    return $ unsafeCoerce out

softmaxBackward :: forall m d a
                . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
                => CuDNN.Handle d
                -> CuDNN.SoftmaxAlgorithm
                -> CuDNN.SoftmaxMode
                -> MTensor (PrimState m) d 4 a
                -> MTensor (PrimState m) d 4 a
                -> m (MTensor (PrimState m) d 4 a)
softmaxBackward handle algorithm mode src upgrad = embedCudaError unsafeIOToPrim $ do
  let p = Proxy :: Proxy d
  withTensor4d (unsafeCoerce src :: IOTensor d 4 a) $ \srcdesc srcptr -> do
    withTensor4d (unsafeCoerce upgrad :: IOTensor d 4 a) $ \upgraddesc upgradptr -> do
      out <- MT.emptyTensor $ MT.shape upgrad :: CudaErrorT IO (IOTensor d 4 a)
      withTensor4d out $ \outdesc outptr -> do
        (alpha, beta) <- liftIO $ pure (,) <*> newArray [1] <*> newArray [0]
        handleStatus (Proxy :: Proxy BadParam)
          $ handleStatus (Proxy :: Proxy ExecutionFailed)
          $ liftIO $ runDeviceM p
          $ CuDNN.softmaxBackward handle algorithm mode alpha srcdesc srcptr
          upgraddesc upgradptr beta outdesc outptr
        liftIO $ free alpha >> free beta
      return $ unsafeCoerce out

nchw_to_nhwc' :: forall m d a
              . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
              => CuDNN.Handle d
              -> MTensor (PrimState m) d 4 a
              -> m (MTensor (PrimState m) d 4 a)
nchw_to_nhwc' handle src = embedCudaError unsafeIOToPrim $ do
  res <- transformTensorIO handle CuDNN.nchw CuDNN.nhwc
         (unsafeCoerce src :: IOTensor d 4 a)
  return $ unsafeCoerce res

nhwc_to_nchw' :: forall m d a
              . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
              => CuDNN.Handle d
              -> MTensor (PrimState m) d 4 a
              -> m (MTensor (PrimState m) d 4 a)
nhwc_to_nchw' handle src = embedCudaError unsafeIOToPrim $ do
  res <- transformTensorIO handle CuDNN.nhwc CuDNN.nchw
         (unsafeCoerce src :: IOTensor d 4 a)
  return $ unsafeCoerce res

-- Unsafe but useful.
transformTensorIO :: forall m d a
                  . (TensorScalar a, MonadIO m, PrimMonad m, PrimState m ~ RealWorld,
                     MonadCudaError m, Device d)
                  => CuDNN.Handle d
                  -> CuDNN.TensorFormat
                  -> CuDNN.TensorFormat
                  -> IOTensor d 4 a
                  -> m (IOTensor d 4 a)
transformTensorIO handle srcf dstf src = do
  datatype <- MT.dtype src
  let [a',b',c',d'] = dimensions $ MT.shape src
      [n,c,h,w] = if srcf == CuDNN.nchw then [a',b',c',d']
                  else if srcf == CuDNN.nhwc then [a',d',b',c']
                       else error $ "Unsupported input format: " ++ show srcf
      shp = (n:.c:.h:.w:.Z)
      p = Proxy :: Proxy d
  withTensorDesc p srcf datatype shp $ \srcdesc -> do
      withTensorDesc p dstf datatype shp $ \dstdesc -> do
          dst <- (emptyTensor $ if dstf == CuDNN.nchw
                                then n:.c:.h:.w:.Z
                                else n:.h:.w:.c:.Z) :: CudaErrorT IO (IOTensor d 4 a)
          eres1 <- liftIO $ withDevicePtr src $ \srcptr -> runExceptT $ do
            eres2 <- liftIO $ withDevicePtr dst $ \dstptr -> runExceptT $ do
              alpha <- liftIO $ newArray [1]
              beta <- liftIO $ newArray [0]
              handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy ExecutionFailed)
                $ liftIO $ runDeviceM p
                $ CuDNN.transformTensor handle alpha srcdesc srcptr beta
                dstdesc dstptr
            embedExcept eres2
          embedExcept eres1
          return dst
