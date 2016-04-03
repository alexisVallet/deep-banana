{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
module DeepBanana.Layer.CUDA.CuDNN (
    convolution2d
  , CuDNN.convolution_fwd_algo_implicit_gemm
  , CuDNN.convolution_fwd_algo_implicit_precomp_gemm
  , CuDNN.convolution_fwd_algo_gemm
  , CuDNN.convolution_fwd_algo_direct
  , bias
  , activation
  , CuDNN.activation_sigmoid
  , CuDNN.activation_relu
  , CuDNN.activation_tanh
  , pooling2d
  , CuDNN.pooling_max
  , CuDNN.pooling_average_count_include_padding
  , CuDNN.pooling_average_count_exclude_padding
  , nhwc_to_nchw
  , nchw_to_nhwc
  , cudnnHandle
  ) where

import Control.Monad.Primitive (unsafePrimToPrim)
import Foreign.Marshal
import Foreign.Marshal.Array
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import Unsafe.Coerce
import System.IO.Unsafe

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

convolution2d :: forall m a . (MonadCuda m, TensorScalar a)
              => (Int,Int)
              -> (Int,Int)
              -> CuDNN.ConvolutionFwdAlgo
              -> Layer m a '[Tensor 4 a] (Tensor 4 a) (Tensor 4 a)
convolution2d padding stride algo =
  combinePasses convfwd convbwd
  where convfwd (HLS (HCons filters HNil)) fmaps = do
          embedCudaFromST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd cudnnHandle padding stride algo fmaps' filters'
            unsafeFreeze convres
        convbwd (HLS (HCons filters HNil)) fmaps out = do
          let bwdfilters upgrad =
                unsafeRunCudaError $ embedCudaErrorFromST $ do
                  filters' <- unsafeThaw filters
                  fmaps' <- unsafeThaw fmaps
                  upgrad' <- unsafeThaw upgrad
                  filtersgrad <- convolution2dBwdFilters cudnnHandle padding stride fmaps' filters' upgrad'
                  unsafeFreeze filtersgrad
              bwdinputs upgrad =
                unsafeRunCudaError $ embedCudaErrorFromST $ do
                  filters' <- unsafeThaw filters
                  fmaps' <- unsafeThaw fmaps
                  upgrad' <- unsafeThaw upgrad
                  inputsgrad <- convolution2dBwdInputs cudnnHandle padding stride fmaps' filters' upgrad'
                  unsafeFreeze inputsgrad
          return $ broadcast' (shape out)
            >>> \upgrad -> (HLS $ bwdfilters upgrad `HCons` HNil, bwdinputs upgrad)

bias :: (MonadCuda m, TensorScalar a)
     => Layer m a '[Tensor 1 a] (Tensor 4 a) (Tensor 4 a)
bias = combinePasses biasFwd biasBwd
  where biasFwd (HLS (HCons bias_w HNil)) batch = embedCudaFromST $ do
          let _:.c:._ = shape batch
          bias_w' <- unsafeThaw bias_w >>= MT.reshape (1:.c:.1:.1:.Z)
          batch' <- unsafeThaw batch
          out <- biasForward cudnnHandle bias_w' batch'
          unsafeFreeze out
        biasBwd _ _ out =
          return
          $ broadcast' (shape out)
          >>> \upgrad ->
               let biasgrad = unsafeRunCudaError $ embedCudaErrorFromST $ do
                     upgrad' <- unsafeThaw upgrad
                     grad <- biasBackward cudnnHandle upgrad' >>= MT.reshape (c:.Z)
                     unsafeFreeze grad
                   _:.c:._ = shape out
               in (HLS (HCons biasgrad HNil), upgrad)

activation :: (MonadCuda m, TensorScalar a, Shape (Dim n))
           => CuDNN.ActivationMode
           -> Layer m a '[] (Tensor n a) (Tensor n a)
activation mode =
  combinePasses' actfwd actbwd
  where to4 t = reshape' (size (shape t):.1:.1:.1:.Z) t
        actfwd fmaps = do
          embedCudaFromST $ do
            fmaps' <- unsafeThaw $ to4 fmaps
            activations <- activationFwd cudnnHandle mode fmaps'
            fmap (reshape' (shape fmaps)) $ unsafeFreeze activations
        actbwd inp out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              inp' <- unsafeThaw $ to4 inp
              out' <- unsafeThaw $ to4 out
              upgrad' <- unsafeThaw $ to4 upgrad
              grad <- activationBwd cudnnHandle mode inp' out' upgrad'
              fmap (reshape' (shape inp)) $ unsafeFreeze grad

-- pooling
pooling2d :: (MonadCuda m, TensorScalar a)
          => (Int,Int)
          -> (Int,Int)
          -> (Int,Int)
          -> CuDNN.PoolingMode
          -> Layer m a '[] (Tensor 4 a) (Tensor 4 a)
pooling2d psize padding stride mode =
  combinePasses' poolfwd poolbwd
  where poolfwd fmaps = do
          embedCudaFromST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd cudnnHandle psize padding stride mode fmaps'
            unsafeFreeze poolres
        poolbwd inp out = do
          return $ broadcast' (shape out) >>> \
            upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              inp' <- unsafeThaw inp
              out' <- unsafeThaw out
              upgrad' <- unsafeThaw upgrad
              grad <- pooling2dBwd cudnnHandle psize padding stride mode inp' out' upgrad'
              unsafeFreeze grad

nchw_to_nhwc :: (MonadCuda m, TensorScalar a)
             => Layer m a '[] (Tensor 4 a) (Tensor 4 a)
nchw_to_nhwc = combinePasses' fwdTrans bwdTrans
  where fwdTrans t = do
          embedCudaFromST $ do
            mt <- unsafeThaw t
            mt' <- nchw_to_nhwc' cudnnHandle mt
            unsafeFreeze mt'
        bwdTrans _ out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              mu <- unsafeThaw upgrad
              mu' <- nhwc_to_nchw' cudnnHandle mu
              unsafeFreeze mu'

nhwc_to_nchw :: (MonadCuda m, TensorScalar a)
             => Layer m a '[] (Tensor 4 a) (Tensor 4 a)
nhwc_to_nchw = combinePasses' fwdTrans bwdTrans
  where fwdTrans t = do
          embedCudaFromST $ do
            mt <- unsafeThaw t
            mt' <- nhwc_to_nchw' cudnnHandle mt
            unsafeFreeze mt'
        bwdTrans _ out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              mu <- unsafeThaw upgrad
              mu' <- nchw_to_nhwc' cudnnHandle mu
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

withTensor4d :: (MonadIO m, PrimMonad m, PrimState m ~ RealWorld, MonadError t m,
                 Variant t AllocFailed, Variant t BadParam,
                 Variant t NotSupported, TensorScalar a)
             => IOTensor 4 a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> ExceptT t IO b)
             -> m b
withTensor4d tensor action = do
  datatype <- MT.dtype tensor
  withTensorDesc CuDNN.nchw datatype (MT.shape tensor) $
    \tensordesc -> do
      eres <- liftIO $ withDevicePtr tensor
              $ \dvcptr -> runExceptT $ action tensordesc dvcptr
      embedExcept eres

withTensorDesc :: (MonadIO m, MonadError t m, Variant t AllocFailed,
                   Variant t BadParam, Variant t NotSupported)
               => CuDNN.TensorFormat
               -> CuDNN.DataType
               -> Dim 4
               -> (CuDNN.TensorDescriptor -> ExceptT t IO a)
               -> m a
withTensorDesc format dtype (n:.c:.h:.w:.Z) action = do
  let [n',c',h',w'] = fmap fromIntegral [n,c,h,w]
      create4d descptr = handleStatus (Proxy :: Proxy AllocFailed)
                         $ liftIO $ CuDNN.createTensorDescriptor descptr
      set4d d = handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy NotSupported)
                $ liftIO
                $ CuDNN.setTensor4dDescriptor d CuDNN.nchw dtype n' c' h' w'
      destroy4d desc = liftIO $ CuDNN.destroyTensorDescriptor desc
  withDescriptor create4d set4d destroy4d
    $ \descptr -> do
      eres <- liftIO $ runExceptT $ action descptr
      embedExcept eres

withFilter4d :: (MonadIO m, PrimMonad m, PrimState m ~ RealWorld,
                 MonadError t m, Variant t AllocFailed,
                 Variant t BadParam, TensorScalar a)
             => IOTensor 4 a
             -> (CuDNN.FilterDescriptor -> CUDA.DevicePtr a -> ExceptT t IO b)
             -> m b
withFilter4d tensor action = do
  datatype <- MT.dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ dimensions $ MT.shape tensor
      createFilter = handleStatus (Proxy :: Proxy AllocFailed)
                     . liftIO . CuDNN.createFilterDescriptor
      setFilter d = handleStatus (Proxy :: Proxy BadParam)
                    $ liftIO $ CuDNN.setFilter4dDescriptor d datatype n c h w
      destroyFilter = liftIO . CuDNN.destroyFilterDescriptor
  withDescriptor createFilter setFilter destroyFilter
    $ \filtdesc -> do
      eres <- liftIO $ withDevicePtr tensor
              $ \dvcptr -> runExceptT $ action filtdesc dvcptr
      embedExcept eres

withConvDesc :: (MonadIO m, MonadError t m, Variant t AllocFailed, Variant t BadParam,
                 Variant t NotSupported)
             => (Int,Int) -> (Int,Int) -> (Int,Int)
             -> (CuDNN.ConvolutionDescriptor -> m a) -> m a
withConvDesc (padh,padw) (strh,strw) (uph,upw) action = do
  let [cpadh,cpadw,cstrh,cstrw,cupw,cuph] =
        map fromIntegral [padh,padw,strh,strw,uph,upw]
      createConv = handleStatus (Proxy :: Proxy AllocFailed)
                   . liftIO . CuDNN.createConvolutionDescriptor
      setConv d = handleStatus (Proxy :: Proxy BadParam)
                  $ handleStatus (Proxy :: Proxy NotSupported)
                  $ liftIO
                  $ CuDNN.setConvolution2dDescriptor d cpadh cpadw cstrh cstrw cupw cuph
                  CuDNN.convolution
      destroyConv = liftIO . CuDNN.destroyConvolutionDescriptor
  withDescriptor createConv setConv destroyConv action

withPoolDesc :: (MonadIO m, MonadError t m, Variant t AllocFailed,
                 Variant t BadParam)
             => (Int, Int)
             -> (Int, Int)
             -> (Int, Int)
             -> CuDNN.PoolingMode
             -> (CuDNN.PoolingDescriptor -> m a)
             -> m a
withPoolDesc (wh,ww) (padh,padw) (strh,strw) mode action = do
  let [cwh,cww,cpadh,cpadw,cstrh,cstrw] =
        fmap fromIntegral [wh,ww,padh,padw,strh,strw]
      createPool = handleStatus (Proxy :: Proxy AllocFailed)
                   . liftIO . CuDNN.createPoolingDescriptor
      setPool d = handleStatus (Proxy :: Proxy BadParam)
                  $ liftIO
                  $ CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw
                  cstrh cstrw
      destroyPool = liftIO . CuDNN.destroyPoolingDescriptor
  withDescriptor createPool setPool destroyPool action

convOutShape :: Dim 4 -> Dim 4 -> (Int,Int) -> (Int,Int) -> Dim 4
convOutShape (n1:.c1:.h1:.w1:.Z) (n2:.c2:.h2:.w2:.Z) (padh,padw) (strh,strw) =
  n1:.n2:.convOutDim h1 h2 padh strh:.convOutDim w1 w2 padw strw:.Z
  where convOutDim input_dim filter_dim padding stride =
          1 + (input_dim + (2 * padding) - filter_dim) `div` stride

-- convolution
convolution2dFwd :: forall t m a
                 . (PrimMonad m, MonadError t m, Variant t AllocFailed,
                    Variant t BadParam, Variant t NotSupported,
                    Variant t MappingError, Variant t ExecutionFailed,
                    Variant t OutOfMemory, Variant t MemoryAllocation,
                    TensorScalar a)
                 => CuDNN.Handle
                 -> (Int,Int)
                 -> (Int,Int)
                 -> CuDNN.ConvolutionFwdAlgo
                 -> MTensor (PrimState m) 4 a
                 -> MTensor (PrimState m) 4 a
                 -> m (MTensor (PrimState m) 4 a)
convolution2dFwd handle (padh,padw) (strh,strw) algo fmaps filters = do
  -- make the descriptors
  let outshp = convOutShape (MT.shape fmaps) (MT.shape filters) (padh,padw) (strh,strw)
  eres <- unsafeIOToPrim $ runExceptT $
    withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
      withTensor4d (unsafeCoerce fmaps :: IOTensor 4 a)
        $ \inputdesc inputptr -> do
        withFilter4d (unsafeCoerce filters :: IOTensor 4 a)
          $ \filtersdesc filtersptr -> do
          output <- emptyTensor outshp
          withTensor4d output $ \outputdesc outputptr -> do
            -- allocate workspace
            wkspcsizeptr <- liftIO $ malloc
            workspacesize <- do
              handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy NotSupported)
                $ liftIO $ CuDNN.getConvolutionForwardWorkspaceSize
                handle inputdesc filtersdesc convdesc outputdesc algo
                wkspcsizeptr
              liftIO $ peek wkspcsizeptr
            liftIO $ free wkspcsizeptr
            workspace <- attemptGCThenRetryOn (Proxy :: Proxy MemoryAllocation)
                         $ handleCUDAException (Proxy :: Proxy MemoryAllocation)
                         $ CUDA.mallocArray (fromIntegral workspacesize)
            -- allocate alpha and beta
            alpha <- liftIO $ newArray [1]
            beta <- liftIO $ newArray [0]
            -- finally run the damn thing
            handleStatus (Proxy :: Proxy BadParam)
              $ handleStatus (Proxy :: Proxy NotSupported)
              $ handleStatus (Proxy :: Proxy MappingError)
              $ handleStatus (Proxy :: Proxy ExecutionFailed)
              $ liftIO $ CuDNN.convolutionForward
              handle alpha inputdesc inputptr filtersdesc filtersptr
              convdesc algo workspace workspacesize beta outputdesc
              outputptr
            liftIO $ free alpha
            liftIO $ free beta
            liftIO $ CUDA.free workspace
          return $ unsafeCoerce output
  embedExcept eres

convolution2dBwdFilters :: forall t m a
                        . (PrimMonad m, MonadError t m, Variant t BadParam,
                           Variant t AllocFailed, Variant t NotSupported,
                           Variant t MappingError, Variant t ExecutionFailed,
                           Variant t OutOfMemory, TensorScalar a)
                        => CuDNN.Handle
                        -> (Int,Int)
                        -> (Int,Int)
                        -> MTensor (PrimState m) 4 a
                        -> MTensor (PrimState m) 4 a
                        -> MTensor (PrimState m) 4  a
                        -> m (MTensor (PrimState m) 4 a)
convolution2dBwdFilters handle (padh,padw) (strh,strw) fmaps filters upgrad = do
  -- make the descriptors
  eres <- unsafeIOToPrim $ runExceptT $
          withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
            withTensor4d (unsafeCoerce fmaps :: IOTensor 4 a)
            $ \inputdesc inputptr -> do
              withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
              $ \upgraddesc upgradptr -> do
                alpha <- liftIO $ newArray [1]
                beta <- liftIO $ newArray [0]
                -- compute gradient with regards to the filters
                filtersgrad <- emptyTensor $ MT.shape filters
                withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr ->
                  handleStatus (Proxy :: Proxy BadParam)
                  $ handleStatus (Proxy :: Proxy NotSupported)
                  $ handleStatus (Proxy :: Proxy MappingError)
                  $ handleStatus (Proxy :: Proxy ExecutionFailed)
                  $ liftIO
                  $ CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
                  upgraddesc upgradptr convdesc beta filtersgraddesc filtersgradptr
                liftIO $ free alpha
                liftIO $ free beta
                return $ unsafeCoerce filtersgrad
  embedExcept eres

convolution2dBwdInputs :: forall t m a
                       . (PrimMonad m, MonadError t m, Variant t BadParam,
                          Variant t NotSupported, Variant t AllocFailed,
                          Variant t MappingError, Variant t ExecutionFailed,
                          Variant t OutOfMemory, TensorScalar a)
                       => CuDNN.Handle
                       -> (Int,Int)
                       -> (Int,Int)
                       -> MTensor (PrimState m) 4 a
                       -> MTensor (PrimState m) 4 a
                       -> MTensor (PrimState m) 4 a
                       -> m (MTensor (PrimState m) 4 a)
convolution2dBwdInputs handle (padh,padw) (strh,strw) fmaps filters upgrad = do
  -- make the descriptors
  eres <- unsafeIOToPrim $ runExceptT $
          withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
            withFilter4d (unsafeCoerce filters :: IOTensor 4 a)
            $ \filtersdesc filtersptr -> do
              withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
              $ \upgraddesc upgradptr -> do
                alpha <- liftIO $ newArray [1]
                beta <- liftIO $ newArray [0]
                -- compute gradient with regards to the input feature maps
                inputsgrad <- emptyTensor $ MT.shape fmaps
                withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr ->
                  handleStatus (Proxy :: Proxy BadParam)
                  $ handleStatus (Proxy :: Proxy NotSupported)
                  $ handleStatus (Proxy :: Proxy MappingError)
                  $ handleStatus (Proxy :: Proxy ExecutionFailed)
                  $ liftIO
                  $ CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
                  upgraddesc upgradptr convdesc beta inputsgraddesc inputsgradptr
                liftIO $ free alpha
                liftIO $ free beta
                return $ unsafeCoerce inputsgrad
  embedExcept eres

-- bias
biasForward :: forall m a . (MonadCudaError m, PrimMonad m, TensorScalar a)
            => CuDNN.Handle
            -> MTensor (PrimState m) 4 a
            -> MTensor (PrimState m) 4 a
            -> m (MTensor (PrimState m) 4 a)
biasForward handle bias batch = embedCudaError unsafeIOToPrim $ do
  withTensor4d (unsafeCoerce bias :: IOTensor 4 a) $ \biasdesc biasptr -> do
    out <- MT.copy (unsafeCoerce batch :: IOTensor 4 a)
    withTensor4d out $ \outdesc outptr -> do
      (alpha, beta) <- liftIO $ pure (,) <*> newArray [1] <*> newArray [1]
      handleStatus (Proxy :: Proxy NotSupported)
        $ handleStatus (Proxy :: Proxy BadParam)
        $ handleStatus (Proxy :: Proxy ExecutionFailed)
        $ liftIO
        $ CuDNN.addTensor handle CuDNN.add_same_c alpha biasdesc biasptr
        beta outdesc outptr
      liftIO $ free alpha >> free beta
    return $ unsafeCoerce out

biasBackward :: forall m a . (MonadCudaError m, PrimMonad m, TensorScalar a)
             => CuDNN.Handle
             -> MTensor (PrimState m) 4 a
             -> m (MTensor (PrimState m) 4 a)
biasBackward handle upgrad = embedCudaError unsafeIOToPrim $ do
  withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)$ \upgraddesc upgradptr -> do
    let _:.c:._ = MT.shape upgrad
    biasgrad <- emptyTensor $ 1:.c:.1:.1:.Z
    withTensor4d biasgrad $ \biasgraddesc biasgradptr -> do
      (alpha, beta) <- liftIO $ pure (,) <*> newArray [1] <*> newArray [0]
      handleStatus (Proxy :: Proxy BadParam)
        $ liftIO
        $ CuDNN.convolutionBackwardBias handle alpha upgraddesc upgradptr
        beta biasgraddesc biasgradptr
      liftIO $ free alpha >> free beta
      return $ unsafeCoerce biasgrad

-- activations
activationFwd :: forall t m a
              . (PrimMonad m, MonadError t m, Variant t BadParam,
                 Variant t ExecutionFailed, Variant t AllocFailed,
                 Variant t NotSupported, Variant t OutOfMemory, TensorScalar a)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
activationFwd handle mode input = do
  eres <- unsafeIOToPrim $ runExceptT $
          withTensor4d (unsafeCoerce input :: IOTensor 4 a) $ \inputdesc inputptr -> do
            output <- emptyTensor $ MT.shape input
            withTensor4d output $ \outputdesc outputptr -> do
              alpha <- liftIO $ newArray [1]
              beta <- liftIO $ newArray [0]
              handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy ExecutionFailed)
                $ liftIO
                $ CuDNN.activationForward handle mode alpha inputdesc
                inputptr beta outputdesc outputptr
              liftIO $ free alpha
              liftIO $ free beta
            return $ unsafeCoerce output
  embedExcept eres

activationBwd :: forall t m a
              . (PrimMonad m, MonadError t m, Variant t BadParam,
                 Variant t NotSupported, Variant t ExecutionFailed,
                 Variant t AllocFailed, Variant t OutOfMemory, TensorScalar a)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) 4 a
              -> MTensor (PrimState m) 4 a
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
activationBwd handle mode input output upgrad = do
  eres <- unsafeIOToPrim $ runExceptT $
          withTensor4d (unsafeCoerce input :: IOTensor 4 a) $ \inputdesc inputptr -> do
            withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a) $ \upgraddesc upgradptr -> do
              grad <- emptyTensor $ MT.shape input
              withTensor4d (unsafeCoerce output :: IOTensor s a) $ \outputdesc outputptr -> do
                withTensor4d grad $ \graddesc gradptr -> do
                  alpha <- liftIO $ newArray [1]
                  beta <- liftIO $ newArray [0]
                  handleStatus (Proxy :: Proxy BadParam)
                    $ handleStatus (Proxy :: Proxy NotSupported)
                    $ handleStatus (Proxy :: Proxy ExecutionFailed)
                    $ liftIO
                    $ CuDNN.activationBackward handle mode alpha inputdesc inputptr
                    upgraddesc upgradptr outputdesc outputptr beta graddesc
                    gradptr
                  liftIO $ free alpha
                  liftIO $ free beta
              return $ unsafeCoerce grad
  embedExcept eres

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

pooling2dFwd :: forall t m a
             . (PrimMonad m, MonadError t m, Variant t AllocFailed,
                Variant t BadParam, Variant t NotSupported,
                Variant t ExecutionFailed, Variant t OutOfMemory, TensorScalar a)
             => CuDNN.Handle
             -> (Int,Int)
             -> (Int,Int)
             -> (Int,Int)
             -> CuDNN.PoolingMode
             -> MTensor (PrimState m) 4 a
             -> m (MTensor (PrimState m) 4 a)
pooling2dFwd handle size padding stride mode input = do
  let outshp = pooling2dOutputShape size padding stride $ MT.shape input
  eres <- unsafeIOToPrim $ runExceptT $
          withPoolDesc size padding stride mode $ \pooldesc -> do
            withTensor4d (unsafeCoerce input :: IOTensor 4 a)
            $ \inputdesc inputptr -> do
              output <- emptyTensor outshp
              -- actual pooling
              withTensor4d output $ \outputdesc outputptr -> do
                alpha <- liftIO $ newArray [1]
                beta <- liftIO $ newArray [0]
                handleStatus (Proxy :: Proxy BadParam)
                  $ handleStatus (Proxy :: Proxy NotSupported)
                  $ handleStatus (Proxy :: Proxy ExecutionFailed)
                  $ liftIO
                  $ CuDNN.poolingForward handle pooldesc alpha inputdesc
                  inputptr beta outputdesc outputptr
                liftIO $ free alpha
                liftIO $ free beta
              return $ unsafeCoerce output
  embedExcept eres

pooling2dBwd :: forall t m a
             . (PrimMonad m, MonadError t m, Variant t AllocFailed,
                Variant t BadParam, Variant t NotSupported,
                Variant t ExecutionFailed, Variant t OutOfMemory, TensorScalar a)
             => CuDNN.Handle
             -> (Int,Int)
             -> (Int,Int)
             -> (Int,Int)
             -> CuDNN.PoolingMode
             -> MTensor (PrimState m) 4 a
             -> MTensor (PrimState m) 4 a
             -> MTensor (PrimState m) 4 a
             -> m (MTensor (PrimState m) 4 a)
pooling2dBwd handle size padding stride mode inp out upgrad = do
  eres <- unsafeIOToPrim $ runExceptT $
          withPoolDesc size padding stride mode $ \pooldesc -> do
            withTensor4d (unsafeCoerce inp :: IOTensor 4 a)
            $ \inpdesc inpptr -> do
              withTensor4d (unsafeCoerce out :: IOTensor 4 a)
              $ \outdesc outptr -> do
                withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
                $ \upgraddesc upgradptr -> do
                  grad <- emptyTensor $ MT.shape inp
                  withTensor4d grad $ \graddesc gradptr -> do
                    alpha <- liftIO $ newArray [1]
                    beta <- liftIO $ newArray [0]
                    handleStatus (Proxy :: Proxy BadParam)
                      $ handleStatus (Proxy :: Proxy NotSupported)
                      $ handleStatus (Proxy :: Proxy ExecutionFailed)
                      $ liftIO
                      $ CuDNN.poolingBackward handle pooldesc alpha outdesc outptr
                      upgraddesc upgradptr inpdesc inpptr beta graddesc gradptr
                  return $ unsafeCoerce grad
  embedExcept eres

-- -- Softmax
-- softmaxFwd :: forall m a . (PrimMonad m, TensorScalar a)
--            => CuDNN.Handle
--            -> CuDNN.SoftmaxAlgorithm
--            -> CuDNN.SoftmaxMode
--            -> MTensor (PrimState m) 2 a
--            -> m (MTensor (PrimState m) 2 a)
-- softmaxFwd handle algo mode input = unsafeIOToPrim $ do
--   withTensor4d (unsafeCoerce input :: IOTensor s a)$ \inpdesc inpptr -> do
--     output <- emptyTensor :: IO (IOTensor s a)
--     withTensor4d output $ \outdesc outptr -> do
--       withArray [1] $ \alpha -> withArray [0] $ \beta -> do
--         handleError "Couldn't compute softmax." $
--           CuDNN.softmaxForward handle algo mode alpha
--           inpdesc inpptr beta outdesc outptr
--         return $ unsafeCoerce output

-- softmaxBwd :: forall m s a . (PrimMonad m, MT.Shape s, Nbdim s ~ 4, TensorScalar a)
--            => CuDNN.Handle
--            -> CuDNN.SoftmaxAlgorithm
--            -> CuDNN.SoftmaxMode
--            -> MTensor (PrimState m) s a
--            -> MTensor (PrimState m) s a
--            -> m (MTensor (PrimState m) s a)
-- softmaxBwd handle algo mode src srcdiff = unsafeIOToPrim $ do
--   withTensor4d (unsafeCoerce src :: IOTensor s a) $ \srcdesc srcdata -> do
--     withTensor4d (unsafeCoerce srcdiff :: IOTensor s a)
--       $ \srcdiffdesc srcdiffdata -> do
--       destdiff <- emptyTensor :: IO (IOTensor s a)
--       withTensor4d destdiff $ \destdiffdesc destdiffdata -> do
--         withArray [1] $ \alpha -> withArray [0] $ \beta -> do
--           handleError "Couldn't compute softmax backward pass." $
--             CuDNN.softmaxBackward handle algo mode alpha
--             srcdesc srcdata srcdiffdesc srcdiffdata beta
--             destdiffdesc destdiffdata
--           return $ unsafeCoerce destdiff

nchw_to_nhwc' :: forall t m a
              . (PrimMonad m, TensorScalar a, MonadError t m, Variant t BadParam,
                 Variant t ExecutionFailed, Variant t AllocFailed,
                 Variant t NotSupported, Variant t OutOfMemory)
              => CuDNN.Handle
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
nchw_to_nhwc' handle src = do
  eres <- unsafeIOToPrim $ runExceptT $ do
    res <- transformTensorIO handle CuDNN.nchw CuDNN.nhwc
      (unsafeCoerce src :: IOTensor 4 a)
    return $ unsafeCoerce res
  embedExcept eres

nhwc_to_nchw' :: forall t m a
              . (PrimMonad m, TensorScalar a, MonadError t m, Variant t BadParam,
                 Variant t ExecutionFailed, Variant t AllocFailed, Variant t OutOfMemory,
                 Variant t NotSupported)
              => CuDNN.Handle
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
nhwc_to_nchw' handle src = do
  eres <- unsafeIOToPrim $ runExceptT $ do
    res <- transformTensorIO handle CuDNN.nhwc CuDNN.nchw
      (unsafeCoerce src :: IOTensor 4 a)
    return $ unsafeCoerce res
  embedExcept eres

-- Unsafe but useful.
transformTensorIO :: forall t m a
                  . (TensorScalar a, MonadIO m, PrimMonad m, PrimState m ~ RealWorld,
                     MonadError t m, Variant t BadParam,
                     Variant t ExecutionFailed, Variant t AllocFailed,
                     Variant t NotSupported, Variant t OutOfMemory)
                  => CuDNN.Handle
                  -> CuDNN.TensorFormat
                  -> CuDNN.TensorFormat
                  -> IOTensor 4 a
                  -> m (IOTensor 4 a)
transformTensorIO handle srcf dstf src = do
  datatype <- MT.dtype src
  let [a',b',c',d'] = dimensions $ MT.shape src
      [n,c,h,w] = if srcf == CuDNN.nchw then [a',b',c',d']
                  else if srcf == CuDNN.nhwc then [a',d',b',c']
                       else error $ "Unsupported input format: " ++ show srcf
      shp = (n:.c:.h:.w:.Z)
  withTensorDesc srcf datatype shp $ \srcdesc -> do
      withTensorDesc dstf datatype shp $ \dstdesc -> do
          dst <- emptyTensor $ if dstf == CuDNN.nchw
                               then n:.c:.h:.w:.Z
                               else n:.h:.w:.c:.Z
          eres1 <- liftIO $ withDevicePtr src $ \srcptr -> runExceptT $ do
            eres2 <- liftIO $ withDevicePtr dst $ \dstptr -> runExceptT $ do
              alpha <- liftIO $ newArray [1]
              beta <- liftIO $ newArray [0]
              handleStatus (Proxy :: Proxy BadParam)
                $ handleStatus (Proxy :: Proxy ExecutionFailed)
                $ liftIO
                $ CuDNN.transformTensor handle alpha srcdesc srcptr beta
                dstdesc dstptr
            embedExcept eres2
          embedExcept eres1
          return dst

global_cudnn_handle :: IORef (Maybe CuDNN.Handle)
{-# NOINLINE global_cudnn_handle #-}
global_cudnn_handle = unsafePerformIO $ newIORef Nothing

cudnnHandle :: CuDNN.Handle
cudnnHandle = unsafePerformIO $ do
  mh <- readIORef global_cudnn_handle
  case mh of
   Nothing -> do
     h <- alloca $ \hptr -> CuDNN.createHandle hptr >> peek hptr
     writeIORef global_cudnn_handle $ Just h
     return h
   Just h -> return h
