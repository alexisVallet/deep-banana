{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA.CuDNN (
    convolution2d
  , CuDNN.convolution_fwd_algo_implicit_gemm
  , CuDNN.convolution_fwd_algo_implicit_precomp_gemm
  , CuDNN.convolution_fwd_algo_gemm
  , CuDNN.convolution_fwd_algo_direct
  , ConvOutShape
  , activation
  , CuDNN.activation_sigmoid
  , CuDNN.activation_relu
  , CuDNN.activation_tanh
  , pooling2d
  , CuDNN.pooling_max
  , CuDNN.pooling_average_count_include_padding
  , CuDNN.pooling_average_count_exclude_padding
  , PoolOutShape
  , nhwc_to_nchw
  , nchw_to_nhwc
  ) where
import Foreign
import Foreign.C
import Foreign.Storable
import Foreign.Marshal
import Foreign.Marshal.Array
import qualified Foreign.CUDA as CUDA
import qualified Foreign.CUDA.CuDNN as CuDNN
import Control.Exception
import Control.Monad
import Control.Monad.Primitive
import Unsafe.Coerce
import GHC.TypeLits
import Data.Proxy
import Control.Monad.ST

import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Tensor
import qualified DeepBanana.Tensor.Mutable as MT
import DeepBanana.Tensor.Mutable (MTensor, IOTensor, withDevicePtr, emptyTensor)

convolution2d :: (TensorScalar a, Shape input_shape, Shape filter_shape,
                  Shape padding, Shape stride, Shape out_shape,
                  out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                  Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4, Nbdim out_shape ~ 4,
                  Nbdim stride ~ 2, Nbdim padding ~ 2)
              => Proxy [padding,stride]
              -> CuDNN.ConvolutionFwdAlgo
              -> Layer CUDA a
                   '[Tensor filter_shape a]
                   (Tensor input_shape a)
                   (Tensor out_shape a)
convolution2d p algo =
  combinePasses convfwd convbwd
  where convfwd (HLS (HCons filters HNil)) fmaps = do
          handle <- asks cudnnHandle
          return $ runST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd p handle algo fmaps' filters'
            unsafeFreeze convres
        convbwd (HLS (HCons filters HNil)) fmaps _ = do
          handle <- asks cudnnHandle
          let bwdfilters upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                filtersgrad <- convolution2dBwdFilters p handle fmaps' filters' upgrad'
                unsafeFreeze filtersgrad
              bwdinputs upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                inputsgrad <- convolution2dBwdInputs p handle fmaps' filters' upgrad'
                unsafeFreeze inputsgrad
          return $ \upgrad -> (HLS $ bwdfilters upgrad `HCons` HNil, bwdinputs upgrad)

activation :: (TensorScalar a, Shape s, Nbdim s ~ 4)
           => CuDNN.ActivationMode
           -> Layer CUDA a '[] (Tensor s a) (Tensor s a)
activation mode =
  combinePasses' actfwd actbwd
  where actfwd fmaps = do
          handle <- asks cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            activations <- activationFwd handle mode fmaps'
            unsafeFreeze activations
        actbwd inp out = do
          handle <- asks cudnnHandle
          return $ \upgrad -> runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- activationBwd handle mode inp' out' upgrad'
            unsafeFreeze grad

-- pooling
pooling2d :: (TensorScalar a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4, Nbdim pooling_size ~ 2,
              Nbdim padding ~ 2, Nbdim stride ~ 2)
          => Proxy [pooling_size,padding,stride]
          -> CuDNN.PoolingMode
          -> Layer CUDA a '[] (Tensor input_shape a) (Tensor out_shape a)
pooling2d p mode =
  combinePasses' poolfwd poolbwd
  where poolfwd fmaps = do
          handle <- asks cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd p handle mode fmaps'
            unsafeFreeze poolres
        poolbwd inp out = do
          handle <- asks cudnnHandle
          return $ \upgrad -> runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- pooling2dBwd p handle mode inp' out' upgrad'
            unsafeFreeze grad

nchw_to_nhwc :: forall n c h w a
             . (TensorScalar a, KnownNat n, KnownNat c, KnownNat h, KnownNat w)
             => Layer CUDA a '[] (Tensor [n,c,h,w] a) (Tensor [n,h,w,c] a)
nchw_to_nhwc = combinePasses' fwdTrans bwdTrans
  where fwdTrans t = do
          handle <- asks cudnnHandle
          return $ runST $ do
            mt <- unsafeThaw t
            mt' <- nchw_to_nhwc' handle mt
            unsafeFreeze mt'
        bwdTrans _ _ = do
          handle <- asks cudnnHandle
          return $ \upgrad -> runST $ do
            mu <- unsafeThaw upgrad
            mu' <- nhwc_to_nchw' handle mu
            unsafeFreeze mu'

nhwc_to_nchw :: forall n c h w a
                . (TensorScalar a, KnownNat n, KnownNat c, KnownNat h, KnownNat w)
                => Layer CUDA a '[] (Tensor [n,h,w,c] a) (Tensor [n,c,h,w] a)
nhwc_to_nchw = combinePasses' fwdTrans bwdTrans
  where fwdTrans t = do
          handle <- asks cudnnHandle
          return $ runST $ do
            mt <- unsafeThaw t
            mt' <- nhwc_to_nchw' handle mt
            unsafeFreeze mt'
        bwdTrans _ _ = do
          handle <- asks cudnnHandle
          return $ \upgrad -> runST $ do
            mu <- unsafeThaw upgrad
            mu' <- nchw_to_nhwc' handle mu
            unsafeFreeze mu'

-- Helper functions to deal with low-level boilerplate of CuDNN.
handleError :: String -> IO CuDNN.Status -> IO ()
handleError errMsg action = do
  status <- action
  when (status /= CuDNN.success) $ do
    errStr <- CuDNN.getErrorString status >>= peekCString
    ioError $ userError $ errStr ++ " (" ++ errMsg ++ ")"

withDescriptor :: (Storable desc)
               => String
               -> (Ptr desc -> IO CuDNN.Status) -- creation fct
               -> (desc -> IO CuDNN.Status) -- set fct
               -> (desc -> IO CuDNN.Status) -- destroy fct
               -> (desc -> IO a) -- action to perform
               -> IO a
withDescriptor name create set destroy action = do
  desc <- alloca $ \descptr -> do
    handleError ("Couldn't create " ++ name ++ " descriptor.") $ create descptr
    peek descptr
  handleError ("Couldn't set " ++ name ++ " descriptor.") $ set desc
  x <- action desc
  handleError ("Couldn't destroy " ++ name ++ " descriptor.") $ destroy desc
  return x

withTensor4d :: forall a b s . (TensorScalar a, Shape s, Nbdim s ~ 4)
             => IOTensor s a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withTensor4d tensor action = do
  datatype <- MT.dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ dimensions (Proxy :: Proxy s)
  withDescriptor
    "tensor"
    CuDNN.createTensorDescriptor
    (\d -> CuDNN.setTensor4dDescriptor d CuDNN.nchw datatype n c h w)
    CuDNN.destroyTensorDescriptor $ \tensordesc -> withDevicePtr tensor $ \dvcptr -> do
      action tensordesc dvcptr
                
withFilter4d :: forall a b s . (TensorScalar a, Shape s, Nbdim s ~ 4)
             => IOTensor s a
             -> (CuDNN.FilterDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withFilter4d tensor action = do
  datatype <- MT.dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ dimensions (Proxy :: Proxy s)
  withDescriptor
    "filter"
    CuDNN.createFilterDescriptor
    (\d -> CuDNN.setFilter4dDescriptor d datatype n c h w)
    CuDNN.destroyFilterDescriptor $ \filtdesc -> withDevicePtr tensor $ \dvcptr -> do
      action filtdesc dvcptr

withConvDesc :: (Int,Int) -> (Int,Int) -> (Int,Int)
             -> (CuDNN.ConvolutionDescriptor -> IO a) -> IO a
withConvDesc (padh,padw) (strh,strw) (uph,upw) = do
  let [cpadh,cpadw,cstrh,cstrw,cupw,cuph] =
        map fromIntegral [padh,padw,strh,strw,uph,upw]
  withDescriptor
    "convolution"
    CuDNN.createConvolutionDescriptor
    (\d -> CuDNN.setConvolution2dDescriptor d cpadh cpadw cstrh cstrw cupw cuph CuDNN.convolution)
    CuDNN.destroyConvolutionDescriptor

-- CuDNN bindings with mutable tensors.
type family ConvOutDim input_dim filter_dim padding stride where
  ConvOutDim input_dim filter_dim padding stride =
    1 + Quotient (input_dim + (2 * padding) - filter_dim) stride

type family ConvOutShape input_shape filter_shape padding stride where
  ConvOutShape [n1,c1,h1,w1] [n2,c1,h2,w2] [padh,padw] [strh,strw] =
    [n1,n2,ConvOutDim h1 h2 padh strh,ConvOutDim w1 w2 padw strw]

-- convolution
convolution2dFwd :: forall input_shape filter_shape padding stride out_shape m a
                 . (PrimMonad m, TensorScalar a, Shape input_shape,
                    Shape filter_shape, Shape padding, Shape stride, Shape out_shape,
                    out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                    Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4, Nbdim out_shape ~ 4)
                 => Proxy [padding,stride]
                 -> CuDNN.Handle
                 -> CuDNN.ConvolutionFwdAlgo
                 -> MTensor (PrimState m) input_shape a
                 -> MTensor (PrimState m) filter_shape a
                 -> m (MTensor (PrimState m) out_shape a)
convolution2dFwd p handle algo fmaps filters = unsafePrimToPrim $ do
  -- Getting padding and stride into the value-level realm.
  let [padh,padw] = dimensions (Proxy :: Proxy padding)
      [strh,strw] = dimensions (Proxy :: Proxy stride)
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withTensor4d (unsafeCoerce fmaps :: IOTensor input_shape a)
          $ \inputdesc inputptr -> do
          withFilter4d (unsafeCoerce filters :: IOTensor filter_shape a)
            $ \filtersdesc filtersptr -> do
            output <- emptyTensor :: IO (IOTensor out_shape a)
            withTensor4d output $ \outputdesc outputptr -> do
              -- allocate workspace
              workspacesize <- alloca $ \wkspcsizeptr -> do
                handleError "Couldn't compute workspace size." $
                  CuDNN.getConvolutionForwardWorkspaceSize
                  handle inputdesc filtersdesc convdesc outputdesc algo
                  wkspcsizeptr
                peek wkspcsizeptr
              CUDA.allocaArray (fromIntegral workspacesize) $ \workspace -> do
                -- allocate alpha and beta
                withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                  -- finally run the damn thing
                  handleError "Couldn't compute convolution." $
                    CuDNN.convolutionForward
                    handle alpha inputdesc inputptr filtersdesc filtersptr
                    convdesc algo workspace workspacesize beta outputdesc
                    outputptr
            return $ unsafeCoerce output
  runConv `catch` \e -> do
    error $ "Exception thrown in convolution forward pass: " ++ show (e :: SomeException) ++ "\n filter shape: " ++ show (dimensions (Proxy :: Proxy filter_shape)) ++ ", image shape: " ++ show (dimensions (Proxy :: Proxy input_shape))

convolution2dBwdFilters :: forall input_shape filter_shape padding stride out_shape m a
                         . (PrimMonad m, TensorScalar a, Shape input_shape,
                            Shape filter_shape, Shape padding, Shape stride,
                            Shape out_shape,
                            out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                            Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                            Nbdim out_shape ~ 4)
                         => Proxy [padding,stride]
                         -> CuDNN.Handle
                         -> MTensor (PrimState m) input_shape a
                         -> MTensor (PrimState m) filter_shape a
                         -> MTensor (PrimState m) out_shape  a
                         -> m (MTensor (PrimState m) filter_shape a)
convolution2dBwdFilters p handle fmaps filters upgrad = unsafePrimToPrim $ do
  -- Getting padding and stride into the value-level realm.
  let [padh,padw] = dimensions (Proxy :: Proxy padding)
      [strh,strw] = dimensions (Proxy :: Proxy stride)
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withTensor4d (unsafeCoerce fmaps :: IOTensor input_shape a)
          $ \inputdesc inputptr -> do
          withTensor4d (unsafeCoerce upgrad :: IOTensor out_shape a)
            $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the filters
              filtersgrad <- emptyTensor :: IO (IOTensor filter_shape a)
              withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to filters." $
                CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
                upgraddesc upgradptr convdesc beta filtersgraddesc filtersgradptr
              return $ unsafeCoerce filtersgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to filters: " ++ show (e :: IOException)

convolution2dBwdInputs :: forall input_shape filter_shape padding stride out_shape m a
                       . (PrimMonad m, TensorScalar a, Shape input_shape,
                          Shape filter_shape, Shape padding, Shape stride,
                          Shape out_shape,
                          out_shape ~ ConvOutShape input_shape filter_shape padding stride,
                          Nbdim input_shape ~ 4, Nbdim filter_shape ~ 4,
                          Nbdim out_shape ~ 4)
                       => Proxy [padding,stride]
                       -> CuDNN.Handle
                       -> MTensor (PrimState m) input_shape a
                       -> MTensor (PrimState m) filter_shape a
                       -> MTensor (PrimState m) out_shape a
                       -> m (MTensor (PrimState m) input_shape a)
convolution2dBwdInputs p handle fmaps filters upgrad = unsafePrimToPrim $ do
  -- Getting padding and stride into the value-level realm.
  let [padh,padw] = dimensions (Proxy :: Proxy padding)
      [strh,strw] = dimensions (Proxy :: Proxy stride)
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withFilter4d (unsafeCoerce filters :: IOTensor filter_shape a)
          $ \filtersdesc filtersptr -> do
          withTensor4d (unsafeCoerce upgrad :: IOTensor out_shape a)
            $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the input feature maps
              inputsgrad <- emptyTensor :: IO (IOTensor input_shape a)
              withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to the inputs." $
                CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
                upgraddesc upgradptr convdesc beta inputsgraddesc inputsgradptr
              return $ unsafeCoerce inputsgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to data: " ++ show (e :: IOException)

-- activations
activationFwd :: forall s m a . (PrimMonad m, Shape s, TensorScalar a, Nbdim s ~ 4)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) s a
              -> m (MTensor (PrimState m) s a)
activationFwd handle mode input = unsafePrimToPrim $ do
  withTensor4d (unsafeCoerce input :: IOTensor s a) $ \inputdesc inputptr -> do
    output <- emptyTensor :: IO (IOTensor s a)
    withTensor4d output $ \outputdesc outputptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute activations." $
          CuDNN.activationForward handle mode alpha inputdesc
          inputptr beta outputdesc outputptr
        return $ unsafeCoerce output

activationBwd :: forall s m a . (PrimMonad m, Shape s, TensorScalar a, Nbdim s ~ 4)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) s a
              -> MTensor (PrimState m) s a
              -> MTensor (PrimState m) s a
              -> m (MTensor (PrimState m) s a)
activationBwd handle mode input output upgrad = unsafePrimToPrim $ do
  withTensor4d (unsafeCoerce input :: IOTensor s a) $ \inputdesc inputptr -> do
    withTensor4d (unsafeCoerce upgrad :: IOTensor s a) $ \upgraddesc upgradptr -> do
      grad <- emptyTensor :: IO (IOTensor s a)
      withTensor4d (unsafeCoerce output :: IOTensor s a) $ \outputdesc outputptr -> do
        withTensor4d grad $ \graddesc gradptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute activation backward pass." $
              CuDNN.activationBackward handle mode alpha inputdesc inputptr
              upgraddesc upgradptr outputdesc outputptr beta graddesc
              gradptr
            return $ unsafeCoerce grad

-- 2d pooling
-- Helper to compute output shape.
pooling2dOutputShape :: (Int,Int) -> (Int,Int) -> (Int,Int) -> [Int] -> [Int]
pooling2dOutputShape (szr,szc) (padr,padc) (strr, strc) [n,ch,r,c] =
  [n,ch,outr,outc]
  where
    inr = r + padr
    inc = c + padc
    outr = (inr - overlap_r) `div` strr
    outc = (inc - overlap_c) `div` strc
    overlap_r = max 0 (szr - strr)
    overlap_c = max 0 (szc - strc)

-- Pooling output shape, at the type level.
type family PoolOutDim input_size pooling_size padding stride where
  PoolOutDim input_size pooling_size padding stride =
    Quotient (input_size + padding - Max 0 (pooling_size - stride)) stride

type family PoolOutShape input_shape pooling_shape padding stride where
  PoolOutShape [n,c,h,w] [poolh,poolw] [padh,padw] [strh,strw] =
    [n,c,PoolOutDim h poolh padh strh,PoolOutDim w poolw padw strw]

pooling2dFwd :: forall input_shape pooling_size padding stride out_shape m a
           . (PrimMonad m, TensorScalar a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4)
           => Proxy [pooling_size,padding,stride]
           -> CuDNN.Handle
           -> CuDNN.PoolingMode
           -> MTensor (PrimState m) input_shape a
           -> m (MTensor (PrimState m) out_shape a)
pooling2dFwd p handle mode input = unsafePrimToPrim $ do
  -- pooling descriptor
  let [cwh,cww] = fmap fromIntegral $ dimensions (Proxy :: Proxy pooling_size)
      [cpadh,cpadw] = fmap fromIntegral $ dimensions (Proxy :: Proxy padding)
      [cstrh,cstrw] = fmap fromIntegral $ dimensions (Proxy :: Proxy stride)
  withDescriptor
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d (unsafeCoerce input :: IOTensor input_shape a)
        $ \inputdesc inputptr -> do
        output <- emptyTensor :: IO (IOTensor out_shape a)
        -- actual pooling
        withTensor4d output $ \outputdesc outputptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute pooling." $
              CuDNN.poolingForward handle pooldesc alpha inputdesc
              inputptr beta outputdesc outputptr
            return $ unsafeCoerce output

pooling2dBwd :: forall input_shape pooling_size padding stride out_shape m a
           . (PrimMonad m, TensorScalar a, Shape input_shape, Shape pooling_size,
              Shape padding, Shape stride, Shape out_shape,
              out_shape ~ PoolOutShape input_shape pooling_size padding stride,
              Nbdim input_shape ~ 4, Nbdim out_shape ~ 4)
           => Proxy [pooling_size,padding,stride]
           -> CuDNN.Handle
           -> CuDNN.PoolingMode
           -> MTensor (PrimState m) input_shape a
           -> MTensor (PrimState m) out_shape a
           -> MTensor (PrimState m) out_shape a
           -> m (MTensor (PrimState m) input_shape a)
pooling2dBwd p handle mode inp out upgrad = unsafePrimToPrim $ do
  -- pooling descriptor
  let [cwh,cww] = fmap fromIntegral $ dimensions (Proxy :: Proxy pooling_size)
      [cpadh,cpadw] = fmap fromIntegral $ dimensions (Proxy :: Proxy padding)
      [cstrh,cstrw] = fmap fromIntegral $ dimensions (Proxy :: Proxy stride)
  withDescriptor
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d (unsafeCoerce inp :: IOTensor input_shape a)
        $ \inpdesc inpptr -> do
        withTensor4d (unsafeCoerce out :: IOTensor out_shape a)
          $ \outdesc outptr -> do
          withTensor4d (unsafeCoerce upgrad :: IOTensor out_shape a)
            $ \upgraddesc upgradptr -> do
            grad <- emptyTensor :: IO (IOTensor input_shape a)
            withTensor4d grad $ \graddesc gradptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't compute backward pooling." $
                  CuDNN.poolingBackward handle pooldesc alpha outdesc outptr
                  upgraddesc upgradptr inpdesc inpptr beta graddesc gradptr
              return $ unsafeCoerce grad

-- Softmax
softmaxFwd :: forall m s a . (PrimMonad m, Shape s, Nbdim s ~ 4, TensorScalar a)
           => CuDNN.Handle
           -> CuDNN.SoftmaxAlgorithm
           -> CuDNN.SoftmaxMode
           -> MTensor (PrimState m) s a
           -> m (MTensor (PrimState m) s a)
softmaxFwd handle algo mode input = unsafePrimToPrim $ do
  withTensor4d (unsafeCoerce input :: IOTensor s a)$ \inpdesc inpptr -> do
    output <- emptyTensor :: IO (IOTensor s a)
    withTensor4d output $ \outdesc outptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute softmax." $
          CuDNN.softmaxForward handle algo mode alpha
          inpdesc inpptr beta outdesc outptr
        return $ unsafeCoerce output

softmaxBwd :: forall m s a . (PrimMonad m, Shape s, Nbdim s ~ 4, TensorScalar a)
           => CuDNN.Handle
           -> CuDNN.SoftmaxAlgorithm
           -> CuDNN.SoftmaxMode
           -> MTensor (PrimState m) s a
           -> MTensor (PrimState m) s a
           -> m (MTensor (PrimState m) s a)
softmaxBwd handle algo mode src srcdiff = unsafePrimToPrim $ do
  withTensor4d (unsafeCoerce src :: IOTensor s a) $ \srcdesc srcdata -> do
    withTensor4d (unsafeCoerce srcdiff :: IOTensor s a)
      $ \srcdiffdesc srcdiffdata -> do
      destdiff <- emptyTensor :: IO (IOTensor s a)
      withTensor4d destdiff $ \destdiffdesc destdiffdata -> do
        withArray [1] $ \alpha -> withArray [0] $ \beta -> do
          handleError "Couldn't compute softmax backward pass." $
            CuDNN.softmaxBackward handle algo mode alpha
            srcdesc srcdata srcdiffdesc srcdiffdata beta
            destdiffdesc destdiffdata
          return $ unsafeCoerce destdiff

nchw_to_nhwc' :: forall n c h w a m
             . (PrimMonad m, TensorScalar a, KnownNat n, KnownNat c, KnownNat h,
                KnownNat w)
             => CuDNN.Handle
             -> MTensor (PrimState m) [n,c,h,w] a
             -> m (MTensor (PrimState m) [n,h,w,c] a)
nchw_to_nhwc' handle src = unsafePrimToPrim $ do
  res <- transformTensorIO handle CuDNN.nchw CuDNN.nhwc
         (unsafeCoerce src :: IOTensor [n,c,h,w] a) :: IO (IOTensor [n,h,w,c] a)
  return $ unsafeCoerce res

nhwc_to_nchw' :: forall n c h w a m
             . (PrimMonad m, TensorScalar a, KnownNat n, KnownNat c, KnownNat h,
                KnownNat w)
             => CuDNN.Handle
             -> MTensor (PrimState m) [n,h,w,c] a
             -> m (MTensor (PrimState m) [n,c,h,w] a)
nhwc_to_nchw' handle src = unsafePrimToPrim $ do
  res <- transformTensorIO handle CuDNN.nhwc CuDNN.nchw
         (unsafeCoerce src :: IOTensor [n,h,w,c] a) :: IO (IOTensor [n,c,h,w] a)
  return $ unsafeCoerce res

-- Unsafe but useful.
transformTensorIO :: forall a s1 s2
                  . (TensorScalar a, Shape s1, Shape s2,
                     Nbdim s1 ~ 4, Nbdim s2 ~ 4)
                  => CuDNN.Handle
                  -> CuDNN.TensorFormat
                  -> CuDNN.TensorFormat
                  -> IOTensor s1 a
                  -> IO (IOTensor s2 a)
transformTensorIO handle srcf dstf src = do
  datatype <- MT.dtype src
  let [a',b',c',d'] = dimensions (Proxy :: Proxy s1)
      [n,c,h,w] = if srcf == CuDNN.nchw then [a',b',c',d']
                  else if srcf == CuDNN.nhwc then [a',d',b',c']
                       else error $ "Unsupported input format: " ++ show srcf
      [cn,cc,ch,cw] = fmap fromIntegral [n,c,h,w]
  withDescriptor
    "tensor"
    CuDNN.createTensorDescriptor
    (\d -> CuDNN.setTensor4dDescriptor d srcf datatype cn cc ch cw)
    CuDNN.destroyTensorDescriptor $ \srcdesc -> do
      withDescriptor
        "tensor"
        CuDNN.createTensorDescriptor
        (\d -> CuDNN.setTensor4dDescriptor d dstf datatype cn cc ch cw)
        CuDNN.destroyTensorDescriptor $ \dstdesc -> do
          dst <- emptyTensor
          withDevicePtr src $ \srcptr -> do
            withDevicePtr dst $ \dstptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't transform the tensor." $
                  CuDNN.transformTensor handle alpha srcdesc srcptr beta
                  dstdesc dstptr
                return dst

addTensor :: forall a m s1 s2 . (TensorScalar a, PrimMonad m, Shape s1, Nbdim s1 ~ 4,
                                 Shape s2, Nbdim s2 ~ 4)
          => CuDNN.Handle
          -> CuDNN.AddMode
          -> MTensor (PrimState m) s1 a
          -> MTensor (PrimState m) s2 a
          -> m (MTensor (PrimState m) s1 a)
addTensor handle addMode bias src = unsafePrimToPrim $ do
  out <- MT.copy (unsafeCoerce src :: IOTensor s1 a)
  withTensor4d (unsafeCoerce bias :: IOTensor s2 a) $ \biasdesc biasptr -> do
    withTensor4d (unsafeCoerce out :: IOTensor s1 a) $ \outdesc outptr -> do
      withArray [1] $ \alpha -> withArray [1] $ \beta -> do
        handleError "Couldn't add the tensors." $
          CuDNN.addTensor handle addMode alpha biasdesc biasptr
          beta outdesc outptr
  return $ unsafeCoerce out

convolutionBackwardBias :: forall n c h w a m
                        . (TensorScalar a, PrimMonad m, KnownNat n, KnownNat c,
                           KnownNat h, KnownNat w)
                        => CuDNN.Handle
                        -> MTensor (PrimState m) [n,c,h,w] a
                        -> m (MTensor (PrimState m) [1,c,1,1] a)
convolutionBackwardBias handle upgrad = unsafePrimToPrim $ do
  biasgrad <- emptyTensor :: IO (IOTensor [1,c,1,1] a)
  withTensor4d (unsafeCoerce upgrad :: IOTensor [n,c,h,w] a)
    $ \upgraddesc upgradptr -> do
    withTensor4d biasgrad $ \biasgraddesc biasgradptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute convolution backward pass with regards to bias" $
          CuDNN.convolutionBackwardBias handle alpha upgraddesc upgradptr
          beta biasgraddesc biasgradptr
  return $ unsafeCoerce biasgrad
