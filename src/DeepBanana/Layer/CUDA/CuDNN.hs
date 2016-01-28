{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA.CuDNN (
    convolution2d
  , CuDNN.convolution_fwd_algo_implicit_gemm
  , CuDNN.convolution_fwd_algo_implicit_precomp_gemm
  , CuDNN.convolution_fwd_algo_gemm
  , CuDNN.convolution_fwd_algo_direct
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

convolution2d :: (TensorScalar a)
              => (Int,Int)
              -> (Int,Int)
              -> CuDNN.ConvolutionFwdAlgo
              -> Layer CUDA a '[Tensor 4 a] (Tensor 4 a) (Tensor 4 a)
convolution2d padding stride algo =
  combinePasses convfwd convbwd
  where convfwd (HLS (HCons filters HNil)) fmaps = do
          handle <- asks cudnnHandle
          return $ runST $ do
            filters' <- unsafeThaw filters
            fmaps' <- unsafeThaw fmaps
            convres <- convolution2dFwd handle padding stride algo fmaps' filters'
            unsafeFreeze convres
        convbwd (HLS (HCons filters HNil)) fmaps _ = do
          handle <- asks cudnnHandle
          let bwdfilters upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                filtersgrad <- convolution2dBwdFilters handle padding stride fmaps' filters' upgrad'
                unsafeFreeze filtersgrad
              bwdinputs upgrad = runST $ do
                filters' <- unsafeThaw filters
                fmaps' <- unsafeThaw fmaps
                upgrad' <- unsafeThaw upgrad
                inputsgrad <- convolution2dBwdInputs handle padding stride fmaps' filters' upgrad'
                unsafeFreeze inputsgrad
          return $ \upgrad -> (HLS $ bwdfilters upgrad `HCons` HNil, bwdinputs upgrad)

activation :: (TensorScalar a)
           => CuDNN.ActivationMode
           -> Layer CUDA a '[] (Tensor 4 a) (Tensor 4 a)
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
pooling2d :: (TensorScalar a)
          => (Int,Int)
          -> (Int,Int)
          -> (Int,Int)
          -> CuDNN.PoolingMode
          -> Layer CUDA a '[] (Tensor 4 a) (Tensor 4 a)
pooling2d psize padding stride mode =
  combinePasses' poolfwd poolbwd
  where poolfwd fmaps = do
          handle <- asks cudnnHandle
          return $ runST $ do
            fmaps' <- unsafeThaw fmaps
            poolres <- pooling2dFwd handle psize padding stride mode fmaps'
            unsafeFreeze poolres
        poolbwd inp out = do
          handle <- asks cudnnHandle
          return $ \upgrad -> runST $ do
            inp' <- unsafeThaw inp
            out' <- unsafeThaw out
            upgrad' <- unsafeThaw upgrad
            grad <- pooling2dBwd handle psize padding stride mode inp' out' upgrad'
            unsafeFreeze grad

nchw_to_nhwc :: (TensorScalar a)
             => Layer CUDA a '[] (Tensor 4 a) (Tensor 4 a)
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

nhwc_to_nchw :: (TensorScalar a)
                => Layer CUDA a '[] (Tensor 4 a) (Tensor 4 a)
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

withTensor4d :: (TensorScalar a)
             => IOTensor 4 a
             -> (CuDNN.TensorDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withTensor4d tensor action = do
  datatype <- MT.dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ dimensions $ MT.shape tensor
  withDescriptor
    "tensor"
    CuDNN.createTensorDescriptor
    (\d -> CuDNN.setTensor4dDescriptor d CuDNN.nchw datatype n c h w)
    CuDNN.destroyTensorDescriptor $ \tensordesc -> withDevicePtr tensor $ \dvcptr -> do
      action tensordesc dvcptr

withFilter4d :: (TensorScalar a)
             => IOTensor 4 a
             -> (CuDNN.FilterDescriptor -> CUDA.DevicePtr a -> IO b)
             -> IO b
withFilter4d tensor action = do
  datatype <- MT.dtype tensor
  let [n,c,h,w] = fmap fromIntegral $ dimensions $ MT.shape tensor
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

convOutShape :: Dim 4 -> Dim 4 -> (Int,Int) -> (Int,Int) -> Dim 4
convOutShape (n1:.c1:.h1:.w1:.Z) (n2:.c2:.h2:.w2:.Z) (padh,padw) (strh,strw) =
  n1:.n2:.convOutDim h1 h2 padh strh:.convOutDim w1 w2 padw strw:.Z
  where convOutDim input_dim filter_dim padding stride =
          1 + (input_dim + (2 * padding) - filter_dim) `div` stride

-- convolution
convolution2dFwd :: forall m a . (PrimMonad m, TensorScalar a)
                 => CuDNN.Handle
                 -> (Int,Int)
                 -> (Int,Int)
                 -> CuDNN.ConvolutionFwdAlgo
                 -> MTensor (PrimState m) 4 a
                 -> MTensor (PrimState m) 4 a
                 -> m (MTensor (PrimState m) 4 a)
convolution2dFwd handle (padh,padw) (strh,strw) algo fmaps filters = unsafePrimToPrim $ do
  -- make the descriptors
  let outshp = convOutShape (MT.shape fmaps) (MT.shape filters) (padh,padw) (strh,strw)
      runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withTensor4d (unsafeCoerce fmaps :: IOTensor 4 a)
          $ \inputdesc inputptr -> do
          withFilter4d (unsafeCoerce filters :: IOTensor 4 a)
            $ \filtersdesc filtersptr -> do
            output <- emptyTensor outshp :: IO (IOTensor 4 a)
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
    error $ "Exception thrown in convolution forward pass: " ++ show (e :: SomeException) ++ "\n filter shape: " ++ show (MT.shape filters) ++ ", image shape: " ++ show (MT.shape fmaps)

convolution2dBwdFilters :: forall m a . (PrimMonad m, TensorScalar a)
                        => CuDNN.Handle
                        -> (Int,Int)
                        -> (Int,Int)
                        -> MTensor (PrimState m) 4 a
                        -> MTensor (PrimState m) 4 a
                        -> MTensor (PrimState m) 4  a
                        -> m (MTensor (PrimState m) 4 a)
convolution2dBwdFilters handle (padh,padw) (strh,strw) fmaps filters upgrad = unsafePrimToPrim $ do
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withTensor4d (unsafeCoerce fmaps :: IOTensor 4 a)
          $ \inputdesc inputptr -> do
          withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
            $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the filters
              filtersgrad <- emptyTensor $ MT.shape filters :: IO (IOTensor 4 a)
              withFilter4d filtersgrad $ \filtersgraddesc filtersgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to filters." $
                CuDNN.convolutionBackwardFilter handle alpha inputdesc inputptr
                upgraddesc upgradptr convdesc beta filtersgraddesc filtersgradptr
              return $ unsafeCoerce filtersgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to filters: " ++ show (e :: IOException)

convolution2dBwdInputs :: forall m a . (PrimMonad m, TensorScalar a)
                       => CuDNN.Handle
                       -> (Int,Int)
                       -> (Int,Int)
                       -> MTensor (PrimState m) 4 a
                       -> MTensor (PrimState m) 4 a
                       -> MTensor (PrimState m) 4 a
                       -> m (MTensor (PrimState m) 4 a)
convolution2dBwdInputs handle (padh,padw) (strh,strw) fmaps filters upgrad = unsafePrimToPrim $ do
  -- make the descriptors
  let runConv = withConvDesc (padh,padw) (strh,strw) (1,1) $ \convdesc -> do
        withFilter4d (unsafeCoerce filters :: IOTensor 4 a)
          $ \filtersdesc filtersptr -> do
          withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
            $ \upgraddesc upgradptr -> do
            withArray [1] $ \alpha -> withArray [0] $ \beta -> do
              -- compute gradient with regards to the input feature maps
              inputsgrad <- emptyTensor $ MT.shape fmaps :: IO (IOTensor 4 a)
              withTensor4d inputsgrad $ \inputsgraddesc inputsgradptr ->
                handleError
                "Couldn't compute convolution gradient with respect to the inputs." $
                CuDNN.convolutionBackwardData handle alpha filtersdesc filtersptr
                upgraddesc upgradptr convdesc beta inputsgraddesc inputsgradptr
              return $ unsafeCoerce inputsgrad
  runConv `catch` \e -> error $ "Exception thrown in convolution backward pass with regards to data: " ++ show (e :: IOException)

-- activations
activationFwd :: forall m a . (PrimMonad m, TensorScalar a)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
activationFwd handle mode input = unsafePrimToPrim $ do
  withTensor4d (unsafeCoerce input :: IOTensor 4 a) $ \inputdesc inputptr -> do
    output <- emptyTensor $ MT.shape input :: IO (IOTensor 4 a)
    withTensor4d output $ \outputdesc outputptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute activations." $
          CuDNN.activationForward handle mode alpha inputdesc
          inputptr beta outputdesc outputptr
        return $ unsafeCoerce output

activationBwd :: forall m a . (PrimMonad m, TensorScalar a)
              => CuDNN.Handle
              -> CuDNN.ActivationMode
              -> MTensor (PrimState m) 4 a
              -> MTensor (PrimState m) 4 a
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
activationBwd handle mode input output upgrad = unsafePrimToPrim $ do
  withTensor4d (unsafeCoerce input :: IOTensor 4 a) $ \inputdesc inputptr -> do
    withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a) $ \upgraddesc upgradptr -> do
      grad <- emptyTensor $ MT.shape input :: IO (IOTensor 4 a)
      withTensor4d (unsafeCoerce output :: IOTensor s a) $ \outputdesc outputptr -> do
        withTensor4d grad $ \graddesc gradptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute activation backward pass." $
              CuDNN.activationBackward handle mode alpha inputdesc inputptr
              upgraddesc upgradptr outputdesc outputptr beta graddesc
              gradptr
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

pooling2dFwd :: forall m a . (PrimMonad m, TensorScalar a)
             => CuDNN.Handle
             -> (Int,Int)
             -> (Int,Int)
             -> (Int,Int)
             -> CuDNN.PoolingMode
             -> MTensor (PrimState m) 4 a
             -> m (MTensor (PrimState m) 4 a)
pooling2dFwd handle (wh,ww) (padh,padw) (strh,strw) mode input = unsafePrimToPrim $ do
  let [cwh,cww,cpadh,cpadw,cstrh,cstrw] = fmap fromIntegral [wh,ww,padh,padw,strh,strw]
      outshp = pooling2dOutputShape (wh,ww) (padh,padw) (strh,strw) $ MT.shape input
  withDescriptor
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d (unsafeCoerce input :: IOTensor 4 a)
        $ \inputdesc inputptr -> do
        output <- emptyTensor outshp :: IO (IOTensor 4 a)
        -- actual pooling
        withTensor4d output $ \outputdesc outputptr -> do
          withArray [1] $ \alpha -> withArray [0] $ \beta -> do
            handleError "Couldn't compute pooling." $
              CuDNN.poolingForward handle pooldesc alpha inputdesc
              inputptr beta outputdesc outputptr
            return $ unsafeCoerce output

pooling2dBwd :: forall m a . (PrimMonad m, TensorScalar a)
             => CuDNN.Handle
             -> (Int,Int)
             -> (Int,Int)
             -> (Int,Int)
             -> CuDNN.PoolingMode
             -> MTensor (PrimState m) 4 a
             -> MTensor (PrimState m) 4 a
             -> MTensor (PrimState m) 4 a
             -> m (MTensor (PrimState m) 4 a)
pooling2dBwd handle (wh,ww) (padh,padw) (strh,strw) mode inp out upgrad = unsafePrimToPrim $ do
  let [cwh,cww,cpadh,cpadw,cstrh,cstrw] = fmap fromIntegral [wh,ww,padh,padw,strh,strw]
  withDescriptor
    "pooling"
    CuDNN.createPoolingDescriptor
    (\d -> CuDNN.setPooling2dDescriptor d mode cwh cww cpadh cpadw cstrh cstrw)
    CuDNN.destroyPoolingDescriptor $ \pooldesc -> do
      withTensor4d (unsafeCoerce inp :: IOTensor 4 a)
        $ \inpdesc inpptr -> do
        withTensor4d (unsafeCoerce out :: IOTensor 4 a)
          $ \outdesc outptr -> do
          withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
            $ \upgraddesc upgradptr -> do
            grad <- emptyTensor $ MT.shape inp :: IO (IOTensor 4 a)
            withTensor4d grad $ \graddesc gradptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't compute backward pooling." $
                  CuDNN.poolingBackward handle pooldesc alpha outdesc outptr
                  upgraddesc upgradptr inpdesc inpptr beta graddesc gradptr
              return $ unsafeCoerce grad

-- -- Softmax
-- softmaxFwd :: forall m a . (PrimMonad m, TensorScalar a)
--            => CuDNN.Handle
--            -> CuDNN.SoftmaxAlgorithm
--            -> CuDNN.SoftmaxMode
--            -> MTensor (PrimState m) 2 a
--            -> m (MTensor (PrimState m) 2 a)
-- softmaxFwd handle algo mode input = unsafePrimToPrim $ do
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
-- softmaxBwd handle algo mode src srcdiff = unsafePrimToPrim $ do
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

nchw_to_nhwc' :: forall m a . (PrimMonad m, TensorScalar a)
              => CuDNN.Handle
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
nchw_to_nhwc' handle src = unsafePrimToPrim $ do
  res <- transformTensorIO handle CuDNN.nchw CuDNN.nhwc
         (unsafeCoerce src :: IOTensor 4 a) :: IO (IOTensor 4 a)
  return $ unsafeCoerce res

nhwc_to_nchw' :: forall m a . (PrimMonad m, TensorScalar a)
              => CuDNN.Handle
              -> MTensor (PrimState m) 4 a
              -> m (MTensor (PrimState m) 4 a)
nhwc_to_nchw' handle src = unsafePrimToPrim $ do
  res <- transformTensorIO handle CuDNN.nhwc CuDNN.nchw
         (unsafeCoerce src :: IOTensor 4 a) :: IO (IOTensor 4 a)
  return $ unsafeCoerce res

-- Unsafe but useful.
transformTensorIO :: forall a . (TensorScalar a)
                  => CuDNN.Handle
                  -> CuDNN.TensorFormat
                  -> CuDNN.TensorFormat
                  -> IOTensor 4 a
                  -> IO (IOTensor 4 a)
transformTensorIO handle srcf dstf src = do
  datatype <- MT.dtype src
  let [a',b',c',d'] = dimensions $ MT.shape src
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
          dst <- emptyTensor $ if dstf == CuDNN.nchw
                               then n:.c:.h:.w:.Z
                               else n:.h:.w:.c:.Z
          withDevicePtr src $ \srcptr -> do
            withDevicePtr dst $ \dstptr -> do
              withArray [1] $ \alpha -> withArray [0] $ \beta -> do
                handleError "Couldn't transform the tensor." $
                  CuDNN.transformTensor handle alpha srcdesc srcptr beta
                  dstdesc dstptr
                return dst

addTensor :: forall a m . (TensorScalar a, PrimMonad m)
          => CuDNN.Handle
          -> CuDNN.AddMode
          -> MTensor (PrimState m) 4 a
          -> MTensor (PrimState m) 4 a
          -> m (MTensor (PrimState m) 4 a)
addTensor handle addMode bias src = unsafePrimToPrim $ do
  out <- MT.copy (unsafeCoerce src :: IOTensor 4 a)
  withTensor4d (unsafeCoerce bias :: IOTensor 4 a) $ \biasdesc biasptr -> do
    withTensor4d (unsafeCoerce out :: IOTensor 4 a) $ \outdesc outptr -> do
      withArray [1] $ \alpha -> withArray [1] $ \beta -> do
        handleError "Couldn't add the tensors." $
          CuDNN.addTensor handle addMode alpha biasdesc biasptr
          beta outdesc outptr
  return $ unsafeCoerce out

convolutionBackwardBias :: forall a m
                        . (TensorScalar a, PrimMonad m)
                        => CuDNN.Handle
                        -> MTensor (PrimState m) 4 a
                        -> m (MTensor (PrimState m) 4 a)
convolutionBackwardBias handle upgrad = unsafePrimToPrim $ do
  let n:.c:.h:.w:.Z = MT.shape upgrad
      outshp = 1:.c:.1:.1:.Z
  biasgrad <- emptyTensor outshp :: IO (IOTensor 4 a)
  withTensor4d (unsafeCoerce upgrad :: IOTensor 4 a)
    $ \upgraddesc upgradptr -> do
    withTensor4d biasgrad $ \biasgraddesc biasgradptr -> do
      withArray [1] $ \alpha -> withArray [0] $ \beta -> do
        handleError "Couldn't compute convolution backward pass with regards to bias" $
          CuDNN.convolutionBackwardBias handle alpha upgraddesc upgradptr
          beta biasgraddesc biasgradptr
  return $ unsafeCoerce biasgrad
