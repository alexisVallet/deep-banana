{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Layer.CUDA.Cublas (
    dot
  , linear
  , sumCols
  , sumRows
  , replicateAsRows
  , replicateAsCols
  , cublasHandle
  ) where

import qualified Foreign.CUDA.Cublas as Cublas
import Control.Monad.Primitive

import System.IO.Unsafe
import Unsafe.Coerce

import DeepBanana.Device
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Prelude
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception
import DeepBanana.Tensor.Mutable (MTensor, IOTensor)
import qualified DeepBanana.Tensor.Mutable as MT

-- linear layer
dot :: forall m d a . (MonadCuda m, Device d, TensorScalar a)
    => Layer m a '[] (Tensor d 2 a,Tensor d 2 a) (Tensor d 2 a)
dot = combinePasses' fwddot bwddot
  where fwddot (x,w) = do
          let n :. m :. Z = shape x
              m' :. k :. Z = shape w
          when (m /= m') $ throwVariant $ incompatibleShape $
            "Incompatible shapes " ++ show (shape x) ++ " and " ++ show (shape w) ++ " for dot product."
          embedCudaFromST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            out <- MT.emptyTensor $ n :. k :. Z
            gemmFwd (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 mx mw 0 out
            unsafeFreeze out
        bwddot (x,w) out = do
          return $ broadcast' (shape out) >>> \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            mupgrad <- unsafeThaw upgrad
            mx' <- MT.emptyTensor $ shape x
            mw' <- MT.emptyTensor $ shape w
            gemmBwdA (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 mw mupgrad mx'
            gemmBwdB (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 mx mupgrad mw'
            x' <- unsafeFreeze mx'
            w' <- unsafeFreeze mw'
            return (x', w')

linear :: (MonadCuda m, Device d, TensorScalar a)
       => Layer m a '[Tensor d 2 a] (Tensor d 2 a) (Tensor d 2 a)
linear = Layer $ \(W ((:.) w Z)) x -> do
  (y, bwd) <- forwardBackward dot (W Z) (x,w)
  return (y, broadcast' (shape y) >>> \y' -> let (_, (x',w')) = bwd y'
                                                  in (W $ (:.) w' Z, x'))

-- matrix sum reductions
sumCols :: forall m d a . (MonadCuda m, Device d, TensorScalar a)
        => Layer m a '[] (Tensor d 2 a) (Tensor d 1 a)
sumCols = combinePasses' fwdsumcols bwdsumcols
  where fwdsumcols x = embedCudaFromST $ do
            let n :. m :. Z = shape x
            ones <- MT.ones $ 1 :. n :. Z
            out <- MT.emptyTensor $ 1 :. m :. Z
            mx <- unsafeThaw x
            gemmFwd (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 ones mx 0 out
            fout <- unsafeFreeze out
            return $ reshape' (m:.Z) fout
        bwdsumcols x out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              let n :. m :. Z = shape x
              ones <- MT.ones $ 1 :. n :. Z
              out <- MT.emptyTensor $ shape x
              mupgrad <- fmap (MT.reshape' $ 1 :. m :. Z) $ unsafeThaw upgrad
              gemmBwdB (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 ones mupgrad out
              unsafeFreeze out

sumRows :: forall m d a . (MonadCuda m, Device d, TensorScalar a)
        => Layer m a '[] (Tensor d 2 a) (Tensor d 1 a)
sumRows = combinePasses' fwdsumrows bwdsumrows
  where fwdsumrows x = do
          embedCudaFromST $ do
            let n :. m :. Z = shape x
            ones <- MT.ones $ m :. 1 :. Z
            out <- MT.emptyTensor $ n :. 1 :. Z
            mx <- unsafeThaw x
            gemmFwd (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 mx ones 0 out
            fmap (reshape' $ n :. Z) $ unsafeFreeze out
        bwdsumrows x out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              let n :. m :. Z = shape x
              ones <- MT.ones $ m :. 1 :. Z
              out <- MT.emptyTensor $ shape x
              mupgrad <- fmap (MT.reshape' $ n :. 1 :. Z) $ unsafeThaw upgrad
              gemmBwdA (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 ones mupgrad out
              unsafeFreeze out

replicateAsRows :: forall m d a . (MonadCuda m, Device d, TensorScalar a)
                => Int
                -> Layer m a '[] (Tensor d 1 a) (Tensor d 2 a)
replicateAsRows n = combinePasses' fwdRepRows bwdRepRows
  where fwdRepRows x = do
          embedCudaFromST $ do
            let m :. Z = shape x
            ones <- MT.ones $ n :. 1 :. Z
            out <- MT.emptyTensor $ n :. m :. Z
            mx <- fmap (MT.reshape' $ 1 :. m :. Z) $ unsafeThaw x
            gemmFwd (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdRepRows x out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              let m :. Z = shape x
              ones <- MT.ones $ n :. 1 :. Z
              out <- MT.emptyTensor $ 1 :. m :. Z
              mupgrad <- unsafeThaw upgrad
              gemmBwdB (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 ones mupgrad out
              unsafeFreeze $ MT.reshape' (m :. Z) out

replicateAsCols :: forall m d a . (MonadCuda m, Device d, TensorScalar a)
                => Int
                -> Layer m a '[] (Tensor d 1 a) (Tensor d 2 a)
replicateAsCols n = combinePasses' fwdRepCols bwdRepCols
  where fwdRepCols x = do
          embedCudaFromST $ do
            let m :. Z = shape x
            ones <- MT.ones $ 1 :. n :. Z
            out <- MT.emptyTensor $ m :. n :. Z
            mx <- fmap (MT.reshape' $ m :. 1 :. Z) $ unsafeThaw x
            gemmFwd (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdRepCols x out = do
          return $ broadcast' (shape out) >>>
            \upgrad -> unsafeRunCudaError $ embedCudaErrorFromST $ do
              let m :. Z = shape x
              ones <- MT.ones $ 1 :. n :. Z
              out <- MT.emptyTensor $ m :. 1 :. Z
              mupgrad <- unsafeThaw upgrad
              gemmBwdA (cublasHandle (Proxy :: Proxy d)) Cublas.N Cublas.N 1 ones mupgrad out
              unsafeFreeze $ MT.reshape' (m :. Z) out

-- Utility functions leveraging CuBlas.
-- GEMM wrapper, used to implement roughly everything else.
-- Computes alpha * dot(A,B) + beta * C, where all three
-- matrices are in row-major order.
-- Serves to implement nearly all the rest, including its
-- own gradients.
gemmFwd :: forall m d a
        . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
        => Cublas.Handle
        -> Cublas.Operation
        -> Cublas.Operation
        -> a
        -> MTensor (PrimState m) d 2 a
        -> MTensor (PrimState m) d 2 a
        -> a
        -> MTensor (PrimState m) d 2 a
        -> m ()
gemmFwd handle transa transb alpha a b beta c = do
  -- Figuring out the parameters to pass GEMM so it does
  -- the right thing in row-major layout, depending on the
  -- user requested transforms.
  let ra :. ca :. Z = MT.shape a
      rb :. cb :. Z = MT.shape b
      rc :. cc :. Z = MT.shape c
  -- The rules for m, n and k were derived on paper
  -- by considering all 4 possible cases, taking into
  -- account that due to layout differences gemm will
  -- read all matrices as transpose, in addition to
  -- the user supplied transformation. This also relies
  -- on the property that the transpose operation
  -- will work on both column-major and row-major data
  -- seamlessly.
  let shapeError = throwVariant $ incompatibleShape $
        "Incompatible shapes/operations combination for matrix multiplication: "
        ++ show [ra, ca] ++ ", " ++ show [rb, cb] ++ ", " ++ show [rc, cc]
        ++ ", " ++ show transa ++ ", " ++ show transb
  (m, n, k) <- case (transa, transb) of
    (Cublas.N, Cublas.N) -> do
      when (ca /= rb) shapeError
      return (cb, ra, rb)
    (Cublas.T, Cublas.N) -> do
      when (ra /= rb) shapeError
      return (cb, ca, ra)
    (Cublas.N, Cublas.T) -> do
      when (ca /= cb) shapeError
      return (rb, ra, ca)
    (Cublas.T, Cublas.T) -> do
      when (ra /= cb) shapeError
      return (rb, ca, ra)
  embedCudaError unsafeIOToPrim $ liftIO $ do
    -- since row-major matrices will be read as transposed,
    -- the leading dimension are their number of columns.
    let
      lda = ca
      ldb = cb
      ldc = cc
    MT.withDevicePtr (unsafeCoerce a :: IOTensor d 2 a) $ \aptr -> do
      MT.withDevicePtr (unsafeCoerce b :: IOTensor d 2 a) $ \bptr -> do
        MT.withDevicePtr (unsafeCoerce c :: IOTensor d 2 a) $ \cptr -> do
          Cublas.gemm handle transb transa m n k alpha bptr ldb aptr lda beta cptr ldc

-- Composes 2 cublas operations into one.
compop :: Cublas.Operation -> Cublas.Operation -> Cublas.Operation
compop Cublas.N op = op
compop op Cublas.N = op
compop Cublas.T Cublas.T = Cublas.N
compop op1 op2 = error $ "Unsupported operations: " ++ show (op1, op2)

gemmBwdA :: (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) d 2 a
         -> MTensor (PrimState m) d 2 a
         -> MTensor (PrimState m) d 2 a
         -> m ()
gemmBwdA handle transa transb alpha b upgrad out = do
  -- Need to distinguish between the case where
  -- we have an operation on A or not.
  case transa of
    Cublas.N -> do
      gemmFwd handle Cublas.N (compop transb Cublas.T) alpha upgrad b 0 out
    Cublas.T -> do
      gemmFwd handle transb Cublas.T alpha b upgrad 0 out

gemmBwdB :: (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) d 2 a
         -> MTensor (PrimState m) d 2 a
         -> MTensor (PrimState m) d 2 a
         -> m ()
gemmBwdB handle transa transb alpha a upgrad out = do
  case transb of
    Cublas.N -> do
      gemmFwd handle (compop transa Cublas.T) Cublas.N alpha a upgrad 0 out
    Cublas.T -> do
      gemmFwd handle Cublas.T transa alpha upgrad a 0 out

gemmBwdC :: forall m d a 
         . (PrimMonad m, MonadCudaError m, Device d, TensorScalar a)
         => Cublas.Handle
         -> a
         -> MTensor (PrimState m) d 2 a
         -> MTensor (PrimState m) d 2 a
         -> m ()
gemmBwdC handle beta upgrad out = do
  when (MT.shape upgrad /= MT.shape out) $ throwVariant $ incompatibleShape $
    "Incompatible shape for GEMM backward pass argument C: "
    ++ show (MT.shape upgrad) ++ ", " ++ show (MT.shape out)
  unsafePrimToPrim $ do
    MT.withDevicePtr (unsafeCoerce upgrad :: IOTensor d 2 a) $ \upgradptr -> do
      MT.withDevicePtr (unsafeCoerce out :: IOTensor d 2 a) $ \outptr -> do
        Cublas.copy handle (size $ MT.shape upgrad) upgradptr 1 outptr 1
        Cublas.scal handle (size $ MT.shape out) beta outptr 1
