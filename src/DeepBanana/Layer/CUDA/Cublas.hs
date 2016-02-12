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
import Control.Monad.Except
import Control.Monad.Primitive
import Control.Monad.ST
import Control.Monad.Trans
import Data.Proxy
import Data.IORef
import GHC.TypeLits
import Prelude hiding (id, (.))
import System.IO.Unsafe
import Unsafe.Coerce

import DeepBanana.Tensor
import DeepBanana.Tensor.Mutable (MTensor, IOTensor)
import qualified DeepBanana.Tensor.Mutable as MT
import DeepBanana.Layer

-- linear layer
dot :: (Monad m, TensorScalar a)
    => Layer m a '[] (Tensor 2 a,Tensor 2 a) (Tensor 2 a)
dot = combinePasses' fwddot bwddot
  where fwddot (x,w) = do
          let n :. m :. Z = shape x
              m' :. k :. Z = shape w
          when (m /= m') $ error $
            "Incompatible shapes " ++ show (shape x) ++ " and " ++ show (shape w) ++ " for dot product."
          return $ runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            out <- MT.emptyTensor $ n :. k :. Z
            gemmFwd cublasHandle Cublas.N Cublas.N 1 mx mw 0 out
            unsafeFreeze out
        bwddot (x,w) _ = do
          return $ \upgrad -> runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            mupgrad <- unsafeThaw upgrad
            mx' <- MT.emptyTensor $ shape x
            mw' <- MT.emptyTensor $ shape w
            gemmBwdA cublasHandle Cublas.N Cublas.N 1 mw mupgrad mx'
            gemmBwdB cublasHandle Cublas.N Cublas.N 1 mx mupgrad mw'
            x' <- unsafeFreeze mx'
            w' <- unsafeFreeze mw'
            return (x', w')

linear :: (Monad m, TensorScalar a)
       => Layer m a '[Tensor 2 a] (Tensor 2 a) (Tensor 2 a)
linear = Layer $ \(HLS (HCons w HNil)) x -> do
  (y, bwd) <- forwardBackward dot (HLS HNil) (x,w)
  return (y, \y' -> let (_, (x',w')) = bwd y'
                    in (HLS $ HCons w' HNil, x'))

-- matrix sum reductions
sumCols :: (Monad m, TensorScalar a)
        => Layer m a '[] (Tensor 2 a) (Tensor 1 a)
sumCols = combinePasses' fwdsumcols bwdsumcols
  where fwdsumcols x = do
          return $ runST $ do
            let n :. m :. Z = shape x
            ones <- MT.ones $ 1 :. n :. Z
            out <- MT.emptyTensor $ 1 :. m :. Z
            mx <- unsafeThaw x
            gemmFwd cublasHandle Cublas.N Cublas.N 1 ones mx 0 out
            fout <- unsafeFreeze out
            return $ unsafeReshape (m:.Z) fout
        bwdsumcols x _ = do
          return $ \upgrad -> runST $ do
            let n :. m :. Z = shape x
            ones <- MT.ones $ 1 :. n :. Z
            out <- MT.emptyTensor $ shape x
            mupgrad <- fmap (MT.unsafeReshape $ 1 :. m :. Z) $ unsafeThaw upgrad
            gemmBwdB cublasHandle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

sumRows :: (Monad m, TensorScalar a)
        => Layer m a '[] (Tensor 2 a) (Tensor 1 a)
sumRows = combinePasses' fwdsumrows bwdsumrows
  where fwdsumrows x = do
          return $ runST $ do
            let n :. m :. Z = shape x
            ones <- MT.ones $ m :. 1 :. Z
            out <- MT.emptyTensor $ n :. 1 :. Z
            mx <- unsafeThaw x
            gemmFwd cublasHandle Cublas.N Cublas.N 1 mx ones 0 out
            fmap (unsafeReshape $ n :. Z) $ unsafeFreeze out
        bwdsumrows x _ = do
          return $ \upgrad -> runST $ do
            let n :. m :. Z = shape x
            ones <- MT.ones $ m :. 1 :. Z
            out <- MT.emptyTensor $ shape x
            mupgrad <- fmap (MT.unsafeReshape $ n :. 1 :. Z) $ unsafeThaw upgrad
            gemmBwdA cublasHandle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

replicateAsRows :: (Monad m, TensorScalar a)
                => Int
                -> Layer m a '[] (Tensor 1 a) (Tensor 2 a)
replicateAsRows n = combinePasses' fwdRepRows bwdRepRows
  where fwdRepRows x = do
          return $ runST $ do
            let m :. Z = shape x
            ones <- MT.ones $ n :. 1 :. Z
            out <- MT.emptyTensor $ n :. m :. Z
            mx <- fmap (MT.unsafeReshape $ 1 :. m :. Z) $ unsafeThaw x
            gemmFwd cublasHandle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdRepRows x _ = do
          return $ \upgrad -> runST $ do
            let m :. Z = shape x
            ones <- MT.ones $ n :. 1 :. Z
            out <- MT.emptyTensor $ 1 :. m :. Z
            mupgrad <- unsafeThaw upgrad
            gemmBwdB cublasHandle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.unsafeReshape (m :. Z) out

replicateAsCols :: (Monad m, TensorScalar a)
                => Int
                -> Layer m a '[] (Tensor 1 a) (Tensor 2 a)
replicateAsCols n = combinePasses' fwdRepCols bwdRepCols
  where fwdRepCols x = do
          return $ runST $ do
            let m :. Z = shape x
            ones <- MT.ones $ 1 :. n :. Z
            out <- MT.emptyTensor $ m :. n :. Z
            mx <- fmap (MT.unsafeReshape $ m :. 1 :. Z) $ unsafeThaw x
            gemmFwd cublasHandle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdRepCols x _ = do
          return $ \upgrad -> runST $ do
            let m :. Z = shape x
            ones <- MT.ones $ 1 :. n :. Z
            out <- MT.emptyTensor $ m :. 1 :. Z
            mupgrad <- unsafeThaw upgrad
            gemmBwdA cublasHandle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.unsafeReshape (m :. Z) out

-- Utility functions leveraging CuBlas.
-- GEMM wrapper, used to implement roughly everything else.
-- Computes alpha * dot(A,B) + beta * C, where all three
-- matrices are in row-major order.
-- Serves to implement nearly all the rest, including its
-- own gradients.
gemmFwd :: forall m a . (PrimMonad m, TensorScalar a)
        => Cublas.Handle
        -> Cublas.Operation
        -> Cublas.Operation
        -> a
        -> MTensor (PrimState m) 2 a
        -> MTensor (PrimState m) 2 a
        -> a
        -> MTensor (PrimState m) 2 a
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
  let shapeError = error $
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
  unsafePrimToPrim $ do
    -- since row-major matrices will be read as transposed,
    -- the leading dimension are their number of columns.
    let
      lda = ca
      ldb = cb
      ldc = cc
    MT.withDevicePtr (unsafeCoerce a :: IOTensor 2 a) $ \aptr -> do
      MT.withDevicePtr (unsafeCoerce b :: IOTensor 2 a) $ \bptr -> do
        MT.withDevicePtr (unsafeCoerce c :: IOTensor 2 a) $ \cptr -> do
          Cublas.gemm handle transb transa m n k alpha bptr ldb aptr lda beta cptr ldc

-- Composes 2 cublas operations into one.
compop :: Cublas.Operation -> Cublas.Operation -> Cublas.Operation
compop Cublas.N op = op
compop op Cublas.N = op
compop Cublas.T Cublas.T = Cublas.N
compop op1 op2 = error $ "Unsupported operations: " ++ show (op1, op2)

gemmBwdA :: (PrimMonad m, TensorScalar a)
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) 2 a
         -> MTensor (PrimState m) 2 a
         -> MTensor (PrimState m) 2 a
         -> m ()
gemmBwdA handle transa transb alpha b upgrad out = do
  -- Need to distinguish between the case where
  -- we have an operation on A or not.
  case transa of
    Cublas.N -> do
      gemmFwd handle Cublas.N (compop transb Cublas.T) alpha upgrad b 0 out
    Cublas.T -> do
      gemmFwd handle transb Cublas.T alpha b upgrad 0 out

gemmBwdB :: (PrimMonad m, TensorScalar a)
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) 2 a
         -> MTensor (PrimState m) 2 a
         -> MTensor (PrimState m) 2 a
         -> m ()
gemmBwdB handle transa transb alpha a upgrad out = do
  case transb of
    Cublas.N -> do
      gemmFwd handle (compop transa Cublas.T) Cublas.N alpha a upgrad 0 out
    Cublas.T -> do
      gemmFwd handle Cublas.T transa alpha upgrad a 0 out

gemmBwdC :: forall m a . (PrimMonad m, TensorScalar a)
         => Cublas.Handle
         -> a
         -> MTensor (PrimState m) 2 a
         -> MTensor (PrimState m) 2 a
         -> m ()
gemmBwdC handle beta upgrad out = unsafePrimToPrim $ do
  when (MT.shape upgrad /= MT.shape out) $ error $
    "Incompatible shape for GEMM backward pass argument C: "
    ++ show (MT.shape upgrad) ++ ", " ++ show (MT.shape out)
  MT.withDevicePtr (unsafeCoerce upgrad :: IOTensor 2 a) $ \upgradptr -> do
    MT.withDevicePtr (unsafeCoerce out :: IOTensor 2 a) $ \outptr -> do
      Cublas.copy handle (size $ MT.shape upgrad) upgradptr 1 outptr 1
      Cublas.scal handle (size $ MT.shape out) beta outptr 1

global_cublas_handle :: IORef (Maybe Cublas.Handle)
{-# NOINLINE global_cublas_handle #-}
global_cublas_handle = unsafePerformIO $ newIORef Nothing

cublasHandle :: Cublas.Handle
cublasHandle = unsafePerformIO $ do
  mh <- readIORef global_cublas_handle
  case mh of
   Nothing -> do
     h <- Cublas.create
     writeIORef global_cublas_handle $ Just h
     return h
   Just h -> return h
