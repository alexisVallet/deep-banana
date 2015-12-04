module DeepBanana.Layer.CUDA.Cublas (
    linear
  , sumCols
  , sumRows
  , replicateAsRows
  , replicateAsCols
  ) where

import qualified Foreign.CUDA.Cublas as Cublas
import Control.Monad.ST
import Control.Monad.Primitive
import Data.Proxy
import GHC.TypeLits
import Prelude hiding (id, (.))
import Unsafe.Coerce

import DeepBanana.Tensor
import DeepBanana.Tensor.Mutable (MTensor, IOTensor)
import qualified DeepBanana.Tensor.Mutable as MT
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad

-- linear layer
linear :: (TensorScalar a, KnownNat m, KnownNat n, KnownNat k)
       => Layer CUDA a '[Tensor [m,k] a] (Tensor [n,m] a) (Tensor [n,k] a)
linear = combinePasses fwdlinear bwdlinear
  where fwdlinear (HLS (HCons w HNil)) x = do
          handle <- asks cublasHandle
          return $ runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            out <- MT.emptyTensor
            gemmFwd handle Cublas.N Cublas.N 1 mx mw 0 out
            unsafeFreeze out
        bwdlinear (HLS (HCons w HNil)) x _ = do
          handle <- asks cublasHandle
          return $ \upgrad -> runST $ do
            mw <- unsafeThaw w
            mx <- unsafeThaw x
            mupgrad <- unsafeThaw upgrad
            mx' <- MT.emptyTensor
            mw' <- MT.emptyTensor
            gemmBwdA handle Cublas.N Cublas.N 1 mw mupgrad mx'
            gemmBwdB handle Cublas.N Cublas.N 1 mx mupgrad mw'
            x' <- unsafeFreeze mx'
            w' <- unsafeFreeze mw'
            return (HLS $ w' `HCons` HNil, x')

-- matrix sum reductions
sumCols :: forall a n m . (TensorScalar a, KnownNat n, KnownNat m)
        => Layer CUDA a '[] (Tensor [n,m] a) (Tensor [1,m] a)
sumCols = combinePasses' fwdsumcols bwdsumcols
  where fwdsumcols x = do
          handle <- asks cublasHandle
          return $ runST $ do
            ones <- MT.ones :: ST s (MTensor s [1,n] a)
            out <- MT.emptyTensor
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdsumcols x _ = do
          handle <- asks cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.ones :: ST s (MTensor s [1,n] a)
            out <- MT.emptyTensor
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

sumRows :: forall a n m . (TensorScalar a, KnownNat n, KnownNat m)
        => Layer CUDA a '[] (Tensor [n,m] a) (Tensor [n,1] a)
sumRows = combinePasses' fwdsumrows bwdsumrows
  where fwdsumrows x = do
          handle <- asks cublasHandle
          return $ runST $ do
            ones <- MT.ones :: ST s (MTensor s [m,1] a)
            out <- MT.emptyTensor
            mx <- unsafeThaw x
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdsumrows x _ = do
          handle <- asks cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.ones :: ST s (MTensor s [m,1] a)
            out <- MT.emptyTensor
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze out

replicateAsRows :: forall a n m
                . (TensorScalar a, KnownNat n, KnownNat m)
                => Proxy n
                -> Layer CUDA a '[] (Tensor '[m] a) (Tensor [n,m] a)
replicateAsRows _ = combinePasses' fwdRepRows bwdRepRows
  where fwdRepRows x = do
          handle <- asks cublasHandle
          return $ runST $ do
            ones <- MT.shaped (Proxy :: Proxy [n,1]) MT.ones
            out <- MT.emptyTensor
            mx <- MT.shaped (Proxy :: Proxy [1,m]) $ unsafeThaw x >>= return . MT.reshape
            gemmFwd handle Cublas.N Cublas.N 1 ones mx 0 out
            unsafeFreeze out
        bwdRepRows x _ = do
          handle <- asks cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.shaped (Proxy :: Proxy [n,1]) $ MT.ones
            out <- MT.shaped (Proxy :: Proxy [1,m]) $ MT.emptyTensor
            mupgrad <- unsafeThaw upgrad
            gemmBwdB handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.reshape out

replicateAsCols :: forall a n m
                . (TensorScalar a, KnownNat n, KnownNat m)
                => Proxy n
                -> Layer CUDA a '[] (Tensor '[m] a) (Tensor [m,n] a)
replicateAsCols _ = combinePasses' fwdRepCols bwdRepCols
  where fwdRepCols x = do
          handle <- asks cublasHandle
          return $ runST $ do
            ones <- MT.ones :: ST s (MTensor s [1,n] a)
            out <- MT.emptyTensor
            mx <- unsafeThaw x >>= return . MT.reshape :: ST s (MTensor s [m,1] a)
            gemmFwd handle Cublas.N Cublas.N 1 mx ones 0 out
            unsafeFreeze out
        bwdRepCols x _ = do
          handle <- asks cublasHandle
          return $ \upgrad -> runST $ do
            ones <- MT.ones :: ST s (MTensor s [1,n] a)
            out <- MT.emptyTensor :: ST s (MTensor s [m,1] a)
            mupgrad <- unsafeThaw upgrad
            gemmBwdA handle Cublas.N Cublas.N 1 ones mupgrad out
            unsafeFreeze $ MT.reshape out

-- Utility functions leveraging CuBlas.
-- GEMM wrapper, used to implement roughly everything else.
-- Computes alpha * dot(A,B) + beta * C, where all three
-- matrices are in row-major order.
-- Serves to implement nearly all the rest, including its
-- own gradients.
gemmFwd :: forall ra ca rb cb rc cc m a
        . (PrimMonad m, TensorScalar a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
        => Cublas.Handle
        -> Cublas.Operation
        -> Cublas.Operation
        -> a
        -> MTensor (PrimState m) [ra,ca] a
        -> MTensor (PrimState m) [rb,cb] a
        -> a
        -> MTensor (PrimState m) [rc,cc] a
        -> m ()
gemmFwd handle transa transb alpha a b beta c = unsafePrimToPrim $ do
  -- Figuring out the parameters to pass GEMM so it does
  -- the right thing in row-major layout, depending on the
  -- user requested transforms.
  let [ra, ca] = shape (Proxy :: Proxy [ra,ca])
      [rb, cb] = shape (Proxy :: Proxy [rb,cb])
      [rc, cc] = shape (Proxy :: Proxy [rc,cc])
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
  -- since row-major matrices will be read as transposed,
  -- the leading dimension are their number of columns.
  let
    lda = ca
    ldb = cb
    ldc = cc
  MT.withDevicePtr (unsafeCoerce a :: IOTensor [ra,ca] a) $ \aptr -> do
    MT.withDevicePtr (unsafeCoerce b :: IOTensor [rb,cb] a) $ \bptr -> do
      MT.withDevicePtr (unsafeCoerce c :: IOTensor [rc,cc] a) $ \cptr -> do
        Cublas.gemm handle transb transa m n k alpha bptr ldb aptr lda beta cptr ldc

-- Composes 2 cublas operations into one.
compop :: Cublas.Operation -> Cublas.Operation -> Cublas.Operation
compop Cublas.N op = op
compop op Cublas.N = op
compop Cublas.T Cublas.T = Cublas.N
compop op1 op2 = error $ "Unsupported operations: " ++ show (op1, op2)

gemmBwdA :: forall ra ca rb cb rc cc m a
         . (PrimMonad m, TensorScalar a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) [rb,cb] a
         -> MTensor (PrimState m) [rc,cc] a
         -> MTensor (PrimState m) [ra,ca] a
         -> m ()
gemmBwdA handle transa transb alpha b upgrad out = do
  -- Need to distinguish between the case where
  -- we have an operation on A or not.
  case transa of
    Cublas.N -> gemmFwd handle Cublas.N (compop transb Cublas.T)
                alpha upgrad b 0 out
    Cublas.T -> gemmFwd handle transb Cublas.T alpha b upgrad 0 out

gemmBwdB :: forall ra ca rb cb rc cc m a
         . (PrimMonad m, TensorScalar a, Shape [ra,ca], Shape [rb,cb], Shape [rc,cc])
         => Cublas.Handle
         -> Cublas.Operation
         -> Cublas.Operation
         -> a
         -> MTensor (PrimState m) [ra,ca] a
         -> MTensor (PrimState m) [rc,cc] a
         -> MTensor (PrimState m) [rb,cb] a
         -> m ()
gemmBwdB handle transa transb alpha a upgrad out = do
  case transb of
    Cublas.N -> gemmFwd handle (compop transa Cublas.T) Cublas.N alpha
                a upgrad 0 out
    Cublas.T -> gemmFwd handle Cublas.T transa alpha upgrad a 0 out

gemmBwdC :: forall rc cc m a
         . (PrimMonad m, TensorScalar a, Shape [rc,cc])
         => Cublas.Handle
         -> a
         -> MTensor (PrimState m) [rc,cc] a
         -> MTensor (PrimState m) [rc,cc] a
         -> m ()
gemmBwdC handle beta upgrad out = unsafePrimToPrim $ do
  MT.withDevicePtr (unsafeCoerce upgrad :: IOTensor [rc,cc] a) $ \upgradptr -> do
    MT.withDevicePtr (unsafeCoerce out :: IOTensor [rc,cc] a) $ \outptr -> do
      Cublas.copy handle (size (Proxy :: Proxy [rc,cc])) upgradptr 1 outptr 1
      Cublas.scal handle (size (Proxy :: Proxy [rc,cc])) beta outptr 1
