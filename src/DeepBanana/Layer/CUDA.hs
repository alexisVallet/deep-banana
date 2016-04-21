{-# LANGUAGE TypeFamilies, RankNTypes #-}
module DeepBanana.Layer.CUDA (
    module DeepBanana.Layer.CUDA.CuDNN
  , module DeepBanana.Layer.CUDA.CuDNN.Exception
  , module DeepBanana.Layer.CUDA.CuRAND
  , module DeepBanana.Layer.CUDA.Exception
  , module DeepBanana.Layer.CUDA.Monad
  , module DeepBanana.Layer.CUDA.Numeric
  , dot
  , linear
  , sumRows
  , replicateAsCols
  , lreshape
  , toScalar
  , mlrCost
  ) where

import DeepBanana.Device
import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.CuDNN
import DeepBanana.Layer.CUDA.CuDNN.Exception
import DeepBanana.Layer.CUDA.CuRAND
import DeepBanana.Layer.CUDA.Exception
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Layer.CUDA.Numeric
import DeepBanana.Tensor
import DeepBanana.Tensor.Exception

-- Performs XY^T via 1x1 convolution.
dot :: (MonadCuda m, Device d, TensorScalar a)
    => Layer m a '[] (Tensor d 2 a, Tensor d 2 a) (Tensor d 2 a)
dot = Layer $ \_ (x,y) -> do
  let rx:.cx:.Z = shape x
      cy:.cx':.Z = shape y
  x' <- reshape (rx:.cx:.1:.1:.Z) x
  y' <- reshape (cy:.cx':.1:.1:.Z) y
  (xy, bwdxy) <- forwardBackward
                 (convolution2d (0,0) (1,1) convolution_fwd_algo_implicit_gemm convolution_bwd_data_algo_0 convolution_bwd_filter_algo_0)
                 (W $ y':.Z)
                 x'
  xy' <- reshape (rx:.cy:.Z) xy
  return (xy', \xygrad -> let xygrad' = reshape' (rx:.cy:.1:.1:.Z) xygrad
                              (W (ygrad:.Z), xgrad) = bwdxy xygrad'
                          in (W Z, (reshape' (rx:.cx:.Z) xgrad,
                                    reshape' (cy:.cx':.Z) ygrad)))

linear :: (MonadCuda m, Device d, TensorScalar a)
       => Layer m a '[Tensor d 2 a] (Tensor d 2 a) (Tensor d 2 a)
linear = Layer $ \(W (y:.Z)) x -> do
  (xy, bwdxy) <- forwardBackward dot (W Z) (x,y)
  return (xy, \xygrad -> let (_, (xgrad,ygrad)) = bwdxy xygrad
                         in (W $ ygrad:.Z, xgrad))

sumRows :: (MonadCuda m, Device d, TensorScalar a)
        => Layer m a '[] (Tensor d 2 a) (Tensor d 1 a)
sumRows = Layer $ \_ x -> do
  let rows:.cols:.Z = shape x
  oneColVec <- ones $ 1:.cols:.Z
  (y, bwdy) <- forwardBackward dot (W Z) (x,oneColVec)
  y' <- reshape (rows:.Z) y
  return (y', \ygrad -> let ygrad' = reshape' (rows:.1:.Z) ygrad
                            (_, (xgrad,_)) = bwdy ygrad'
                        in (W Z, xgrad))

replicateAsCols :: (MonadCuda m, Device d, TensorScalar a)
                => Int -> Layer m a '[] (Tensor d 1 a) (Tensor d 2 a)
replicateAsCols cols = Layer $ \_ x -> do
  let rows:.Z = shape x
  x' <- reshape (rows:.1:.Z) x
  oneColVec <- ones $ cols:.1:.Z
  (y, bwdy) <- forwardBackward dot (W Z) (x', oneColVec)
  return (y, \ygrad -> let (_,(xgrad,_)) = bwdy ygrad
                       in (W Z, reshape' (rows:.Z) xgrad))

lreshape :: (MonadCuda m, TensorScalar a, Shape (Dim n), Shape (Dim k))
         => Dim k -> Layer m a '[] (Tensor d n a) (Tensor d k a)
lreshape newshp = combinePasses' fwdmul bwdmul
  where fwdmul x = reshape newshp x
        bwdmul x _ = return $ \upgrad -> reshape' (shape x) upgrad

toScalar :: (MonadCuda m, TensorScalar a, Device d, Shape (Dim n))
         => Layer m a '[] (Tensor d n a) a
toScalar = noWeights $ fwdBwdToScalar
  where fwdBwdToScalar x = do
          when (size (shape x) /= 1) $ throwVariant $ incompatibleSize $ 
            "Can only convert to scalar tensor with size 1.\nshape: " ++ show (shape x)
          return (unsafeHead $ tensorToList x,
                  \upgrad -> tensorFromList' (shape x) [upgrad])

mlrCost :: (MonadCuda m, Device d, TensorScalar a)
        => Dim 2 -> Layer m a '[] (Tensor d 2 a, Tensor d 2 a) (Tensor d 1 a)
mlrCost s@(n:.m:.Z) =
  id' *** (add -< 10E-5 >+> softmax softmax_accurate softmax_mode_instance)
  >+> multiply
  >+> sumRows
  >+> add -< 10E-5
  >+> llog
  >+> lreshape (1:.n:.Z)
  >+> sumRows
  >+> scale -< (-1 / fromIntegral n)
