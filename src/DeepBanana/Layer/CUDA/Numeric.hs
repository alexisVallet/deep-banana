module DeepBanana.Layer.CUDA.Numeric (
    llog
  , inv
  , lexp
  , scale
  , scaleByCst
  , multiply
  , add
  ) where

import DeepBanana.Device
import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Prelude
import DeepBanana.Tensor

-- elementwise log (useful for cost fct)
llog :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
     => Layer m a '[] (Tensor d s a) (Tensor d s a)
llog = combinePasses' fwdlog bwdlog
  where fwdlog inp = return $ liftFixed1 log inp
        bwdlog inp _ = return $ \upgrad ->
          unsafeRunCudaError $ liftFixed2 (*) upgrad (liftFixed1 recip inp)

inv :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
    => Layer m a '[] (Tensor d s a) (Tensor d s a)
inv = combinePasses' fwdInv bwdInv
  where fwdInv inp = return $ liftFixed1 recip inp
        bwdInv inp invinp = return $ \upgrad ->
          unsafeRunCudaError $ liftFixed2 (\dy y -> dy * (-(y*y))) upgrad invinp

-- elementwise exponential
lexp :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
     => Layer m a '[] (Tensor d s a) (Tensor d s a)
lexp = combinePasses' fwdExp bwdExp
  where fwdExp inp = return $ liftFixed1 exp inp
        bwdExp inp einp = return $ \upgrad ->
          unsafeRunCudaError $ liftFixed2 (*) upgrad einp

scale :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
      => Layer m a '[] (a, Tensor d s a) (Tensor d s a)
scale = combinePasses' fwdscale bwdscale
  where fwdscale (x,y) = return $ liftFixed1 (x *^) y
        bwdscale (x,y) _ = return $ \upgrad -> unsafeRunCudaError $ do
          case toAnyFixed $ shape y of
           AnyFixed fshp -> do
             fy <- shapeConvert fshp y
             fdy <- shapeConvert fshp upgrad
             return (fy <.> fdy, liftFixed1 (x *^) upgrad)

scaleByCst :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
           => a -> Layer m a '[] (Tensor d s a) (Tensor d s a)
scaleByCst c = Layer $ \_ x -> do
  (y, bwd) <- forwardBackward scale (W Z) (c,x)
  return (y, \dy -> let (_,(_,dx)) = bwd dy in (W Z, dx))

multiply :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
         => Layer m a '[] (Tensor d s a, Tensor d s a) (Tensor d s a)
multiply = combinePasses' fwdMul bwdMul
  where fwdMul (x1,x2) = liftFixed2 (*) x1  x2
        bwdMul (x1,x2) _ = return $ \upgrad ->
          unsafeRunCudaError
          $ pure (,)
          <*> liftFixed2 (*) x2 upgrad
          <*> liftFixed2 (*) x1 upgrad

add :: (Device d, MonadCudaError m, Shape s, TensorScalar a)
    => Layer m a '[] (Tensor d s a, Tensor d s a) (Tensor d s a)
add = combinePasses' fwdadd bwdadd
  where fwdadd (x,y) = liftFixed2 (+) x y
        bwdadd (x,y) _ = return $ \upgrad -> (upgrad, upgrad)
