module DeepBanana.Layer.CUDA.Numeric (
    llog
  , inv
  , lexp
  , scale
  , multiply
  , add
  ) where

import DeepBanana.Layer
import DeepBanana.Layer.CUDA.Monad
import DeepBanana.Tensor
import qualified DeepBanana.Tensor.Mutable as MT

-- elementwise log (useful for cost fct)
llog :: (TensorScalar a, Shape s)
     => Layer CUDA a '[] (Tensor s a) (Tensor s a)
llog = combinePasses' fwdlog bwdlog
  where fwdlog inp = return $ log inp
        bwdlog inp _ = return $ \upgrad -> upgrad * recip inp

inv :: (TensorScalar a, Shape s)
    => Layer CUDA a '[] (Tensor s a) (Tensor s a)
inv = combinePasses' fwdInv bwdInv
  where fwdInv inp = return $ recip inp
        bwdInv inp invinp = return $ \upgrad -> upgrad * (-(invinp * invinp))

-- elementwise exponential
lexp :: (TensorScalar a, Shape s)
     => Layer CUDA a '[] (Tensor s a) (Tensor s a)
lexp = combinePasses' fwdExp bwdExp
  where fwdExp inp = return $ exp inp
        bwdExp inp einp = return $ \upgrad -> upgrad * einp

scale :: (TensorScalar a, Shape s)
      => Layer CUDA a '[] (a, Tensor s a) (Tensor s a)
scale = combinePasses' fwdscale bwdscale
  where fwdscale (x,y) = return $ x *^ y
        bwdscale (x,y) _ = return $ \upgrad -> (upgrad <.> y, x *^ upgrad)

multiply :: (TensorScalar a, Shape s)
         => Layer CUDA a '[] (Tensor s a, Tensor s a) (Tensor s a)
multiply = combinePasses' fwdMul bwdMul
  where fwdMul (x1,x2) = return $ x1 * x2
        bwdMul (x1,x2) _ = return $ \upgrad -> (x2 * upgrad, x1 * upgrad)

add :: (TensorScalar a, Shape s)
    => Layer CUDA a '[] (Tensor s a, Tensor s a) (Tensor s a)
add = combinePasses' fwdadd bwdadd
  where fwdadd (x,y) = return $ x + y
        bwdadd (x,y) _ = return $ \upgrad -> (upgrad, upgrad)