module DeepBanana.Layer.CUDA.Numeric (
    llog
  , inv
  , lexp
  , scale
  , multiply
  , add
  ) where

import DeepBanana.Layer
import DeepBanana.Tensor
import qualified DeepBanana.Tensor.Mutable as MT

-- elementwise log (useful for cost fct)
llog :: (Monad m, TensorScalar a, Shape (Dim n))
     => Layer m a '[] (Tensor n a) (Tensor n a)
llog = combinePasses' fwdlog bwdlog
  where fwdlog inp = return $ log inp
        bwdlog inp _ = return $ \upgrad -> upgrad * recip inp

inv :: (Monad m, TensorScalar a, Shape (Dim n))
    => Layer m a '[] (Tensor n a) (Tensor n a)
inv = combinePasses' fwdInv bwdInv
  where fwdInv inp = return $ recip inp
        bwdInv inp invinp = return $ \upgrad -> upgrad * (-(invinp * invinp))

-- elementwise exponential
lexp :: (Monad m, TensorScalar a, Shape (Dim n))
     => Layer m a '[] (Tensor n a) (Tensor n a)
lexp = combinePasses' fwdExp bwdExp
  where fwdExp inp = return $ exp inp
        bwdExp inp einp = return $ \upgrad -> upgrad * einp

scale :: (Monad m, TensorScalar a, Shape (Dim n))
      => Layer m a '[] (a, Tensor n a) (Tensor n a)
scale = combinePasses' fwdscale bwdscale
  where fwdscale (x,y) = return $ x *^ y
        bwdscale (x,y) _ = return $ \upgrad -> (upgrad <.> y, x *^ upgrad)

multiply :: (Monad m, TensorScalar a, Shape (Dim n))
         => Layer m a '[] (Tensor n a, Tensor n a) (Tensor n a)
multiply = combinePasses' fwdMul bwdMul
  where fwdMul (x1,x2) = return $ x1 * x2
        bwdMul (x1,x2) _ = return $ \upgrad -> (x2 * upgrad, x1 * upgrad)

add :: (Monad m, TensorScalar a, Shape (Dim n))
    => Layer m a '[] (Tensor n a, Tensor n a) (Tensor n a)
add = combinePasses' fwdadd bwdadd
  where fwdadd (x,y) = return $ x + y
        bwdadd (x,y) _ = return $ \upgrad -> (upgrad, upgrad)
