module DeepBanana.Layer.CUDA.Numeric (
    llog
  , inv
  , lexp
  , scale
  , multiply
  , add
  ) where

import DeepBanana.Layer
import DeepBanana.Prelude

-- elementwise log (useful for cost fct)
llog :: (Monad m, Floating t)
     => Layer m (Scalar t) '[] t t
llog = combinePasses' fwdlog bwdlog
  where fwdlog inp = return $ log inp
        bwdlog inp _ = return $ \upgrad -> upgrad * recip inp

inv :: (Monad m, Fractional t)
    => Layer m (Scalar t) '[] t t
inv = combinePasses' fwdInv bwdInv
  where fwdInv inp = return $ recip inp
        bwdInv inp invinp = return $ \upgrad -> upgrad * (-(invinp * invinp))

-- elementwise exponential
lexp :: (Monad m, Floating t)
     => Layer m (Scalar t) '[] t t
lexp = combinePasses' fwdExp bwdExp
  where fwdExp inp = return $ exp inp
        bwdExp inp einp = return $ \upgrad -> upgrad * einp

scale :: (Monad m, InnerSpace t)
      => Layer m (Scalar t) '[] (Scalar t, t) t
scale = combinePasses' fwdscale bwdscale
  where fwdscale (x,y) = return $ x *^ y
        bwdscale (x,y) _ = return $ \upgrad -> (upgrad <.> y, x *^ upgrad)

multiply :: (Monad m, Num t)
         => Layer m (Scalar t) '[] (t, t) t
multiply = combinePasses' fwdMul bwdMul
  where fwdMul (x1,x2) = return $ x1 * x2
        bwdMul (x1,x2) _ = return $ \upgrad -> (x2 * upgrad, x1 * upgrad)

add :: (Monad m, AdditiveGroup t)
    => Layer m (Scalar t) '[] (t, t) t
add = combinePasses' fwdadd bwdadd
  where fwdadd (x,y) = return $ x ^+^ y
        bwdadd (x,y) _ = return $ \upgrad -> (upgrad, upgrad)
