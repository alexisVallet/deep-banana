module DeepBanana.Layer.Recurrent (
    InfList(..)
  , lunfold
  , infListToList
  ) where

import Control.Monad.Trans
import Control.Category
import Data.VectorSpace
import DeepBanana
import Prelude hiding ((.), id)

data InfList a = Cons a (InfList a)

lunfold :: (Monad m, AdditiveGroup b, AdditiveGroup (HLSpace s w))
        => Layer m s w b (a, b) -> Layer m s w b (InfList a)
lunfold l = Layer $ \w b1 -> do
  ((a, b2), bwda) <- forwardBackward l w b1
  ~(as, bwdas) <- forwardBackward (lunfold l) w b2
  return (Cons a as, \ ~(Cons a' as') -> let ~(w2', b2') = bwdas as'
                                             (w1', b1') = bwda (a', b2') in
                                         (w1' ^+^ w2', b1' ^+^ b2'))

infListToList :: InfList a -> [a]
infListToList ~(Cons x xs) = x : infListToList xs
