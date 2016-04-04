{-# LANGUAGE GADTs, TypeFamilies, DataKinds, UndecidableInstances, StandaloneDeriving, DeriveGeneric, RankNTypes #-}
module DeepBanana.Tensor.Shape (
    module DeepBanana.HList
  , Shape(..)
  , Dim
  ) where

import DeepBanana.HList
import DeepBanana.Prelude hiding (get, put)
import Data.Serialize (Serialize)

class (Show s, Eq s, Ord s, Serialize s, NFData s) => Shape s where
  dimensions :: s -> [Int]
  scalarShape :: s
  nbdim :: s -> Int
  nbdim = length . dimensions
  size :: s -> Int
  size s = case dimensions s of
    [] -> 0
    [i] -> i
    xs -> product xs
  broadcastable :: s -> s -> Bool
  broadcastable s1 s2 = broadcastable' (dimensions s1) (dimensions s2)
    where broadcastable' s1' s2' = case (s1',s2') of
            ([],[]) -> True
            (n1:ns1,n2:ns2) -> (n1 == 1 || n1 == n2) && broadcastable' ns1 ns2
            _ -> False

instance Shape (HList '[]) where
  dimensions = const []
  scalarShape = Z

instance forall l . (Shape (HList l)) => Shape (HList (Int ': l)) where
  dimensions (i :. l) = i : dimensions l
  scalarShape = 1 :. (scalarShape :: HList l)
                                                       
type Dim (n :: Nat) = SizedList n Int

