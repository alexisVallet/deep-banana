{-# LANGUAGE GADTs, TypeFamilies, UndecidableInstances, StandaloneDeriving, DeriveGeneric #-}
module DeepBanana.Tensor.Shape (
    Shape(..)
  , Dim
  , Z(..)
  , SCons(..)
  ) where

import Control.DeepSeq
import Data.Proxy
import Data.Serialize
import GHC.TypeLits
import GHC.Generics
import Data.HList.HList
import Unsafe.Coerce

infixr 3 :.
data Z = Z
data SCons b = Int :. b

deriving instance Show Z
deriving instance (Show s) => Show (SCons s)

deriving instance Eq Z
deriving instance (Eq s) => Eq (SCons s)

deriving instance Ord Z
deriving instance (Ord s) => Ord (SCons s)

deriving instance Generic Z
deriving instance (Generic s) => Generic (SCons s)

instance Serialize Z where
  put Z = return ()
  get = return Z

instance (Serialize s) => Serialize (SCons s) where
  put (i :. s) = do
    put i
    put s
  get = do
    i <- get
    s <- get
    return (i :. s)

instance NFData Z
instance (Generic (SCons s), NFData s) => NFData (SCons s)

class (Show s, Eq s, Ord s, Generic s, Serialize s, NFData s) => Shape s where
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

instance Shape Z where
  dimensions Z = []
  scalarShape = Z

instance forall s . (Shape s) => Shape (SCons s) where
  dimensions (i :. s) = i : dimensions s
  scalarShape = 1 :. (scalarShape :: s)

type family Dim (n :: Nat) where
  Dim 0 = Z
  Dim n = SCons (Dim (n - 1))
