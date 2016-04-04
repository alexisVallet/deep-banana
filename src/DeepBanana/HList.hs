{-# LANGUAGE DataKinds, TypeOperators, TypeFamilies, GADTs, FlexibleContexts, UndecidableInstances, MultiParamTypeClasses, FlexibleInstances, ScopedTypeVariables #-}
module DeepBanana.HList (
    HList(..)
  , SizedList
  , SizedList'
  , Concat(..)
  ) where

import DeepBanana.Prelude hiding (get, put)
import Data.Serialize

infixr 3 :.
data family HList (l :: [*])
data instance HList '[] = Z
data instance HList (e ': l) = (:.) e (HList l)

-- Standard typeclass instances
instance Eq (HList '[]) where
  _ == _ = True

instance (Eq e, Eq (HList l)) => Eq (HList (e ': l)) where
  (e1 :. l1) == (e2 :. l2) = (e1 == e2) && (l1 == l2)

instance Ord (HList '[]) where
  compare _ _ = EQ

instance (Ord e, Ord (HList l)) => Ord (HList (e ': l)) where
  compare (e1 :. l1) (e2 :. l2) = compare (e1,l1) (e2,l2)

instance Show (HList '[]) where
  show _ = "Z"

instance (Show e, Show (HList l)) => Show (HList (e ': l)) where
  show (e :. l) = show e ++ " :. " ++ show l

instance NFData (HList '[]) where
  rnf Z = ()

instance (NFData e, NFData (HList l)) => NFData (HList (e ': l)) where
  rnf (e :. l) = rnf (e, l)

instance Serialize (HList '[]) where
  put _ = return ()
  get = return Z

instance (Serialize e, Serialize (HList l)) => Serialize (HList (e ': l)) where
  put (e :. l) = do
    put e
    put l
  get = do
    e <- get
    l <- get
    return $ e :. l

-- fixed size homogeneous lists
type family SizedList' (n :: Nat) (a :: *) where
  SizedList' 0 a = '[]
  SizedList' n a = (a ': SizedList' (n - 1) a)

type SizedList (n :: Nat) (a :: *) = HList (SizedList' n a)

-- concatenating heterogeneous lists
class Concat (l1 :: [*]) (l2 :: [*]) where
  type ConcatRes l1 l2 :: [*]
  hconcat :: HList l1 -> HList l2 -> HList (ConcatRes l1 l2)
  hsplit :: HList (ConcatRes l1 l2) -> (HList l1, HList l2)

instance Concat '[] l2 where
  type ConcatRes '[] l2 = l2
  hconcat _ l = l
  hsplit l = (Z, l)

instance (Concat l1 l2) => Concat (e ': l1) l2 where
  type ConcatRes (e ': l1) l2 = e ': ConcatRes l1 l2
  hconcat (e :. l1) l2 = e :. hconcat l1 l2
  hsplit (e :. l1andl2) = let (l1,l2) = hsplit l1andl2 in (e :. l1, l2)
