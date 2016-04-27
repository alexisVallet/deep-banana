{-# LANGUAGE DataKinds, TypeOperators, TypeFamilies, GADTs, FlexibleContexts, UndecidableInstances, MultiParamTypeClasses, FlexibleInstances, ScopedTypeVariables #-}
module DeepBanana.HList (
    HList(..)
  , SizedList
  , SizedList'
  , Concat(..)
  , HCategory(..)
  , HMonoid(..)
  ) where

import Control.Applicative
import qualified Control.Monad.RWS.Strict as S
import qualified Control.Monad.State.Strict as S
import qualified Control.Monad.Writer.Strict as S
import DeepBanana.Device
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

instance DeviceTransfer (HList '[]) (HList '[]) where
  transfer = return

instance (DeviceTransfer (HList l1) (HList l2), DeviceTransfer e1 e2)
         => DeviceTransfer (HList (e1 ': l1)) (HList (e2 ': l2)) where
  transfer (e1:.l1) = do
    e2 <- transfer e1
    l2 <- transfer l1
    return (e2:.l2)

infixr 3 >+>
class HCategory (cat :: [*] -> * -> * -> *) where
  id' :: cat '[] a a
  (>+>) :: (Concat l1 l2) => cat l1 a b -> cat l2 b c -> cat (ConcatRes l1 l2) a c

infixr 3 <+>
class HMonoid (t :: [*] -> *) where
  hmempty :: t '[]
  (<+>) :: (Concat l1 l2) => t l1 -> t l2 -> t (ConcatRes l1 l2)

instance HMonoid HList where
  hmempty = Z
  (<+>) = hconcat
