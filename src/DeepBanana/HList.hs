{-# LANGUAGE DataKinds, TypeOperators, TypeFamilies, GADTs, FlexibleContexts, UndecidableInstances, MultiParamTypeClasses, FlexibleInstances, ScopedTypeVariables #-}
{-|
Heterogeneous lists. Used notably for:

  - Tensor shapes

  - Neural network weights

  - Multi GPU

It is similar in principle and very much inspired by Oleg Kiselyov's <https://hackage.haskell.org/package/HList HList package>.
-}
module DeepBanana.HList (
  -- * Heterogeneous list datatypes
    HList(..)
  , SizedList
  , SizedList'
  -- * Concatenating heterogeneous lists
  , Concat(..)
  , ConcatRes
  -- * Heterogeneous categories and monoids
  , HCategory(..)
  , HMonoid(..)
  -- * Internals
  , HDeviceTransfer(..)
  ) where

import Control.Applicative
import qualified Control.Monad.RWS.Strict as S
import qualified Control.Monad.State.Strict as S
import qualified Control.Monad.Writer.Strict as S
import DeepBanana.Device
import DeepBanana.Exception
import DeepBanana.Prelude hiding (get, put)
import DeepBanana.Tensor.Exception
import Data.Serialize
import Unsafe.Coerce

infixr 3 :.
-- | An @'HList' l@ is an heterogeneous list indexed by a type-level list @l@ of its
-- elements datatypes.
--
-- @
-- 1:."Hi!":.(+):.Z :: HList '[Int,String,Float -> Float -> Float]
-- @
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

-- | Helper type family for @'SizedList'@.
type family SizedList' (n :: Nat) (a :: *) where
  SizedList' 0 a = '[]
  SizedList' n a = (a ': SizedList' (n - 1) a)

-- | Fixed size homogeneous list datatype. Useful to define shapes with dynamic
-- dimensions and fixed length, for instance.
type SizedList (n :: Nat) (a :: *) = HList (SizedList' n a)

-- | Concatenating and splitting heterogeneous lists.
class Concat (l1 :: [*]) (l2 :: [*]) where
  hconcat :: HList l1 -> HList l2 -> HList (ConcatRes l1 l2)
  hsplit :: HList (ConcatRes l1 l2) -> (HList l1, HList l2)

type family ConcatRes' (acc :: [*]) (xs :: [*]) :: [*] where
  ConcatRes' acc '[] = acc
  ConcatRes' acc (x ': xs) = ConcatRes' (x ': acc) xs

type family ConcatRes (xs :: [*]) (ys :: [*]) :: [*] where
  ConcatRes xs ys = ConcatRes' ys (Reverse xs)

type family Reverse (xs :: [*]) :: [*] where
  Reverse xs = Reverse' xs '[]

type family Reverse' (xs :: [*]) (acc :: [*]) :: [*] where
  Reverse' '[] acc = acc
  Reverse' (x ': xs) acc = Reverse' xs (x ': acc)
                    
instance Concat '[] l2 where
  hconcat _ l = l
  hsplit l = (Z, l)

instance forall e l1 l2 . (Concat l1 l2) => Concat (e ': l1) l2 where
  hconcat (e :. l1) l2 =
    unsafeCoerce $ e :. hconcat l1 l2
  hsplit inp =
    let
      (e :. l1andl2) = unsafeCoerce inp :: (HList (e ': ConcatRes' l2 (Reverse' l1 '[])))
      (l1,l2) = hsplit l1andl2 :: (HList l1, HList l2)
    in (e :. l1, l2)

class (Device d) => HDeviceTransfer (l :: [*]) d where
  type HTransferred l d :: [*]
  htransfer :: (MonadError e m, Variant e OutOfMemory)
            => d -> HList l -> m (HList (HTransferred l d))

instance (Device d) => HDeviceTransfer '[] d where
  type HTransferred '[] d = '[]
  htransfer _ l = return l

instance (DeviceTransfer e d, HDeviceTransfer l d)
         => HDeviceTransfer (e ': l) d where
  type HTransferred (e ': l) d = Transferred e d ': HTransferred l d
  htransfer dev (e:.l) = do
    e' <- transfer dev e
    l' <- htransfer dev l
    return $ e':.l'

instance (HDeviceTransfer l d) => DeviceTransfer (HList l) d where
  type Transferred (HList l) d = HList (HTransferred l d)
  transfer = htransfer

infixr 3 >+>
-- | Heterogeneous category type class. Used to overload feed-forward composition
-- across a few layer datatypes.
class HCategory (cat :: [*] -> * -> * -> *) where
  id' :: cat '[] a a
  (>+>) :: (Concat l1 l2) => cat l1 a b -> cat l2 b c -> cat (ConcatRes l1 l2) a c

infixr 3 <+>
-- | Heterogeneous monoid type class. Used to overload heterogeneous list concatenation
-- for weights datatypes notably.
class HMonoid (t :: [*] -> *) where
  hmempty :: t '[]
  (<+>) :: (Concat l1 l2) => t l1 -> t l2 -> t (ConcatRes l1 l2)

instance HMonoid HList where
  hmempty = Z
  (<+>) = hconcat
