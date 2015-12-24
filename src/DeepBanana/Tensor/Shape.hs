{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-| Shape type class and instances. Defines the shape of a tensor, both at the type level and value level.
-}
module DeepBanana.Tensor.Shape (
  -- * Shape type class
  -- $shape
    Shape(..)
  , Size
  , Nbdim
  -- * Broadcasting
  -- $broadcasting
  , BroadcastPred
  , Broadcast
  -- * Type-level arithmetic utilities
  -- $typearith
  , Quotient
  , Remainder
  , Max
  ) where

import GHC.TypeLits
import Data.Proxy
import Data.HList.HList
import Control.Monad
import Control.Monad.Except

{- $shape
The 'Shape' type class can only be implemented by types of kind '[Dim Nat]'. These fully
define the shape of a given tensor at the type level. Further, 'Shape' also requires its
instances to have size and number of dimension information available, both at the type
level and the value level. These informations are used to ensure at compile time that
most operations on tensor are correctly applied.
-}

-- | The size, i.e. the total number of elements of a tensor of shape s, at the type
--   level. By convention, we set the empty type-level list to size 0.
type family Size (s :: [Nat]) where
  Size '[] = 0
  Size '[n] = n
  Size (n ': l) = n * Size l

-- | The number of dimensions of a tensor of shape s. Nothing if it is unknown at
-- compile time. For instance:
--
-- @
--   Nbdim '[] ~ 0
--   Nbdim '[ 'D 5] ~ 1
--   Nbdim '[ 'D 6, 'D 8] ~ 2
--   Nbdim '[ 'Any, 'D 6] ~ 2
-- @
type family Nbdim (s :: [Nat]) where
  Nbdim s = Length s

type family Length (l :: [k]) where
  Length '[] = 0
  Length (e ': l) = 1 + Length l

-- | Type class for tensor shapes.
class Shape (s :: [Nat]) where
  -- | Value-level shape of a tensor with type-level shape s.
  dimensions :: Proxy s -> [Int]
  -- | Value-level number of dimensions of a tensor with type-level shape s.
  nbdim :: Proxy s -> Int
  nbdim rep = length $ dimensions rep
  -- | Value-level size, or number of elements of a tensor with type-level shape s. As
  --   with the type-level example, we set the empty type-level list to size 0.
  size :: Proxy s -> Int
  size rep = case dimensions rep of
              [] -> 0
              xs -> product xs

instance Shape '[] where
  dimensions _ = []

instance forall l n . (Shape l, KnownNat n) => Shape (n ': l) where
  dimensions _ =
    fromIntegral (natVal (Proxy :: Proxy n)) : dimensions (Proxy :: Proxy l)

{- $broadcasting
Rules are identical to numpy's broadcasting. Assuming 2 shapes 's1' and 's2' have the
same number of dimensions, 's1' broadcasts to 's2' if and only if each dimension
that differs between 's1' and 's2' is 1 in 's1'. If 's1' has fewer dimensions than
's2', we append 1-sized dimensions so it is.
-}

type family BroadcastPred' s1 s2 where
  BroadcastPred' '[] '[] = 'True
  BroadcastPred' (1 ': l1) (n ': l2) = BroadcastPred' l1 l2
  BroadcastPred' (n ': l1) (n ': l2) = BroadcastPred' l1 l2
  BroadcastPred' s1 s2 = 'False

type family PadOnes s n where
  PadOnes s 0 = s
  PadOnes s n = 1 ': PadOnes s (n - 1)

-- | Broadcasting as a predicate. Returns ''True' iff 's1' may be broadcasted to 's2'.
type family BroadcastPred s1 s2 where
  BroadcastPred s1 s2 = BroadcastPred' (PadOnes s1 (Nbdim s2 - Nbdim s1)) s2

-- | Broadcasting as a type class.
class Broadcast (s1 :: [Nat]) (s2 :: [Nat])
instance (BroadcastPred s1 s2 ~ 'True) => Broadcast s1 s2

{- $typearith
Some helper type families to compute mathematical operation in addition to the
operations provided by "GHC.TypeLits". Used for computing convolution and pooling output
shape at compile time.
-}

type family Quotient' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Quotient' m n 'LT = 0
  Quotient' m n 'EQ = 1
  Quotient' m n 'GT = 1 + Quotient' (m - n) n (CmpNat (m - n) n)

type family Remainder' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Remainder' m n 'LT = m
  Remainder' m n 'EQ = 0
  Remainder' m n 'GT = Remainder' (m - n) n (CmpNat (m - n) n)

-- | Type level function to compute the quotient of the division of 'm' by 'n'.
--
-- > natVal (Proxy :: Proxy (Quotient m n)) == natVal (Proxy :: Proxy m) `div` natVal (Proxy :: Proxy n)
type family Quotient (m :: Nat) (n :: Nat) where
  Quotient m n = Quotient' m n (CmpNat m n)

-- | Type level function to compute the remainder of the division of 'm' by 'n'.
--
-- > natVal (Proxy :: Proxy (Remainder m n)) == natVal (Proxy :: Proxy m) `rem` natVal (Proxy :: Proxy n)
type family Remainder (m :: Nat) (n :: Nat) where
  Remainder m n = Remainder' m n (CmpNat m n)

type family Max' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Max' m n 'LT = n
  Max' m n o = m

-- | Type level max function.
--
-- > natVal (Proxy :: Proxy (Max m n)) == max (natVal (Proxy :: Proxy m)) (natVal (Proxy :: Proxy n))
type family Max (m :: Nat) (n :: Nat) where
  Max m n = Max' m n (CmpNat m n)
