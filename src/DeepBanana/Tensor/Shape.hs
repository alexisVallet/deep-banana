{-# LANGUAGE UndecidableInstances #-}
{-| Shape type class and instances. Defines the shape of a tensor, both at the type level and value level.
-}
module DeepBanana.Tensor.Shape (
  -- * Shape type class
  -- $shape
    Shape(..)
  -- * Type-level arithmetic utilities
  -- $typearith
  , Quotient
  , Remainder
  , Max
  ) where

import GHC.TypeLits
import Data.Proxy

{- $shape
The 'Shape' type class can only be implemented by types of kind '[Nat]'. These fully
define the shape of a given tensor at the type level. Further, 'Shape' also requires its
instances to have size and number of dimension information available, both at the type
level and the value level. These informations are used to ensure at compile time that
most operations on tensor are correctly applied.

It also means that currently, we cannot handle tensors with (even partially) dynamic
shapes. Although this is rarely an issue when training a neural network - which mostly
expect inputs of a given size - 2d convolution layers are sometimes applied to inputs
of varying width and height at test time. This should be adressed in a future release.
-}

-- | Type class for tensor shapes.
class Shape (s :: [Nat]) where
  -- | The size, i.e. the total number of elements of a tensor of shape s, at the type
  --   level. By convention, we set the empty type-level list to size 0.
  type Size s :: Nat
  -- | The number of dimensions of a tensor of shape s. For instance:
  --
  -- @
  --   Nbdim '[] ~ 0
  --   Nbdim '[5] ~ 1
  --   Nbdim '[6,8] ~ 2
  -- @
  type Nbdim s :: Nat
  -- | Value-level shape of a tensor with type-level shape s.
  shape :: Proxy s -> [Int]
  -- | Value-level number of dimensions of a tensor with type-level shape s.
  nbdim :: Proxy s -> Int
  nbdim p = length $ shape p
  -- | Value-level size, or number of elements of a tensor with type-level shape s. As
  --   with the type-level example, we set the empty type-level list to size 0.
  size :: Proxy s -> Int
  size p = case shape p of
            [] -> 0
            xs -> product xs

instance Shape '[] where
  type Size '[] = 0
  type Nbdim '[] = 0
  shape _ = []

instance (KnownNat n) => Shape '[n] where
  type Size '[n] = n
  type Nbdim '[n] = 1
  shape _ = [fromIntegral $ natVal (Proxy :: Proxy n)]

instance forall (e1 :: Nat) (e2 :: Nat) (l :: [Nat])
         . (KnownNat e1, Shape (e2 ': l))
         => Shape (e1 ': (e2 ': l)) where
  type Size (e1 ': (e2 ': l)) = e1 * Size (e2 ': l)
  type Nbdim (e1 ': (e2 ': l)) = 1 + Nbdim (e2 ': l)
  shape _ = (fromIntegral $ natVal (Proxy :: Proxy e1)) : shape (Proxy :: Proxy (e2 ': l))

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
