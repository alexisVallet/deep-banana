{-# LANGUAGE UndecidableInstances #-}
module DeepBanana.Tensor.Shape (
    Shape(..)
  , Quotient
  , Remainder
  , Max
  ) where

import GHC.TypeLits
import Data.Proxy

class Shape (s :: [Nat]) where
  type Size s :: Nat
  type Nbdim s :: Nat
  shape :: Proxy s -> [Int]
  nbdim :: Proxy s -> Int
  nbdim p = length $ shape p
  size :: Proxy s -> Int
  size p = product $ shape p

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

-- Type-level Euclidean division and remainder for naturals.
type family Quotient' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Quotient' m n 'LT = 0
  Quotient' m n 'EQ = 1
  Quotient' m n 'GT = 1 + Quotient' (m - n) n (CmpNat (m - n) n)

type family Remainder' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Remainder' m n 'LT = m
  Remainder' m n 'EQ = 0
  Remainder' m n 'GT = Remainder' (m - n) n (CmpNat (m - n) n)

type family Quotient (m :: Nat) (n :: Nat) where
  Quotient m n = Quotient' m n (CmpNat m n)

type family Remainder (m :: Nat) (n :: Nat) where
  Remainder m n = Remainder' m n (CmpNat m n)

-- Type level max for naturals.
type family Max' (m :: Nat) (n :: Nat) (ord :: Ordering) where
  Max' m n 'LT = n
  Max' m n o = m

type family Max (m :: Nat) (n :: Nat) where
  Max m n = Max' m n (CmpNat m n)
