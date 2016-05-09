{-# LANGUAGE GADTs, TypeFamilies, DataKinds, UndecidableInstances, StandaloneDeriving, DeriveGeneric, RankNTypes, BangPatterns, AllowAmbiguousTypes #-}
{-|
Shape datatypes for tensors.
-}
module DeepBanana.Tensor.Shape (
  -- * Shape datatypes
    Dyn
  , Dim
  , Fixed
  -- * Shape type classes
  , Shape(..)
  , broadcastable
  , FixedShape(..)
  , IsFixed
  -- * Converting from fixed to dynamic shape
  , AnyFixed(..)
  , toAnyFixed
  -- * Re-exports
  , module DeepBanana.HList
  ) where

import DeepBanana.HList
import DeepBanana.Prelude hiding (get, put)
import Data.Serialize (Serialize)
import qualified Data.Serialize as S

-- | @'Dyn'@ represents a fully dynamic shape datatype.
type Dyn = [Int]

-- | @'Dim' n@, where @n@ is a type level natural, is a dynamic @n@-dimensional shape.
-- It is represented by an @'HList'@ of fixed size @n@ where all elements are integers.
type Dim (n :: Nat) = SizedList n Int

-- | @'Fixed' l@, where @l@ is a type level list of type level natural numbers,
-- represents a shape which is fully fixed at the type level. It is represented by
-- an @'HList'@ where all elements are @'Proxy' n@ for some type level natural @n@.
type Fixed (l :: [Nat]) = HList (Fixed' l)

-- | Type class that all shape datatypes implement.
class (Eq s, Ord s, Show s, Serialize s, NFData s) => Shape s where
  -- | Dimensions of the shape as a list of ints.
  dimensions :: s -> [Int]
  -- | Smallest shape of the datatype, meaning it can be broadcasted against any
  -- other shape of the datatype.
  scalarShape :: s
  -- | Number of dimensions.
  nbdim :: s -> Int
  nbdim = length . dimensions
  -- | Size, i.e. the product of the dimensions.
  size :: s -> Int
  size s = case dimensions s of
    [] -> 0
    [i] -> i
    xs -> product xs

-- | Returns true if and only if the first shape argument can be broadcasted to
-- to the second shape argument. Currently, broadcasting is limited to shapes
-- holding the same number of dimensions. Then, a dimension @d1@ is broadcastable
-- to a dimension @d2@ if an only if @d1 == d2@ or @d1 == 1@.
broadcastable :: (Shape s1, Shape s2) => s1 -> s2 -> Bool
broadcastable s1 s2 = broadcastable' (dimensions s1) (dimensions s2)
  where broadcastable' s1' s2' =
          case (s1',s2') of
           ([],[]) -> True
           (n1:ns1,n2:ns2) -> (n1 == 1 || n1 == n2) && broadcastable' ns1 ns2
           _ -> False

instance Shape (HList '[]) where
  dimensions = const []
  scalarShape = Z

instance forall l . (Shape (HList l)) => Shape (HList (Int ': l)) where
  dimensions (i :. l) = i : dimensions l
  scalarShape = 1 :. (scalarShape :: HList l)

instance forall l n . (Shape (HList l), KnownNat n, NFData (Proxy n)) => Shape (HList (Proxy n ': l)) where
  dimensions (p :. l) = fromIntegral (natVal p) : dimensions l
  scalarShape = (Proxy :: Proxy n) :. (scalarShape :: HList l)

instance forall l . (Shape (HList l)) => Shape (HList (SomeNat ': l)) where
  dimensions (SomeNat p :. l) = fromIntegral (natVal p) : dimensions l
  scalarShape = SomeNat (Proxy :: Proxy 1) :. (scalarShape :: HList l)

instance Shape [Int] where
  dimensions = id
  scalarShape = [1]
                                                       
-- | Utility type class for shapes which are fully defined at the type level.
class FixedShape (l :: [Nat]) where
  type Fixed' l :: [*]
  type Nbdim l :: Nat
  type Size l :: Nat
  -- | Utility function to define shapes fixed at the type level.
  fixed :: Proxy l -> HList (Fixed' l)

instance FixedShape '[] where
  type Fixed' '[] = '[]
  type Nbdim '[] = 0
  type Size '[] = 0
  fixed _ = Z

instance forall n . (KnownNat n) => FixedShape '[n] where
  type Fixed' '[n] = '[Proxy n]
  type Nbdim '[n] = 1
  type Size '[n] = n
  fixed _  = (Proxy :: Proxy n) :. Z

instance forall l n1 n2
         . (FixedShape (n2 ': l), KnownNat n1, KnownNat n2)
         => FixedShape (n1 ': n2 ': l) where
  type Fixed' (n1 ': n2 ': l) = Proxy n1 ': Fixed' (n2 ': l)
  type Nbdim (n1 ': n2 ': l) = 1 + Nbdim (n2 ': l)
  type Size (n1 ': n2 ': l) = n1 * Size (n2 ': l)
  fixed _ = (Proxy :: Proxy n1) :. fixed (Proxy :: Proxy (n2 ': l))

-- | Utility type class for shapes which are fully fixed at the type level.
class (Shape s) => IsFixed s
instance IsFixed (HList '[])
instance (IsFixed (HList l), KnownNat n, NFData (Proxy n)) => IsFixed (HList (Proxy n ': l))
instance (IsFixed (HList l)) => IsFixed (HList (SomeNat ': l))

instance forall t . Serialize (Proxy t) where
  put _ = return ()
  get = return (Proxy :: Proxy t)

-- | Existential datatype wrapping any fixed shape. Useful when converting from a fixed
-- type level shape to a dynamic shape locally.
data AnyFixed where
  AnyFixed :: (IsFixed (HList l)) => HList l -> AnyFixed

-- | Converts an arbitrary shape to its fixed version.
toAnyFixed :: (Shape s) => s -> AnyFixed
toAnyFixed s = toAnyFixed' $ dimensions s
  where toAnyFixed' [] = AnyFixed Z
        toAnyFixed' (i:is) =
          case toAnyFixed' is of
           AnyFixed rest ->
             case someNatVal $ fromIntegral i of
              Nothing -> error $ "A shape may not have any negative dimensions: " ++ show s
              Just si -> AnyFixed $ si :. rest

-- Orphan instances for NFData and Serialize.
instance NFData SomeNat where
  rnf (SomeNat !p) = ()

instance NFData (Proxy t) where
  rnf !Proxy = ()

instance Serialize SomeNat where
  put (SomeNat p) = S.put $ natVal p
  get = do
    i <- S.get
    case someNatVal i of
     Nothing -> fail $ "Error decoding, a SomeNat value may not have a negative value."
     Just s -> return s
