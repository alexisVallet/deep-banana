{-# LANGUAGE GADTs, TypeFamilies, DataKinds, UndecidableInstances, StandaloneDeriving, DeriveGeneric, RankNTypes #-}
module DeepBanana.Tensor.Shape (
    Shape(..)
  , Fixed(..)
  , Dim(..)
  ) where
import Data.Maybe (fromJust)
import Data.Serialize
import Unsafe.Coerce

import DeepBanana.Prelude hiding (get, put)

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

data family Fixed (l :: [Nat]) :: *
data instance Fixed '[] = SNil
data instance Fixed (n ': ns) = SCons (Proxy n) (Fixed ns)

deriving instance Show (Fixed '[])
deriving instance (Show (Fixed ns)) => Show (Fixed (n ': ns))

deriving instance Eq (Fixed '[])
deriving instance (Eq (Fixed ns)) => Eq (Fixed (n ': ns))

deriving instance Ord (Fixed '[])
deriving instance (Ord (Fixed ns)) => Ord (Fixed (n ': ns))

instance Serialize (Fixed '[]) where
  put SNil = return ()
  get = return SNil

instance forall n ns . (Serialize (Fixed ns))
         => Serialize (Fixed (n ': ns)) where
  put _ = return ()
  get = do
    s <- get
    return ((Proxy :: Proxy n) `SCons` s)

instance NFData (Fixed '[]) where
  rnf SNil = ()

instance (NFData (Fixed ns)) => NFData (Fixed (n ': ns)) where
  rnf (SCons Proxy ns) = rnf ns
                 
instance Shape (Fixed '[]) where
  dimensions SNil = []
  scalarShape = SNil

instance forall n ns
         . (Generic (Proxy n), Serialize (Proxy n), NFData (Proxy n),
            Shape (Fixed ns), KnownNat n)
         => Shape (Fixed (n ': ns)) where
  dimensions (SCons p s) = (fromIntegral $ natVal p) : dimensions s
  scalarShape = (Proxy :: Proxy n) `SCons` (scalarShape :: (Fixed ns))

infixr 3 :.
data Dim (n :: Nat) where
  Z :: Dim 0
  (:.) :: Int -> Dim (n - 1) -> Dim n

type family Length l where
  Length '[] = 0
  Length (x ': xs) = 1 + Length xs

class FixedToDim (l :: [Nat]) where
  fixedToDim :: Fixed l -> Dim (Length l)

instance FixedToDim '[] where
  fixedToDim _ = Z

instance (KnownNat n, FixedToDim ns, ((1 + Length ns) - 1) ~ Length ns)
         => FixedToDim (n ': ns) where
  fixedToDim (SCons p ps) = fromIntegral (natVal p) :. fixedToDim ps

data FixedLength (n :: Nat) where
  FixedLength :: (FixedToDim l, Length l ~ n) => Fixed l -> FixedLength n

class DimToFixedLength (n :: Nat) where
  dimToFixedLength :: Dim n -> FixedLength n
  fixedLengthToDim :: FixedLength n -> Dim n

instance {-# OVERLAPPING #-} DimToFixedLength 0 where
  dimToFixedLength _ = FixedLength SNil
  fixedLengthToDim _ = Z

instance (n ~ (1 + (n - 1)), DimToFixedLength (n - 1))
         => DimToFixedLength n where
  dimToFixedLength (i :. is) =
    case dimToFixedLength is of
     FixedLength fs -> case fromJust $ someNatVal $ fromIntegral i of
       SomeNat p -> FixedLength $ p `SCons` fs
  fixedLengthToDim (FixedLength fs) = fixedToDim fs

instance Show (Dim n) where
  show Z = "Z"
  show (i :. is) = show i ++ " :. " ++ show is

dimToList :: Dim n -> [Int]
dimToList Z = []
dimToList (i :. is) = i : dimToList is

unsafeListToDim :: [Int] -> Dim n
unsafeListToDim [] = unsafeCoerce $ Z
unsafeListToDim (i : is) = i :. unsafeListToDim is

instance Eq (Dim n) where
  Z == Z = True
  (i :. is) == (j :. js) = i == j && is == js

instance Ord (Dim n) where
  compare d1 d2 = compare (dimToList d1) (dimToList d2)

instance NFData (Dim n) where
  rnf Z = ()
  rnf (i :. is) = seq i (rnf is)

instance {-# OVERLAPPING #-} Serialize (Dim 0) where
  put Z = return ()
  get = return Z

instance Serialize (Dim (n - 1)) => Serialize (Dim n) where
  put (i :. is) = do
    put i
    put is
  get = do
    i <- get
    is <- get
    return (i :. is)

instance {-# OVERLAPPING #-} Shape (Dim 0) where
  dimensions _ = []
  scalarShape = Z

instance (Shape (Dim (n - 1))) => Shape (Dim n) where
  dimensions d = dimToList d
  scalarShape = 1 :. scalarShape
