{-# LANGUAGE StandaloneDeriving, GeneralizedNewtypeDeriving, TypeFamilies #-}
module DeepBanana.Weights (
    module DeepBanana.HList
  , Weights(..)
  ) where

import DeepBanana.HList
import DeepBanana.Prelude
import Data.Serialize (Serialize)

newtype Weights (a :: *) l = W {
  unWeights :: HList l
  }

deriving instance (Eq (HList l)) => Eq (Weights a l)
deriving instance (Ord (HList l)) => Ord (Weights a l)
deriving instance (Show (HList l)) => Show (Weights a l)
deriving instance (NFData (HList l)) => NFData (Weights a l)
deriving instance (Serialize (HList l)) => Serialize (Weights a l)

instance AdditiveGroup (Weights a '[]) where
  zeroV = W Z
  negateV _ = W Z
  _ ^+^ _ = W Z

instance forall a l e . (AdditiveGroup (Weights a l), AdditiveGroup e)
         => AdditiveGroup (Weights a (e ': l)) where
  zeroV = W $ zeroV :. unWeights (zeroV :: Weights a l)
  negateV (W (e :. l)) = W $ (negateV e) :. unWeights (negateV (W l :: Weights a l))
  W (e1 :. l1) ^+^ W (e2 :. l2) = W $ (e1 ^+^ e2)
                                      :. unWeights (W l1 ^+^ W l2 :: Weights a l)

instance VectorSpace (Weights a '[]) where
  type Scalar (Weights a '[]) = a
  x *^ _ = W Z

instance forall a l e
         . (VectorSpace e, VectorSpace (Weights a l), Scalar e ~ Scalar (Weights a l))
         => VectorSpace (Weights a (e ': l)) where
  type Scalar (Weights a (e ': l)) = Scalar e
  x *^ W (e :. l) = W $ (x *^ e) :. unWeights (x *^ (W l :: Weights a l))

-- Elementwise numeric instances for weights.
instance Num (Weights a '[]) where
  W Z + W Z = W Z
  W Z - W Z = W Z
  W Z * W Z = W Z
  abs (W Z) = W Z
  signum (W Z) = W Z
  fromInteger _ = W Z

instance forall a e (l :: [*])
         . (Num e, Num (Weights a l))
         => Num (Weights a (e ': l)) where
  W ((:.) x1 xs1) + W ((:.) x2 xs2) =
     W $ (:.) (x1 + x2) (unWeights (W xs1 + W xs2 :: Weights a l))
  W ((:.) x1 xs1) - W ((:.) x2 xs2) =
     W $ (:.) (x1 - x2) (unWeights (W xs1 - W xs2 :: Weights a l))
  W ((:.) x1 xs1) * W ((:.) x2 xs2) =
     W $ (:.) (x1 * x2) (unWeights (W xs1 * W xs2 :: Weights a l))
  abs (W ((:.) x xs)) = W ((:.) (abs x) (unWeights (abs $ W xs :: Weights a l)))
  signum (W ((:.) x xs)) = W ((:.) (signum x) (unWeights (signum $ W xs :: Weights a l)))
  fromInteger i = W ((:.) (fromInteger i) (unWeights (fromInteger i :: Weights a l)))

instance Fractional (Weights a '[]) where
  recip (W Z) = W Z
  fromRational _ = W Z

instance forall a e (l :: [*])
         . (Fractional e, Fractional (Weights a l))
         => Fractional (Weights a (e ': l)) where
  W ((:.) x1 xs1) / W ((:.) x2 xs2) =
     W $ (:.) (x1 / x2) (unWeights (W xs1 / W xs2 :: Weights a l))
  recip (W ((:.) x xs)) = W ((:.) (recip x) (unWeights (recip $ W xs :: Weights a l)))
  fromRational r = W ((:.) (fromRational r) (unWeights (fromRational r :: Weights a l)))

instance Floating (Weights a '[]) where
  pi = W Z
  exp = id
  log = id
  sin = id
  cos = id
  asin = id
  acos = id
  atan = id
  sinh = id
  cosh = id
  tanh = id
  asinh = id
  acosh = id
  atanh = id

instance forall a e (l :: [*])
         . (Floating e, Floating (Weights a l))
         => Floating (Weights a (e ': l)) where
  pi = W ((:.) pi (unWeights (pi :: Weights a l)))
  exp (W ((:.) x xs)) = W ((:.) (exp x) (unWeights (exp (W xs) :: Weights a l)))
  log (W ((:.) x xs)) = W ((:.) (log x) (unWeights (log (W xs) :: Weights a l)))
  sqrt (W ((:.) x xs)) = W ((:.) (sqrt x) (unWeights (sqrt (W xs) :: Weights a l)))
  sin (W ((:.) x xs)) = W ((:.) (sin x) (unWeights (sin (W xs) :: Weights a l)))
  cos (W ((:.) x xs)) = W ((:.) (cos x) (unWeights (cos (W xs) :: Weights a l)))
  tan (W ((:.) x xs)) = W ((:.) (tan x) (unWeights (tan (W xs) :: Weights a l)))
  asin (W ((:.) x xs)) = W ((:.) (asin x) (unWeights (asin (W xs) :: Weights a l)))
  acos (W ((:.) x xs)) = W ((:.) (acos x) (unWeights (acos (W xs) :: Weights a l)))
  atan (W ((:.) x xs)) = W ((:.) (atan x) (unWeights (atan (W xs) :: Weights a l)))
  sinh (W ((:.) x xs)) = W ((:.) (sinh x) (unWeights (sinh (W xs) :: Weights a l)))
  cosh (W ((:.) x xs)) = W ((:.) (cosh x) (unWeights (cosh (W xs) :: Weights a l)))
  tanh (W ((:.) x xs)) = W ((:.) (tanh x) (unWeights (tanh (W xs) :: Weights a l)))
  asinh (W ((:.) x xs)) = W ((:.) (asinh x) (unWeights (asinh (W xs) :: Weights a l)))
  acosh (W ((:.) x xs)) = W ((:.) (acosh x) (unWeights (acosh (W xs) :: Weights a l)))
  atanh (W ((:.) x xs)) = W ((:.) (atanh x) (unWeights (atanh (W xs) :: Weights a l)))
  W ((:.) x1 xs1) ** W ((:.) x2 xs2) =
    W $ (:.) (x1**x2) (unWeights (W xs1 ** W xs2 :: Weights a l))
  logBase (W ((:.) x1 xs1)) (W ((:.) x2 xs2)) =
    W $ (:.) (logBase x1 x2) (unWeights (logBase (W xs1) (W xs2) :: Weights a l))

