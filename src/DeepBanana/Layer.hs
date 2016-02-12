{-# LANGUAGE TypeFamilies #-}
{-|
Compositional layers which learn a set of weights through backpropagation.
-}
module DeepBanana.Layer (
    module Control.Category
  , Layer(..)
  , HLSpace(..)
  , HList(..)
  , forward
  , backward
  , combinePasses
  , combinePasses'
  , id'
  , (***)
  , (&&&)
  , first
  , second
  , terminal
  , (>+>)
  , (-<)
  , space
  , noWeights
  , effect
  ) where
import Prelude hiding (id, (.))
import Control.Category
import Data.VectorSpace
import Data.HList.HList
import Data.HList.HListPrelude
import Data.Proxy
import Data.Serialize
import Control.DeepSeq

-- | A layer 'Layer m a w inp out' is a differentiable computation taking an input of
-- type 'inp' alongside a list of weights 'w' to produce an output of type 'out' within
-- some monad 'm'. Inputs and weights should be vector spaces whose scalar type is 'a'.
--
-- The need for the computation to be in a monad 'm' arises from layers such as dropout
-- which depend on some random number computation. Crucially, the datatype ensures all
-- effects take place only during the forward pass, and that no additional effect take
-- place during the backward pass.
--
-- Layers form a category in 2 separate and useful ways:
-- * One where the parameter 'w' must be identical for all composed layers, meaning they
--   all share the same weights. This leads directly to recurrent neural networks, and
--   happens through the standard instance from "Control.Category".
-- * One where the weight list parameters get concatenated. This leads to feed-forward
--   neural networks, and happens through the '>+>' operator.
newtype Layer m a (w :: [*]) inp out = Layer {
  forwardBackward :: HLSpace a w -> inp -> m (out, out -> (HLSpace a w, inp))
  }

-- | Runs the forward pass of a layer.
forward :: (Monad m) => Layer m a w inp out -> HLSpace a w -> inp -> m out
forward l w inp = do
  ~(out, _) <- forwardBackward l w inp
  return out

-- | Runs the backward pass of a layer.
backward :: (Monad m)
         => Layer m a w inp out -> HLSpace a w -> inp -> out -> m (HLSpace a w, inp)
backward l w inp upgrad = do
  ~(out, bwd) <- forwardBackward l w inp
  return $ bwd upgrad

-- | Creates a layer from separate forward pass and backward pass computations.
combinePasses :: (Monad m)
              => (HLSpace a w -> inp -> m out)
              -> (HLSpace a w -> inp -> out -> m (out -> (HLSpace a w, inp)))
              -> Layer m a w inp out
combinePasses fwd bwd = Layer $ \w inp -> do
  out <- fwd w inp
  bwd' <- bwd w inp out
  return (out, bwd')

combinePasses' :: (Monad m)
               => (inp -> m out)
               -> (inp -> out -> m (out -> inp))
               -> Layer m a '[] inp out
combinePasses' fwd bwd = noWeights $ \inp -> do
  out <- fwd inp
  bwd' <- bwd inp out
  return (out, bwd')

-- Category instance fits recurrent composition. (shared weights)
instance forall a (w :: [*]) m . (AdditiveGroup (HLSpace a w), Monad m)
         => Category (Layer m a w) where
  id = Layer (\w x -> return (x, \x' -> (zeroV, x')))
  Layer fbc . Layer fab = Layer $ \w a -> do
    ~(b, bwdab) <- fab w a
    ~(c, bwdbc) <- fbc w b
    return $ (c, \c' -> let (wgrad1,bgrad) = bwdbc c'
                            (wgrad2,agrad) = bwdab bgrad in
                        (wgrad1 ^+^ wgrad2, agrad))

id' :: (Monad m) => Layer m a '[] inp inp
id' = id

infixr 4 ***
(***) :: forall m s w1 w2 a b a' b' n
      . (Monad m, HAppendList w1 w2, HSplitAt n (HAppendListR w1 w2) w1 w2)
      => Layer m s w1 a b -> Layer m s w2 a' b'
      -> Layer m s (HAppendListR w1 w2) (a,a') (b,b')
Layer f1 *** Layer f2 = Layer $ \w1w2 (a,a') -> do
  let (w1,w2) = hSplitAt (Proxy :: Proxy n) $ unHLS w1w2
  ~(b, bwd1) <- f1 (HLS w1) a
  ~(b', bwd2) <- f2 (HLS w2) a'
  return ((b,b'), \(bgrad,bgrad') -> let (w1grad, agrad) = bwd1 bgrad
                                         (w2grad, agrad') = bwd2 bgrad'
                                     in (HLS $ hAppendList (unHLS w1grad) (unHLS w2grad),
                                         (agrad, agrad')))

infixr 4 &&&
(&&&) :: forall m s w1 w2 a b b' n
      . (Monad m, HAppendList w1 w2, HSplitAt n (HAppendListR w1 w2) w1 w2,
         AdditiveGroup a)
      => Layer m s w1 a b -> Layer m s w2 a b'
      -> Layer m s (HAppendListR w1 w2) a (b,b')
Layer f1 &&& Layer f2 = Layer $ \w1w2 a -> do
  let (w1,w2) = hSplitAt (Proxy :: Proxy n) $ unHLS w1w2
  ~(b,bwd1) <- f1 (HLS w1) a
  ~(b',bwd2) <- f2 (HLS w2) a
  return ((b,b'), \(bgrad,bgrad') -> let (w1grad, agrad1) = bwd1 bgrad
                                         (w2grad, agrad2) = bwd2 bgrad'
                                     in (HLS $ hAppendList (unHLS w1grad) (unHLS w2grad),
                                         agrad1 ^+^ agrad2))

first :: (AdditiveGroup b, Monad m) => Layer m s '[] (a, b) a
first = Layer $ \_ (a, b) -> do
  return (a, \a' -> (zeroV, (a', zeroV)))

second :: (AdditiveGroup a, Monad m) => Layer m s '[] (a, b) b
second = Layer $ \_ (a, b) -> do
  return (b, \b' -> (zeroV, (zeroV, b')))

terminal :: (AdditiveGroup inp, Monad m)
         => out -> Layer m a '[] inp out
terminal x = noWeights $ \_ -> return (x, \_ -> zeroV)

-- Feed-forward composition.
infixr 3 >+>
(>+>) :: forall s w1 w2 a b c m n
      . (Monad m, HAppendList w1 w2, HSplitAt n (HAppendListR w1 w2) w1 w2)
      => Layer m s w1 a b -> Layer m s w2 b c -> Layer m s (HAppendListR w1 w2) a c
Layer fab >+> Layer fbc = Layer $ \w1w2 a -> do
  let (w1,w2) = hSplitAt (Proxy :: Proxy n) $ unHLS w1w2
  ~(b, bwdab) <- fab (HLS w1) a
  ~(c, bwdbc) <- fbc (HLS w2) b
  return (c, \c' -> let (w2grad, bgrad) = bwdbc c'
                        (w1grad, agrad) = bwdab bgrad in
                    (HLS $ hAppendList (unHLS w1grad) (unHLS w2grad), agrad))

-- Making a layer out of a differentiable function that does not depend on a set
-- of weights.
noWeights :: (Monad m)
          => (inp -> m (out, out -> inp)) -> Layer m a '[] inp out
noWeights f = Layer $ \_ x -> do
  ~(y, bwd) <- f x
  return (y, \y' -> (zeroV, bwd y'))

-- Layer that runs an effectful computation on the input and passes it along
-- untouched. Useful for debugging.
effect :: (Monad m)
       => (a -> m b) -> Layer m s '[] a a
effect f = noWeights $ \x -> do
  f x
  return (x, \x' -> x')

infixr 4 -<
(-<) :: forall m i1 i2 w a out
     . (Monad m, VectorSpace i1, VectorSpace i2)
     => Layer m a w (i1,i2) out -> i1 -> Layer m a w i2 out
nn -< inp = terminal inp &&& id' >+> nn

-- We store weights in heterogeneous lists internally, which get concatenated
-- by composition.
newtype HLSpace (a :: *) l = HLS {
  unHLS :: HList l
  }

instance (Show (HList l)) => Show (HLSpace a l) where
  show = show . unHLS

space :: Proxy a -> HList l -> HLSpace a l
space _ = HLS

-- Additive group instance for list of weights.
instance AdditiveGroup (HLSpace a '[]) where
  HLS HNil ^+^ HLS HNil = HLS HNil
  zeroV = HLS HNil
  negateV (HLS HNil) = HLS HNil

instance forall a e (l :: [*])
         . (AdditiveGroup e, AdditiveGroup (HLSpace a l))
         => AdditiveGroup (HLSpace a (e ': l)) where
  HLS (HCons x1 xs1) ^+^ HLS (HCons x2 xs2) =
    HLS $ HCons (x1 ^+^ x2) (unHLS (HLS xs1 ^+^ HLS xs2 :: HLSpace a l))
  zeroV = HLS $ HCons zeroV (unHLS (zeroV :: HLSpace a l))
  negateV (HLS (HCons x xs)) = HLS $ negateV x `HCons` unHLS (negateV (HLS xs :: HLSpace a l))
                                      
-- Vector space instance for list of weights.
instance VectorSpace (HLSpace a '[]) where
  type Scalar (HLSpace a '[]) = a
  x *^ HLS HNil = HLS HNil

instance forall a e (l :: [*])
         . (VectorSpace e, VectorSpace (HLSpace a l), a ~ Scalar e,
            a ~ Scalar (HLSpace a l))
         => VectorSpace (HLSpace a (e ': l)) where
  type Scalar (HLSpace a (e ': l)) = a
  s *^ HLS (HCons x xs) = HLS $ HCons (s *^ x) (unHLS (s *^ HLS xs :: HLSpace a l))

-- Serializable heterogeneous lists of weights.
instance Serialize (HLSpace a '[]) where
  put (HLS HNil) = return ()
  get = return (HLS HNil)
                    
instance forall a e (l :: [*])
         . (Serialize e, Serialize (HLSpace a l))
         => Serialize (HLSpace a (e ': l)) where
  put (HLS (HCons x xs)) = do
    put x
    put (HLS xs :: HLSpace a l)
  get = do
    x <- (get :: Get e)
    HLS xs <- (get :: Get (HLSpace a l))
    return (HLS $ x `HCons` xs)

-- Deepseqable heterogeneous lists of weights.
instance NFData (HLSpace a '[]) where
  rnf (HLS HNil) = ()

instance forall e a l
         . (NFData e, NFData (HLSpace a l))
         => NFData (HLSpace a (e ': l)) where
  rnf (HLS (HCons x xs)) = deepseq (x, HLS xs :: HLSpace a l) ()

-- Elementwise numeric instances for weights.
instance Num (HLSpace a '[]) where
  HLS HNil + HLS HNil = HLS HNil
  HLS HNil - HLS HNil = HLS HNil
  HLS HNil * HLS HNil = HLS HNil
  abs (HLS HNil) = HLS HNil
  signum (HLS HNil) = HLS HNil
  fromInteger _ = HLS HNil

instance forall a e (l :: [*])
         . (Num e, Num (HLSpace a l))
         => Num (HLSpace a (e ': l)) where
  HLS (HCons x1 xs1) + HLS (HCons x2 xs2) =
     HLS $ HCons (x1 + x2) (unHLS (HLS xs1 + HLS xs2 :: HLSpace a l))
  HLS (HCons x1 xs1) - HLS (HCons x2 xs2) =
     HLS $ HCons (x1 - x2) (unHLS (HLS xs1 - HLS xs2 :: HLSpace a l))
  HLS (HCons x1 xs1) * HLS (HCons x2 xs2) =
     HLS $ HCons (x1 * x2) (unHLS (HLS xs1 * HLS xs2 :: HLSpace a l))
  abs (HLS (HCons x xs)) = HLS (HCons (abs x) (unHLS (abs $ HLS xs :: HLSpace a l)))
  signum (HLS (HCons x xs)) = HLS (HCons (signum x) (unHLS (signum $ HLS xs :: HLSpace a l)))
  fromInteger i = HLS (HCons (fromInteger i) (unHLS (fromInteger i :: HLSpace a l)))

instance Fractional (HLSpace a '[]) where
  recip (HLS HNil) = HLS HNil
  fromRational _ = HLS HNil

instance forall a e (l :: [*])
         . (Fractional e, Fractional (HLSpace a l))
         => Fractional (HLSpace a (e ': l)) where
  HLS (HCons x1 xs1) / HLS (HCons x2 xs2) =
     HLS $ HCons (x1 / x2) (unHLS (HLS xs1 / HLS xs2 :: HLSpace a l))
  recip (HLS (HCons x xs)) = HLS (HCons (recip x) (unHLS (recip $ HLS xs :: HLSpace a l)))
  fromRational r = HLS (HCons (fromRational r) (unHLS (fromRational r :: HLSpace a l)))

instance Floating (HLSpace a '[]) where
  pi = HLS HNil
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
         . (Floating e, Floating (HLSpace a l))
         => Floating (HLSpace a (e ': l)) where
  pi = HLS (HCons pi (unHLS (pi :: HLSpace a l)))
  exp (HLS (HCons x xs)) = HLS (HCons (exp x) (unHLS (exp (HLS xs) :: HLSpace a l)))
  log (HLS (HCons x xs)) = HLS (HCons (log x) (unHLS (log (HLS xs) :: HLSpace a l)))
  sqrt (HLS (HCons x xs)) = HLS (HCons (sqrt x) (unHLS (sqrt (HLS xs) :: HLSpace a l)))
  sin (HLS (HCons x xs)) = HLS (HCons (sin x) (unHLS (sin (HLS xs) :: HLSpace a l)))
  cos (HLS (HCons x xs)) = HLS (HCons (cos x) (unHLS (cos (HLS xs) :: HLSpace a l)))
  tan (HLS (HCons x xs)) = HLS (HCons (tan x) (unHLS (tan (HLS xs) :: HLSpace a l)))
  asin (HLS (HCons x xs)) = HLS (HCons (asin x) (unHLS (asin (HLS xs) :: HLSpace a l)))
  acos (HLS (HCons x xs)) = HLS (HCons (acos x) (unHLS (acos (HLS xs) :: HLSpace a l)))
  atan (HLS (HCons x xs)) = HLS (HCons (atan x) (unHLS (atan (HLS xs) :: HLSpace a l)))
  sinh (HLS (HCons x xs)) = HLS (HCons (sinh x) (unHLS (sinh (HLS xs) :: HLSpace a l)))
  cosh (HLS (HCons x xs)) = HLS (HCons (cosh x) (unHLS (cosh (HLS xs) :: HLSpace a l)))
  tanh (HLS (HCons x xs)) = HLS (HCons (tanh x) (unHLS (tanh (HLS xs) :: HLSpace a l)))
  asinh (HLS (HCons x xs)) = HLS (HCons (asinh x) (unHLS (asinh (HLS xs) :: HLSpace a l)))
  acosh (HLS (HCons x xs)) = HLS (HCons (acosh x) (unHLS (acosh (HLS xs) :: HLSpace a l)))
  atanh (HLS (HCons x xs)) = HLS (HCons (atanh x) (unHLS (atanh (HLS xs) :: HLSpace a l)))
  HLS (HCons x1 xs1) ** HLS (HCons x2 xs2) =
    HLS $ HCons (x1**x2) (unHLS (HLS xs1 ** HLS xs2 :: HLSpace a l))
  logBase (HLS (HCons x1 xs1)) (HLS (HCons x2 xs2)) =
    HLS $ HCons (logBase x1 x2) (unHLS (logBase (HLS xs1) (HLS xs2) :: HLSpace a l))
