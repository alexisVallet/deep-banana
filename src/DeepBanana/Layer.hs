{-# LANGUAGE TypeFamilies, RankNTypes, InstanceSigs #-}
{-|
Compositional layers which learn a set of weights through backpropagation.
-}
module DeepBanana.Layer (
    module DeepBanana.Weights
  , Layer(..)
  , InitLayer
  , initWeights
  , layer
  , init
  , init'
  , forward
  , backward
  , combinePasses
  , combinePasses'
  , (***)
  , (&&&)
  , first
  , second
  , terminal
  , (-<)
  , noWeights
  , effect
  ) where

import DeepBanana.Prelude hiding (get, put)
import DeepBanana.Weights

newtype Layer m a (w :: [*]) inp out = Layer {
  forwardBackward :: Weights a w -> inp -> m (out, out -> (Weights a w, inp))
  }

data InitLayer m a w inp out = InitLayer {
    layer :: Layer m a w inp out
  , initWeights :: m (Weights a w)
  }

init :: Layer m a w inp out -> m (Weights a w) -> InitLayer m a w inp out
init = InitLayer

init' :: (Applicative m) => Layer m a '[] inp out -> InitLayer m a '[] inp out
init' l = InitLayer l $ pure hmempty

instance (Monad m, AdditiveGroup out, AdditiveGroup inp, AdditiveGroup (Weights a w))
         => AdditiveGroup (Layer m a w inp out) where
  l1 ^+^ l2 = Layer $ \w inp -> do
    (out1, bwd1) <- forwardBackward l1 w inp
    (out2, bwd2) <- forwardBackward l2 w inp
    return (out1 ^+^ out2, \out' -> let (w1', inp1') = bwd1 out'
                                        (w2', inp2') = bwd2 out' in
                                     (w1' ^+^ w2', inp1' ^+^ inp2'))
  zeroV = Layer $ \w inp -> do
    return (zeroV, \_ -> (zeroV, zeroV))
  negateV l = Layer $ \w inp -> do
    (out, bwd) <- forwardBackward l w inp
    return (negateV out, \out' -> negateV $ bwd out')

instance (Monad m, VectorSpace out, VectorSpace inp, VectorSpace (Weights a w),
          Scalar out ~ a, Scalar inp ~ a, Scalar (Weights a w) ~ a)
         => VectorSpace (Layer m a w inp out) where
  type Scalar (Layer m a w inp out) = a
  x *^ l = Layer $ \w inp -> do
    (out, bwd) <- forwardBackward l w inp
    return (x *^ out, \out' -> let (w', inp') = bwd out' in
                                (x *^ w', x *^ inp'))

-- | Runs the forward pass of a layer.
forward :: (Monad m) => Layer m a w inp out -> Weights a w -> inp -> m out
forward l w inp = do
  ~(out, _) <- forwardBackward l w inp
  return out

-- | Runs the backward pass of a layer.
backward :: (Monad m)
         => Layer m a w inp out -> Weights a w -> inp -> out -> m (Weights a w, inp)
backward l w inp upgrad = do
  ~(out, bwd) <- forwardBackward l w inp
  return $ bwd upgrad

-- | Creates a layer from separate forward pass and backward pass computations.
combinePasses :: (Monad m)
              => (Weights a w -> inp -> m out)
              -> (Weights a w -> inp -> out -> m (out -> (Weights a w, inp)))
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
instance forall a (w :: [*]) m . (AdditiveGroup (Weights a w), Monad m)
         => Category (Layer m a w) where
  id = Layer (\w x -> return (x, \x' -> (zeroV, x')))
  Layer fbc . Layer fab = Layer $ \w a -> do
    ~(b, bwdab) <- fab w a
    ~(c, bwdbc) <- fbc w b
    return $ (c, \c' -> let (wgrad1,bgrad) = bwdbc c'
                            (wgrad2,agrad) = bwdab bgrad in
                        (wgrad1 ^+^ wgrad2, agrad))

infixr 4 ***
(***) :: forall m s w1 w2 a b a' b' n
      . (Monad m, Concat w1 w2)
      => Layer m s w1 a b -> Layer m s w2 a' b'
      -> Layer m s (ConcatRes w1 w2) (a,a') (b,b')
Layer f1 *** Layer f2 = Layer $ \w1w2 (a,a') -> do
  let (w1,w2) = hsplit $ unWeights w1w2 :: (HList w1, HList w2)
  ~(b, bwd1) <- f1 (W w1) a
  ~(b', bwd2) <- f2 (W w2) a'
  return ((b,b'), \(bgrad,bgrad') -> let (w1grad, agrad) = bwd1 bgrad
                                         (w2grad, agrad') = bwd2 bgrad'
                                     in (W $ hconcat (unWeights w1grad) (unWeights w2grad),
                                         (agrad, agrad')))

infixr 4 &&&
(&&&) :: forall m s w1 w2 a b b' n
      . (Monad m, Concat w1 w2, AdditiveGroup a)
      => Layer m s w1 a b -> Layer m s w2 a b'
      -> Layer m s (ConcatRes w1 w2) a (b,b')
Layer f1 &&& Layer f2 = Layer $ \w1w2 a -> do
  let (w1,w2) = hsplit $ unWeights w1w2 :: (HList w1, HList w2)
  ~(b,bwd1) <- f1 (W w1) a
  ~(b',bwd2) <- f2 (W w2) a
  return ((b,b'), \(bgrad,bgrad') -> let (w1grad, agrad1) = bwd1 bgrad
                                         (w2grad, agrad2) = bwd2 bgrad'
                                     in (W $ hconcat (unWeights w1grad) (unWeights w2grad),
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
instance (Monad m) => HCategory (Layer m s) where
  id' = id
  (>+>) :: forall w1 w2 s a b c . (Concat w1 w2)
        => Layer m s w1 a b -> Layer m s w2 b c
        -> Layer m s (ConcatRes w1 w2) a c
  Layer fab >+> Layer fbc = Layer $ \w1w2 a -> do
    let (w1,w2) = hsplit $ unWeights w1w2 :: (HList w1, HList w2)
    ~(b, bwdab) <- fab (W w1) a
    ~(c, bwdbc) <- fbc (W w2) b
    return (c, \c' -> let (w2grad, bgrad) = bwdbc c'
                          (w1grad, agrad) = bwdab bgrad in
                      (W $ hconcat (unWeights w1grad) (unWeights w2grad), agrad))

instance (Monad m) => HCategory (InitLayer m s) where
  id' = InitLayer id' (return hmempty)
  InitLayer lab initW1 >+> InitLayer lbc initW2 =
    InitLayer (lab >+> lbc) (pure (<+>) <*> initW1 <*> initW2)

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
