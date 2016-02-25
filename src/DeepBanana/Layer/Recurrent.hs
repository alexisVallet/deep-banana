{-# LANGUAGE TypeFamilies, DataKinds, ScopedTypeVariables, ImplicitParams #-}
module DeepBanana.Layer.Recurrent (
    module DeepBanana.Layer.Recurrent.Exception
  , CFunctor(..)
  , CFoldable(..)
  , CUnfoldable(..)
  , Base(..)
  , lfold
  , lunfold
  , lunfold'
  , recMlrCost
  , lzip
  , lsum
  ) where

import GHC.Stack

import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA
import DeepBanana.Prelude
import DeepBanana.Layer.Recurrent.Exception
import DeepBanana.Tensor

data family Base t :: * -> *

class (Category c) => CFunctor c f where
  cmap :: c a b -> c (f a) (f b)

class (CFunctor c (Base t)) => CFoldable c t where
  cproject :: c t (Base t t)
  cata :: c (Base t a) a -> c t a
  cata f = c where c = f . cmap c . cproject

class (CFunctor c (Base t)) => CUnfoldable c t where
  cembed :: c (Base t t) t
  ana :: c a (Base t a) -> c a t
  ana f = a where a = cembed . cmap a . f

data instance Base [a] b = Nil
                         | Cons a b

-- For the functor, foldable and unfoldable instances, we have to assume
-- throughout that the list fed to the backward pass has the same length
-- as the outputted one. Hence the many incomplete pattern matches.
--- Haven't figured out a user-friendly to ensure it
-- (using fixed size lists using some kind of type level natural works, but
--  can't be used in the very common case where that length is not fixed at compile
--  time. Though Data.Reflection and GHC.TypeLits do provide some ways to deal with
--  that, it's not enough for my use case.)
-- Also requires somewhat janky instances for AdditiveGroup and VectorSpace.
-- To be fixed eventually with a proper fixed size list interface, so at least users
-- have type safety (though library code itself doesn't).
incompatibleBackwardPass :: (?loc :: CallStack) => [a] -> [a] -> IncompatibleLength
incompatibleBackwardPass out upgrad =
  incompatibleLength $ "Incompatible length for the list backward pass input. It should have the same length as the forward pass output.\nForward pass output length: " ++ show (length out) ++ "\nBackward pass input length: " ++ show (length upgrad)

instance (Monad m, AdditiveGroup (HLSpace s w))
         => CFunctor (Layer m s w) (Base [c]) where
  cmap l = Layer $ \w bca -> case bca of
    Nil -> return (Nil, \xs -> case xs of
                                Nil -> (zeroV, Nil)
                                _ -> throw
                                     $ incompatibleLength
                                     $ "CFunctor (Base [c]): incompatible backward pass, expected Nil, got Cons.")
    Cons c a -> do
      ~(b, bwdb) <- forwardBackward l w a
      return (Cons c b, \xs -> case xs of
                                    Cons c' b' -> let (w', a') = bwdb b' in
                                                   (w', Cons c' a')
                                    _ -> throw
                                         $ incompatibleLength
                                         $ "CFunctor (Base [c]): incompatible backward pass, expected Cons, got Nil.")

instance (Monad m, AdditiveGroup (HLSpace s w)) => CFunctor (Layer m s w) [] where
  cmap l = Layer $ \w as -> case as of
    [] -> return ([], \xs -> case xs of
                              [] -> (zeroV, [])
                              _ -> throw $ incompatibleBackwardPass [] xs)
    (x:xs) -> do
      (y, bwdy) <- forwardBackward l w x
      (ys, bwdys) <- forwardBackward (cmap l) w xs
      return (y:ys, \upgrad -> case upgrad of
                                    (y':ys') -> let (w2', xs') = bwdys ys'
                                                    (w1', x') = bwdy y' in
                                                (w1' ^+^ w2', x':xs')
                                    _ -> throw $ incompatibleBackwardPass (y:ys) upgrad)

instance (Monad m, AdditiveGroup (HLSpace s w))
         => CFoldable (Layer m s w) [a] where
  cproject = Layer $ \w as -> case as of
    [] -> return (Nil, \xs -> case xs of
                               Nil -> (zeroV, [])
                               _ -> throw $ incompatibleLength
                                    $ "CFoldable (Layer m s w) [a]: incompatible backward pass, expected Nil, got Cons.")
    ~(a:as) -> return (Cons a as, \xs -> case xs of
                                          Cons a' as' -> (zeroV, a':as')
                                          _ -> throw $ incompatibleLength
                                               $ "CFoldable (Layer m s w) [a]: incompatible backward pass, expected Cons, got Nil.")

instance (Monad m, AdditiveGroup (HLSpace s w))
         => CUnfoldable (Layer m s w) [a] where
  cembed = Layer $ \w as -> case as of
    Nil -> return ([], \xs -> case xs of
                               [] -> (zeroV, Nil)
                               _ -> throw $ incompatibleBackwardPass [] xs)
    ~(Cons a as) -> return (a:as, \xs -> case xs of
                             (a':as') -> (zeroV, Cons a' as')
                             _ -> throw $ incompatibleBackwardPass (a:as) xs)

baseToMaybePair :: (Monad m, AdditiveGroup (HLSpace s w))
                => Layer m s w (Base [a] b) (Maybe (a, b))
baseToMaybePair = Layer $ \w bab -> case bab of
  Nil -> return (Nothing, \Nothing -> (zeroV, Nil))
  ~(Cons a b) -> return (Just (a, b), \(Just (a', b')) -> (zeroV, Cons a' b'))

maybePairToBase :: (Monad m, AdditiveGroup (HLSpace s w))
                => Layer m s w (Maybe (a, b)) (Base [a] b)
maybePairToBase = Layer $ \w mab -> case mab of
  Nothing -> return (Nil, \Nil -> (zeroV, Nothing))
  ~(Just (a, b)) -> return (Cons a b, \(Cons a' b') -> (zeroV, Just (a', b')))

lfold :: (Monad m, AdditiveGroup (HLSpace s w))
      => Layer m s w (Maybe (a, b)) b -> Layer m s w [a] b
lfold l = cata (baseToMaybePair >>> l)

lunfold :: (Monad m, AdditiveGroup (HLSpace s w))
        => Layer m s w b (Maybe (a, b)) -> Layer m s w b [a]
lunfold l = ana (l >>> maybePairToBase)

ltake :: (Monad m, AdditiveGroup (HLSpace s w), AdditiveGroup b)
      => Layer m s w b (a, b) -> Layer (StateT Int m) s w b (Maybe (a, b))
ltake l = Layer $ \w b1 -> do
  i <- get
  case i of
   0 -> return (Nothing, \Nothing -> (zeroV, zeroV))
   n -> do
     modify (\k -> k - 1)
     ((a, b2), bwda) <- lift $ forwardBackward l w b1
     return (Just (a, b2), \(Just (a', b2')) -> let (w', b1') = bwda (a', b2') in
                                                (w', b1'))

levalStateT :: (Monad m) => st -> Layer (StateT st m) s w a b -> Layer m s w a b
levalStateT st l = Layer $ \w a -> flip evalStateT st $ forwardBackward l w a


lunfold' :: (Monad m, AdditiveGroup (HLSpace s w), AdditiveGroup b)
         => Int -> Layer m s w b (a, b) -> Layer m s w b [a]
lunfold' n l = levalStateT n (lunfold (ltake l))

-- Unsightly AdditiveGroup/VectorSpace instances for lists, assuming
-- in binary operations that it's fed lists of the same size, with
-- zeroV outputting an empty list one could broadcast I guess.
-- Could zip the lists and have something kinda somewhat well-behaved
-- but arrr it's ugly. So nope.
-- Once again, to be eventually replaced by well-behaved fixed-size
-- hommogeneous list datatype.

instance AdditiveGroup a => AdditiveGroup [a] where
  zeroV = []
  negateV = fmap negateV
  [] ^+^ [] = []
  (x:xs) ^+^ (y:ys) = (x ^+^ y) : (xs ^+^ ys)
  xs ^+^ ys = throw $ incompatibleLength $ "AdditiveGroup [a]: can't add lists of different length: " ++ show (length xs) ++ " and " ++ show (length ys)

instance (VectorSpace a) => VectorSpace [a] where
  type Scalar [a] = Scalar a
  x *^ [] = []
  x *^ (y:ys) = (x *^ y) : (x *^ ys)

lsum :: (Monad m, AdditiveGroup a) => Layer m s '[] [a] a
lsum = lfold $ Layer $ \_ mab -> case mab of
                                  Nothing -> return (zeroV, \x' -> (zeroV, Nothing))
                                  Just (a, acc) -> do
                                    (res, bwdres) <- forwardBackward add zeroV (a,acc)
                                    return (res, \x' -> let (_,(a', acc')) = bwdres x'
                                                        in (zeroV, Just (a', acc')))

lmean :: (Monad m, InnerSpace a, VectorSpace (Scalar a), Floating (Scalar a)) => Layer m (Scalar a) '[] [a] a
lmean = Layer $ \_ as -> do
  let scaling = 1 / (fromIntegral $ length as)
  forwardBackward (lsum >+> scale -< scaling) zeroV as

lzip :: (Monad m) => Layer m s '[] ([a],[a]) [(a,a)]
lzip = combinePasses' fwdZip bwdZip
  where fwdZip (xs,ys) = return $ zip xs ys
        bwdZip _ _ = return unzip

recMlrCost :: (Monad m, MonadError t m, Variant t IncompatibleShape,
               Variant t IncompatibleSize, Variant t OutOfMemory, Exception t,
               TensorScalar a)
           => Dim 2 -> Layer m a '[] ([Tensor 2 a], [Tensor 2 a]) (Tensor 1 a)
recMlrCost s = lzip >+> cmap (mlrCost s) >+> lmean
