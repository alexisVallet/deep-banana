{-# LANGUAGE UndecidableInstances, GADTs #-}
module Test.DeepBanana.Layer.NumericGrad where

import Prelude hiding ((.), id)
import Foreign.C
import Control.Monad
import Control.Monad.Trans
import Data.VectorSpace
import GHC.TypeLits
import Data.Proxy
import Unsafe.Coerce
import Test.Hspec

import DeepBanana

allClose :: (TensorScalar a, Ord a, Shape (Dim n)) => Tensor n a -> Tensor n a -> Bool
allClose t1 t2 =
  all (\(x1,x2) -> abs (x1 - x2) / (abs x1 + abs x2) < 0.1)
  $ zip (tensorToList t1) (tensorToList t2)

check_backward :: (MonadIO m, TensorScalar a, Ord a, Show a,
                   Show inp, Show (Weights a w), Show out, ToTensor inp,
                   ToTensor (Weights a w), ToTensor out, Scalar out ~ a,
                   Scalar (Weights a w) ~ a, Scalar inp ~ a)
               => Layer m a w inp out
               -> m (Weights a w)
               -> m inp
               -> m out
               -> m ()
check_backward layer weights input upgrad = do
  x <- input
  y <- upgrad
  w <- weights
  num_x' <- genericNumericBwd (\(w1,x1) -> forward layer w1 x1) (w,x) y
  analytic_x' <- backward layer w x y
  let (t_num_x',_) = toTensor num_x'
      (t_analytic_x',_) = toTensor analytic_x'
  if (not $ allClose t_num_x' t_analytic_x')
    then liftIO $
    expectationFailure
    $ "Numeric and analytic gradient do not match:\nNumeric: "
    ++ show num_x' ++ "\nAnalytic: " ++ show analytic_x'
    else return ()

numericBwd :: (TensorScalar a, Shape (Dim n), Shape (Dim k), Monad m)
           => (Tensor n a -> m (Tensor k a))
           -> Tensor n a
           -> Tensor k a
           -> m (Tensor n a)
numericBwd f inp upgrad = do
  let inplist = tensorToList inp
      upgradlist = tensorToList upgrad
      insize = size $ shape inp
      h = 10E-5
      finitediff i = do
        fph <- f (shift i h)
        fmh <- f (shift i (-h))
        return $ (1/(2*h)) *^ (fph - fmh)
      shift i offset = tensorFromList' (shape inp) [if j /= i then inplist!!j else inplist!!j + offset | j <- [0..insize-1]]
  listGrad <- forM [0..insize-1] $ \i -> do
    fdiff <- finitediff i
    return $ fdiff <.> upgrad
  return $ tensorFromList' (shape inp) listGrad

genericNumericBwd :: (ToTensor inp, ToTensor out, Scalar inp ~ Scalar out,
                      Monad m, TensorScalar (Scalar inp))
                  => (inp -> m out)
                  -> inp
                  -> out
                  -> m inp
genericNumericBwd f inp upgrad = do
  let (tinp,sinp) = toTensor inp
      (tupgrad,supgrad) = toTensor upgrad
      tf tx = do
        let x = fromTensor (tx,sinp)
        y <- f x
        return $ fst $ toTensor y
  tgrad <- numericBwd tf tinp tupgrad
  return $ fromTensor (tgrad,sinp)

data ShapeInfo t a where
  TensorShape :: (Shape (Dim n)) => Dim n -> ShapeInfo (Tensor n a) a
  ScalarShape :: ShapeInfo a a
  EmptyShape :: ShapeInfo (Weights a '[]) a
  HListShape :: ShapeInfo t1 a -> ShapeInfo (Weights a l) a -> ShapeInfo (Weights a (t1 ': l)) a
  PairShape :: ShapeInfo t1 a -> ShapeInfo t2 a -> ShapeInfo (t1,t2) a
  ListShape :: [ShapeInfo t a] -> ShapeInfo [t] a

shapeSize :: ShapeInfo t a -> Int
shapeSize (TensorShape shp) = size shp
shapeSize ScalarShape = 1
shapeSize EmptyShape = 0
shapeSize (HListShape s1 s2) = shapeSize s1 + shapeSize s2
shapeSize (PairShape s1 s2) = shapeSize s1 + shapeSize s2
shapeSize (ListShape ss) = sum $ fmap shapeSize ss

class ToTensor t where
  toTensor :: t -> (Tensor 1 (Scalar t), ShapeInfo t (Scalar t))
  fromTensor :: (Tensor 1 (Scalar t), ShapeInfo t (Scalar t)) -> t

instance (TensorScalar a, Shape (Dim n)) => ToTensor (Tensor n a) where
  toTensor t = (flatten t, TensorShape $ shape t)
  fromTensor (t, TensorShape shp) = reshape' shp t

instance ToTensor CFloat where
  toTensor x = (tensorFromList' (1:.Z) [x], ScalarShape)
  fromTensor (t,_) = head $ tensorToList t

instance ToTensor CDouble where
  toTensor x = (tensorFromList' (1:.Z) [x], ScalarShape)
  fromTensor (t,_) = head $ tensorToList t

instance (TensorScalar a) => ToTensor (Weights a '[]) where
  toTensor _ = (tensorFromList' (0:.Z) [], EmptyShape)
  fromTensor _ = W Z

instance forall a e l
         . (TensorScalar a, VectorSpace e, VectorSpace (Weights a l), Scalar e ~ a,
            Scalar (Weights a l) ~ a, ToTensor e, ToTensor (Weights a l))
         => ToTensor (Weights a (e ': l)) where
  toTensor (W ((:.) e l)) =
    let (te,se) = toTensor e
        (tl,sl) = toTensor (W l :: Weights a l)
    in (tconcat' te tl, HListShape se sl)
  fromTensor (t, HListShape se sl) =
    let (te,tl) = tsplitAt' (shapeSize se) t
    in W $ (:.) (fromTensor (te,se) :: e) (unWeights (fromTensor (tl,sl) :: Weights a l))

instance forall a b
         . (ToTensor a, ToTensor b, Scalar a ~ Scalar b, TensorScalar (Scalar a))
         => ToTensor (a,b) where
  toTensor (a,b) =
    let (ta,sa) = toTensor a
        (tb,sb) = toTensor b
    in (tconcat' ta tb, PairShape sa sb :: ShapeInfo (a,b) (Scalar a))
  fromTensor (t,PairShape sa sb) =
    let (ta,tb) = tsplitAt' (shapeSize sa) t
    in (fromTensor (ta,sa), fromTensor (tb,sb))

instance (ToTensor a, TensorScalar (Scalar a), Scalar a ~ Scalar [a]) => ToTensor [a] where
  toTensor [] = (tensorFromList' (0:.Z) [], ListShape [])
  toTensor (x:xs) = let (tx, sx) = toTensor x
                        (txs, sxs) = toTensor xs in
                     case sxs of
                      ListShape sxs' -> (tconcat' tx txs, ListShape (sx:sxs'))
  fromTensor (_, ListShape []) = []
  fromTensor (t, ListShape (sx:sxs)) =
    let (tx, txs) = tsplitAt' (shapeSize sx) t
    in fromTensor (tx,sx) : fromTensor (txs, ListShape sxs)
