{-# LANGUAGE UndecidableInstances, GADTs, MultiParamTypeClasses #-}
module Test.DeepBanana.Layer.NumericGrad where

import Foreign.C
import Prelude ((!!), head)
import Unsafe.Coerce
import Test.Hspec

import Config
import DeepBanana
import DeepBanana.Prelude hiding (head)

allClose :: (TensorScalar a, Device d, Ord a, Shape s)
         => Tensor d s a -> Tensor d s a -> Bool
allClose t1 t2 =
  all (\(x1,x2) -> abs (x1 - x2) / (abs x1 + abs x2) < 0.1)
  $ zip (tensorToList t1) (tensorToList t2)

check_backward :: forall m d a w inp out
               . (MonadIO m, TensorScalar a, Ord a, Show a,
                  Show inp, Show (Weights a w), Show out, ToTensor inp TestDevice,
                  ToTensor (Weights a w) TestDevice, ToTensor out TestDevice,
                  FixScalar out ~ a, FixScalar (Weights a w) ~ a, FixScalar inp ~ a)
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
  let p = Proxy :: Proxy TestDevice
      t_num_x' = fst $ toTensor p num_x'
      t_analytic_x' = fst $ toTensor p analytic_x'
  if (not $ allClose t_num_x' t_analytic_x')
    then liftIO $
    expectationFailure
    $ "Numeric and analytic gradient do not match:\nNumeric: "
    ++ show num_x' ++ "\nAnalytic: " ++ show analytic_x'
    else return ()

numericBwd :: forall a d s1 s2 m
           . (TensorScalar a, Device d, Shape s1, Shape s2,
              Monad m)
           => (Tensor d s1 a -> m (Tensor d s2 a))
           -> Tensor d s1 a
           -> Tensor d s2 a
           -> m (Tensor d s1 a)
numericBwd f inp upgrad = do
  let inplist = tensorToList inp
      upgradlist = tensorToList upgrad
      insize = size $ shape inp
      h = 10E-5
      finitediff i = do
        fph <- f (shift i h)
        fmh <- f (shift i (-h))
        return $ unsafeRunExcept
          (liftVec (\fph' fmh' -> (1/(2*h)) *^ (fph' - fmh')) fph fmh
           :: Either IncompatibleShape (Tensor d s2 a))
      shift i offset = tensorFromList' (shape inp) [if j /= i then inplist!!j else inplist!!j + offset | j <- [0..insize-1]]
  listGrad <- forM [0..insize-1] $ \i -> do
    fdiff <- finitediff i
    return $ unsafeRunExcept $ do
      case toAnyFixed (shape fdiff) of
       AnyFixed fshp -> do
         ffdiff <- shapeConvert fshp fdiff
         fupgrad <- shapeConvert fshp upgrad
         return $ ffdiff <.> fupgrad :: Either IncompatibleShape a
  return $ tensorFromList' (shape inp) listGrad

genericNumericBwd :: forall inp out d m
                  . (ToTensor inp TestDevice, ToTensor out TestDevice,
                     FixScalar inp ~ FixScalar out, Monad m, TensorScalar (FixScalar inp))
                  => (inp -> m out)
                  -> inp
                  -> out
                  -> m inp
genericNumericBwd f inp upgrad = do
  let p = Proxy :: Proxy TestDevice
      (tinp,sinp) = toTensor p inp
      (tupgrad,supgrad) = toTensor p upgrad
      tf tx = do
        let x = fromTensor (tx,sinp)
        y <- f x
        return $ fst $ toTensor p y
  tgrad <- numericBwd tf tinp tupgrad
  return $ fromTensor (tgrad,sinp)

data ShapeInfo t a where
  TensorShape :: (Shape s) => s -> ShapeInfo (Tensor d s a) a
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

class (Device d) => ToTensor t d where
  toTensor :: Proxy d -> t -> (Tensor d (Dim 1) (FixScalar t), ShapeInfo t (FixScalar t))
  fromTensor :: (Tensor d (Dim 1) (FixScalar t), ShapeInfo t (FixScalar t)) -> t

instance (TensorScalar a, Shape s, Device d) => ToTensor (Tensor d s a) d where
  toTensor _ t = (flatten t, TensorShape $ shape t)
  fromTensor (t, TensorShape shp) = reshape' shp $ t

instance (Device d) => ToTensor CFloat d where
  toTensor _ x = (tensorFromList' (1:.Z) [x], ScalarShape)
  fromTensor (t,_) = head $ tensorToList t

instance (Device d) => ToTensor CDouble d where
  toTensor _ x = (tensorFromList' (1:.Z) [x], ScalarShape)
  fromTensor (t,_) = head $ tensorToList t

instance (Device d, TensorScalar a) => ToTensor (Weights a '[]) d where
  toTensor _ _ = (tensorFromList' (0:.Z) [], EmptyShape)
  fromTensor _ = W Z

instance forall a d e l
         . (TensorScalar a, FixShape e, FixShape (Weights a l), FixScalar e ~ a,
            FixScalar (Weights a l) ~ a, ToTensor e d, ToTensor (Weights a l) d)
         => ToTensor (Weights a (e ': l)) d where
  toTensor p (W ((:.) e l)) =
    let (te,se) = toTensor p e
        (tl,sl) = toTensor p (W l :: Weights a l) 
    in (tconcat' te tl, HListShape se sl)
  fromTensor (t, HListShape se sl) =
    let (te,tl) = tsplitAt' (shapeSize se) t
    in W $ (:.) (fromTensor (te,se) :: e) (unWeights (fromTensor (tl,sl) :: Weights a l))

instance forall a b d
         . (ToTensor a d, ToTensor b d, FixScalar a ~ FixScalar b, TensorScalar (FixScalar a))
         => ToTensor (a,b) d where
  toTensor p (a,b) =
    let (ta,sa) = toTensor p a
        (tb,sb) = toTensor p b
    in (tconcat' ta tb, PairShape sa sb :: ShapeInfo (a,b) (FixScalar a))
  fromTensor (t,PairShape sa sb) =
    let (ta,tb) = tsplitAt' (shapeSize sa) t
    in (fromTensor (ta,sa), fromTensor (tb,sb))

instance (ToTensor a d, TensorScalar (FixScalar a), FixScalar a ~ FixScalar [a])
         => ToTensor [a] d where
  toTensor _ [] = (tensorFromList' (0:.Z) [], ListShape [])
  toTensor p (x:xs) = let (tx, sx) = toTensor p x
                          (txs, sxs) = toTensor p xs in
                     case sxs of
                      ListShape sxs' -> (tconcat' tx txs, ListShape (sx:sxs'))
  fromTensor (_, ListShape []) = []
  fromTensor (t, ListShape (sx:sxs)) =
    let (tx, txs) = tsplitAt' (shapeSize sx) t
    in fromTensor (tx,sx) : fromTensor (txs, ListShape sxs)
