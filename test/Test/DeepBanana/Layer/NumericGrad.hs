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

import DeepBanana

numericBwd :: (TensorScalar a, Shape (Dim n), Shape (Dim k), Monad m)
           => (Tensor n a -> m (Tensor k a))
           -> Tensor n a
           -> Tensor k a
           -> m (Tensor n a)
numericBwd f inp upgrad = do
  let inplist = toList inp
      upgradlist = toList upgrad
      insize = size $ shape inp
      h = 10E-5
      finitediff i = do
        fph <- f (shift i h)
        fmh <- f (shift i (-h))
        return $ (1/(2*h)) *^ (fph - fmh)
      shift i offset = fromList (shape inp) [if j /= i then inplist!!j else inplist!!j + offset | j <- [0..insize-1]]
  listGrad <- forM [0..insize-1] $ \i -> do
    fdiff <- finitediff i
    return $ fdiff <.> upgrad
  return $ fromList (shape inp) listGrad

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
  EmptyShape :: ShapeInfo (HLSpace a '[]) a
  ListShape :: ShapeInfo t1 a -> ShapeInfo (HLSpace a l) a -> ShapeInfo (HLSpace a (t1 ': l)) a
  PairShape :: ShapeInfo t1 a -> ShapeInfo t2 a -> ShapeInfo (t1,t2) a

shapeSize :: ShapeInfo t a -> Int
shapeSize (TensorShape shp) = size shp
shapeSize ScalarShape = 1
shapeSize EmptyShape = 0
shapeSize (ListShape s1 s2) = shapeSize s1 + shapeSize s2
shapeSize (PairShape s1 s2) = shapeSize s1 + shapeSize s2

class ToTensor t where
  toTensor :: t -> (Tensor 1 (Scalar t), ShapeInfo t (Scalar t))
  fromTensor :: (Tensor 1 (Scalar t), ShapeInfo t (Scalar t)) -> t

instance (TensorScalar a, Shape (Dim n)) => ToTensor (Tensor n a) where
  toTensor t = (flatten t, TensorShape $ shape t)
  fromTensor (t, TensorShape shp) = unsafeReshape shp t

instance ToTensor CFloat where
  toTensor x = (fromList (1:.Z) [x], ScalarShape)
  fromTensor (t,_) = head $ toList t

instance ToTensor CDouble where
  toTensor x = (fromList (1:.Z) [x], ScalarShape)
  fromTensor (t,_) = head $ toList t

instance (TensorScalar a) => ToTensor (HLSpace a '[]) where
  toTensor _ = (fromList (0:.Z) [],EmptyShape)
  fromTensor _ = HLS HNil

instance forall a e l
         . (TensorScalar a, VectorSpace e, VectorSpace (HLSpace a l), Scalar e ~ a,
            Scalar (HLSpace a l) ~ a, ToTensor e, ToTensor (HLSpace a l))
         => ToTensor (HLSpace a (e ': l)) where
  toTensor (HLS (HCons e l)) =
    let (te,se) = toTensor e
        (tl,sl) = toTensor (HLS l :: HLSpace a l)
    in (tconcat te tl, ListShape se sl)
  fromTensor (t, ListShape se sl) =
    let (te,tl) = case tsplitAt (shapeSize se) t of
          Left err -> error err
          Right out -> out
    in HLS $ HCons (fromTensor (te,se) :: e) (unHLS (fromTensor (tl,sl) :: HLSpace a l))

instance forall a b
         . (ToTensor a, ToTensor b, Scalar a ~ Scalar b, TensorScalar (Scalar a))
         => ToTensor (a,b) where
  toTensor (a,b) =
    let (ta,sa) = toTensor a
        (tb,sb) = toTensor b
    in (tconcat ta tb, PairShape sa sb :: ShapeInfo (a,b) (Scalar a))
  fromTensor (t,PairShape sa sb) =
    let (ta,tb) = case tsplitAt (shapeSize sa) t of
          Left err -> error err
          Right out -> out
    in (fromTensor (ta,sa), fromTensor (tb,sb))
