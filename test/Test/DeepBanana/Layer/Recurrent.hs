module Test.DeepBanana.Layer.Recurrent (
    test_recurrent_layers
  ) where

import Prelude hiding (id, (.))
import Control.Category
import Control.Monad
import Test.Hspec
import Data.HList.HList
import Data.VectorSpace

import DeepBanana
import DeepBanana.Layer.CUDA
import DeepBanana.Layer.Recurrent

import Test.DeepBanana.Layer.NumericGrad

test_recurrent_layers :: Spec
test_recurrent_layers = do
  test_lunfold
  test_recMlrCost
  test_lunfold_and_recMlrCost


xavier :: (Monad m, TensorScalar a) => Dim 2 -> CUDAT m (Tensor 2 a)
xavier s@(nb_input:.nb_output:.Z) = do
  x <- uniform s
  return $ (sqrt 6 / sqrt (fromIntegral $ nb_input + nb_output)) * (x - 0.5)

test_lunfold :: Spec
test_lunfold = describe "DeepBanana.Layer.Recurrent.lunfold" $ do
  it "Has a correct backward pass" $ do
    let nb_samples = 16
        nb_features = 16
        nb_output = 8
        out_length = 20
        h_to_out =
          linear
          >+> activation activation_tanh
        h_to_h =
          linear
          >+> activation activation_tanh
        recnet =
          lunfold' out_length (h_to_out &&& h_to_h)
        w = do
          w' <- pure hBuild
                <*> xavier (nb_features:.nb_output:.Z)
                <*> xavier (nb_features:.nb_features:.Z)
          return $ HLS $ hEnd w'
        h_0 = normal (nb_samples:.nb_features:.Z) 0 0.01 :: CUDAT IO (Tensor 2 CFloat)
        upgrad = replicateM out_length $ normal (nb_samples:.nb_output:.Z) 0 0.01
    runCUDATEx 42 $ check_backward recnet w h_0 upgrad

test_recMlrCost :: Spec
test_recMlrCost = describe "DeepBanana.Layer.Recurrent.recMlrCost" $ do
  it "Has a correct backward pass" $ do
    let nb_samples = 4
        nb_output = 4
        out_length = 3
        out_seq = replicateM out_length $ uniform (nb_samples:.nb_output:.Z)
        labels = replicateM out_length $ uniform (nb_samples:.nb_output:.Z)
        cost = recMlrCost (nb_samples:.nb_output:.Z) >+> toScalar
    runCUDATEx 42 $ check_backward cost (return zeroV) (pure (,) <*> labels <*> out_seq) (return 1 :: CUDAT IO CFloat)

test_lunfold_and_recMlrCost :: Spec
test_lunfold_and_recMlrCost = describe "DeepBanana.Layer.Recurrent: lunfold >+> recMlrCost" $ do
  it "Has a correct backward pass" $ do
    let nb_samples = 4
        nb_features = 8
        nb_output = 4
        out_length = 3
        h_to_out =
          linear
          >+> activation activation_tanh
        h_to_h =
          linear
          >+> activation activation_tanh
        recnet =
          lunfold' out_length (h_to_out &&& h_to_h)
        cost = recMlrCost (nb_samples:.nb_output:.Z) >+> toScalar
        fullNet = (id' *** recnet) >+> cost
        w = do
          w' <- pure hBuild
                <*> xavier (nb_features:.nb_output:.Z)
                <*> xavier (nb_features:.nb_features:.Z)
          return $ HLS $ hEnd w'
        input = do
          x <- uniform (nb_samples:.nb_features:.Z)
          labels <- replicateM out_length $ uniform (nb_samples:.nb_output:.Z)
          return (labels,x)
    runCUDATEx 42 $ check_backward fullNet w input (return 1 :: CUDAT IO CFloat)
