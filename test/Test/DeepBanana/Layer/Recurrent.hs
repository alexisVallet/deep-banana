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

test_lunfold :: Spec
test_lunfold = describe "DeepBanana.Layer.Recurrent.lunfold" $ do
  it "Has a correct backward pass" $ do
    let nb_samples = 16
        nb_features = 16
        nb_output = 8
        out_length = 20
        xavier s@(nb_input:.nb_output:.Z) = do
          x <- uniform s
          return $ (sqrt 6 / sqrt (fromIntegral $ nb_input + nb_output)) * (x - 0.5)
        h_to_out =
          linear
          >+> lreshape (nb_samples:.nb_output:.1:.1:.Z)
          >+> activation activation_tanh
          >+> lreshape (nb_samples:.nb_output:.Z) :: Layer CUDA CFloat '[Tensor 2 CFloat] (Tensor 2 CFloat) (Tensor 2 CFloat)
        h_to_h =
          linear
          >+> lreshape (nb_samples:.nb_features:.1:.1:.Z)
          >+> activation activation_tanh
          >+> lreshape (nb_samples:.nb_features:.Z) :: Layer CUDA CFloat '[Tensor 2 CFloat] (Tensor 2 CFloat) (Tensor 2 CFloat)
        recnet =
          lunfold' out_length (h_to_out &&& h_to_h)
        w = do
          w' <- pure hBuild
                <*> xavier (nb_features:.nb_output:.Z)
                <*> xavier (nb_features:.nb_features:.Z)
          return $ HLS $ hEnd w'
        h_0 = normal (nb_samples:.nb_features:.Z) 0 0.01
        upgrad = replicateM out_length $ normal (nb_samples:.nb_output:.Z) 0 0.01
    runCUDA 42 $ check_backward recnet w h_0 upgrad

test_recMlrCost :: Spec
test_recMlrCost = describe "DeepBanana.Layer.Recurrent.recMlrCost" $ do
  it "Has a correct backward pass" $ do
    let nb_samples = 4
        nb_output = 4
        out_length = 1
        out_seq = replicateM out_length $ uniform (nb_samples:.nb_output:.Z)
        labels = replicateM out_length $ uniform (nb_samples:.nb_output:.Z)
        cost = recMlrCost (nb_samples:.nb_output:.Z) >+> toScalar
    runCUDA 42 $ check_backward cost (return zeroV) (pure (,) <*> labels <*> out_seq) (return 1 :: CUDA CFloat)
