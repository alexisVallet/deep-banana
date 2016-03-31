module Test.DeepBanana.Layer.LSTM (
  test_lstm
  ) where

import Test.Hspec

import DeepBanana.Prelude
import DeepBanana

import Test.DeepBanana.Layer.NumericGrad

test_lstm :: Spec
test_lstm = describe "DeepBanana.Layer.LSTM.lstm" $ do
  it "has a correct backward pass" $ do
    let
      n = 4
      m = 4
      initw s = do
        w' <- uniform s
        return $ w' * 0.16 - 0.08
      w = do
        w' <- pure hBuild
              <*> initw (m:.m:.Z)
              <*> initw (m:.Z)
              <*> initw (m:.m:.Z)
              <*> initw (m:.Z)
              <*> initw (m:.m:.Z)
              <*> initw (m:.m:.Z)
              <*> initw (m:.Z)
        return $ HLS $ hEnd w' :: CUDAT IO (HLSpace CFloat (LSTMWeights CFloat))
      input = pure (,) <*> normal (n:.m:.Z) 0 1 <*> normal (n:.m:.Z) 0 1
    runCUDATEx 42 $ (check_backward lstm w input input :: CUDAT IO ())
