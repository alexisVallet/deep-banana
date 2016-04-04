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
        w' <- do
          w_1 <- initw (m:.m:.Z)
          w_2 <- initw (m:.Z)
          w_3 <- initw (m:.m:.Z)
          w_4 <- initw (m:.Z)
          w_5 <- initw (m:.m:.Z)
          w_6 <- initw (m:.m:.Z)
          w_7 <- initw (m:.Z)
          return $ w_1:.w_2:.w_3:.w_4:.w_5:.w_6:.w_7:.Z
        return $ W $ w' :: CudaT IO (Weights CFloat (LSTMWeights CFloat))
      input = pure (,) <*> normal (n:.m:.Z) 0 1 <*> normal (n:.m:.Z) 0 1
    runCudaTEx (createGenerator rng_pseudo_default 42) $ (check_backward lstm w input input :: CudaT IO ())
    return ()
