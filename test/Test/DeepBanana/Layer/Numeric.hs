module Test.DeepBanana.Layer.Numeric where

import Test.Hspec

import DeepBanana
import DeepBanana.Prelude

import Config
import Test.DeepBanana.Layer.NumericGrad

test_numeric_layers :: Spec
test_numeric_layers = do
  test_add

test_add :: Spec
test_add = describe "DeepBanana.Layer.CUDA.Numeric.add" $ do
  it "has the right backward pass" $ do
    let x = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        y = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        res = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        w = return $ W Z
    runCudaTEx (generator 42)
      $ check_backward add w (pure (,) <*> x <*> y) res
    return ()
