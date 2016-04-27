module Test.DeepBanana.Tensor where

import Test.Hspec
import System.IO.Unsafe

import DeepBanana
import DeepBanana.Prelude

import Config

test_tensor :: Spec
test_tensor = do
  test_broadcast

test_broadcast :: Spec
test_broadcast = describe "DeepBanana.Tensor TestDevice.broadcast" $ do
  it "Works in a simple example" $ do
    let x = tensorFromList' (1:.3:.Z) [1,2,3] :: Tensor TestDevice (Dim 2) CFloat
        expected = tensorFromList' (3:.3:.Z) [1,2,3,
                                       1,2,3,
                                       1,2,3] :: Tensor TestDevice (Dim 2) CFloat
        actual = unsafeRunExcept (
          broadcast (shape expected) x :: Either CudaExceptions (Tensor TestDevice (Dim 2) CFloat)
          )
    actual `shouldBe` expected
