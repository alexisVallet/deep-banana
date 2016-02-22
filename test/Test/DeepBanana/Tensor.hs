module Test.DeepBanana.Tensor where

import Test.Hspec
import DeepBanana.Prelude
import System.IO.Unsafe

import DeepBanana

test_tensor :: Spec
test_tensor = do
  test_broadcast

test_broadcast :: Spec
test_broadcast = describe "DeepBanana.Tensor.broadcast" $ do
  it "Works in a simple example" $ do
    let x = tensorFromList' (1:.3:.Z) [1,2,3] :: Tensor 2 CFloat
        expected = tensorFromList' (3:.3:.Z) [1,2,3,
                                       1,2,3,
                                       1,2,3] :: Tensor 2 CFloat
        actual = unsafeRunExcept (
          broadcast (shape expected) x :: Either CUDAExceptions (Tensor 2 CFloat)
          )
    actual `shouldBe` expected
