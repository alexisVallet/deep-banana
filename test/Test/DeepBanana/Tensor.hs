module Test.DeepBanana.Tensor where

import Prelude hiding (id, (.))
import Test.Hspec
import Data.List
import Control.Monad
import Control.Monad.ST
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import System.IO.Unsafe

import DeepBanana

test_tensor :: Spec
test_tensor = do
  test_broadcast

test_broadcast :: Spec
test_broadcast = describe "DeepBanana.Tensor.broadcast" $ do
  it "Works in a simple example" $ do
    let x = fromList (1:.3:.Z) [1,2,3] :: Tensor 2 CFloat
        expected = fromList (3:.3:.Z) [1,2,3,
                                       1,2,3,
                                       1,2,3] :: Tensor 2 CFloat
        actual = case broadcast (shape expected) x of
          Left err -> error err
          Right out -> out
    actual `shouldBe` expected
