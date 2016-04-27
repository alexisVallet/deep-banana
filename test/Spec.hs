import Test.Hspec

import qualified Foreign.CUDA as CUDA

import Test.DeepBanana.Layer.CUDA
import Test.DeepBanana.Layer.Numeric
import Test.DeepBanana.Tensor

import DeepBanana.Prelude

main :: IO ()
main = do
  CUDA.set 2
  hspec $ do
    test_tensor
    test_numeric_layers
    test_cuda_layers
