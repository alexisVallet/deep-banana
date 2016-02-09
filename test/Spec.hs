import Test.Hspec

import Test.DeepBanana.Layer.CUDA
import Test.DeepBanana.Layer.Recurrent
import Test.DeepBanana.Tensor

main :: IO ()
main = do
  hspec $ do
    test_tensor
    test_cuda_layers
    test_recurrent_layers
