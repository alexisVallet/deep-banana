import Test.Hspec

import Test.DeepBanana.Layer.CUDA

main :: IO ()
main = do
  hspec $ do
    test_cuda_layers
