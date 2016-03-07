import Test.Hspec

import qualified Foreign.CUDA as CUDA

import Test.DeepBanana.Layer.CUDA
import Test.DeepBanana.Layer.Recurrent
import Test.DeepBanana.Layer.LSTM
import Test.DeepBanana.Tensor

main :: IO ()
main = do
  CUDA.set 2
  hspec $ do
    test_tensor
    test_cuda_layers
    test_recurrent_layers
    test_lstm
