module Test.DeepBanana.Layer.Recurrent (
    test_recurrent_layers
  ) where

import Prelude hiding (id, (.))
import Control.Category
import Test.Hspec
import Data.HList.HList
import Data.VectorSpace

import DeepBanana
import DeepBanana.Layer.CUDA
import DeepBanana.Layer.Recurrent

import Test.DeepBanana.Layer.NumericGrad

test_recurrent_layers :: Spec
test_recurrent_layers = return ()

-- test_lunfold :: Spec
-- test_lunfold = describe "DeepBanana.Layer.Recurrent.lunfold" $ do
--   it "Runs somewhat okay" $ do
--     let nb_samples = 16
--         nb_features = 16
--         nb_output = 8
--         h_to_out =
--           linear
--           >+> lreshape (nb_samples:.nb_output:.1:.1:.Z)
--           >+> activation activation_sigmoid
--           >+> lreshape (nb_samples:.nb_output:.Z) :: Layer CUDA CFloat '[Tensor 2 CFloat] (Tensor 2 CFloat) (Tensor 2 CFloat)
--         h_to_h =
--           linear
--           >+> lreshape (nb_samples:.nb_features:.1:.1:.Z)
--           >+> activation activation_sigmoid
--           >+> lreshape (nb_samples:.nb_features:.Z) :: Layer CUDA CFloat '[Tensor 2 CFloat] (Tensor 2 CFloat) (Tensor 2 CFloat)
--         recnet =
--           lunfold $ h_to_out &&& h_to_h
--     runCUDA 42 $ do
--       liftIO $ putStrLn "Initializing weights..."
--       w_ho <- normal (nb_features:.nb_output:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
--       w_hh <- normal (nb_features:.nb_features:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
--       let w = HLS $ w_ho `HCons` w_hh `HCons` HNil
--       liftIO $ putStrLn "Initial state..."
--       h_0 <- normal (nb_samples:.nb_features:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
--       liftIO $ putStrLn "Computing forward pass.. "
--       infList <- forward recnet w h_0
--       liftIO $ putStrLn $ show $ take 3 $ infListToList infList
