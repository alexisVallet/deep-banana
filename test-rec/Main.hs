{-# LANGUAGE DataKinds #-}
module Main where

import Control.Monad

import DeepBanana
import DeepBanana.Layer.CUDA
import DeepBanana.Layer.Recurrent

main :: IO ()
main = do
  let nb_samples = 16
      nb_features = 16
      nb_output = 8
      out_length = 5
      h_to_out =
        linear
        >+> lreshape (nb_samples:.nb_output:.1:.1:.Z)
        >+> activation activation_sigmoid
        >+> lreshape (nb_samples:.nb_output:.Z) :: Layer CUDA CFloat '[Tensor 2 CFloat] (Tensor 2 CFloat) (Tensor 2 CFloat)
      h_to_h =
        linear
        >+> lreshape (nb_samples:.nb_features:.1:.1:.Z)
        >+> activation activation_sigmoid
        >+> lreshape (nb_samples:.nb_features:.Z) :: Layer CUDA CFloat '[Tensor 2 CFloat] (Tensor 2 CFloat) (Tensor 2 CFloat)
      recnet =
        lunfold' out_length (h_to_out &&& h_to_h)
      (list, bwd) = runCUDA 42 $ do
        w_ho <- normal (nb_features:.nb_output:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
        w_hh <- normal (nb_features:.nb_features:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
        let w = HLS $ w_ho `HCons` w_hh `HCons` HNil
        h_0 <- normal (nb_samples:.nb_features:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
        forwardBackward recnet w h_0
  putStrLn $ show list
  putStrLn $ show $ bwd $ runCUDA 42 $ replicateM out_length $
    normal (nb_samples:.nb_output:.Z) 0 0.01
