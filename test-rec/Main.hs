{-# LANGUAGE DataKinds #-}
module Main where

import DeepBanana
import DeepBanana.Layer.CUDA
import DeepBanana.Layer.Recurrent

main :: IO ()
main = do
  let nb_samples = 16
      nb_features = 16
      nb_output = 8
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
        lunfold $ h_to_out &&& h_to_h
      minfList = runCUDA 42 $ do
        w_ho <- normal (nb_features:.nb_output:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
        w_hh <- normal (nb_features:.nb_features:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
        let w = HLS $ w_ho `HCons` w_hh `HCons` HNil
        h_0 <- normal (nb_samples:.nb_features:.Z) 0 0.01 :: CUDA (Tensor 2 CFloat)
        forward recnet w h_0
  case minfList of
   Left err -> error err
   Right infList ->putStrLn $ show $ take 5 $ infListToList infList
