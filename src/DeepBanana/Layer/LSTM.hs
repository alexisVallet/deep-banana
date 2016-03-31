{-# LANGUAGE GADTs #-}
module DeepBanana.Layer.LSTM (
    lstm
  , LSTMWeights
  ) where

import DeepBanana.Prelude
import DeepBanana.Layer
import DeepBanana.Layer.CUDA
import DeepBanana.Tensor

type LSTMWeights a = '[
    Tensor 2 a
  , Tensor 1 a
  , Tensor 2 a
  , Tensor 1 a
  , Tensor 2 a
  , Tensor 2 a
  , Tensor 1 a
  ]

lstm :: (Monad m, TensorScalar a)
     => Layer (CUDAT m) a (LSTMWeights a) (Tensor 2 a, Tensor 2 a) (Tensor 2 a, Tensor 2 a)
lstm =
  let
    diag_linear = Layer $ \(HLS (HCons w HNil)) x -> do
      let n:.m:.Z = shape x
          diag_linear' = (replicateAsRows n *** id') >+> multiply
      (y, bwdy) <- forwardBackward diag_linear' (HLS HNil) (w, x)
      return (y, \y' -> let (_, (w', x')) = bwdy y' in (HLS $ HCons w' HNil, x'))
    sigm = activation activation_sigmoid
    tanh = activation activation_tanh
    h_tm1 = first
    c_tm1 = second
    i_t = (h_tm1 >+> linear) &&& (c_tm1 >+> diag_linear) >+> add >+> sigm
    f_t = (h_tm1 >+> linear) &&& (c_tm1 >+> diag_linear) >+> add >+> sigm
    c_t = (f_t &&& c_tm1 >+> multiply) &&& (i_t &&& (h_tm1 >+> linear >+> tanh) >+> multiply) >+> add
  in h_tm1 &&& c_t
     >+> ((linear *** diag_linear >+> add >+> sigm) &&& second) -- o_t, c_t
     >+> ((id' *** tanh) >+> multiply) &&& second
