module DeepBanana.Layer.Parallel (
    layerPar
  , dataPar
  ) where

import Control.Parallel.Strategies

import DeepBanana.Device
import DeepBanana.Tensor.Exception
import DeepBanana.Exception
import DeepBanana.Layer
import DeepBanana.Layer.CUDA
import DeepBanana.Prelude

layerPar :: forall m d s w1 a1 b1 w2 a2 b2
         .  (MonadCuda m, Concat w1 w2, Device d,
             NFData b1, NFData (Weights s w1,a1),
             NFData b2, NFData (Weights s w2, a2))
         => Proxy d
         -> Layer Cuda s w1 a1 b1
         -> Layer Cuda s w2 a2 b2
         -> Layer m s (ConcatRes w1 w2) (a1, a2) (b1, b2)
layerPar p l1 l2 = Layer $ \(W w1w2) (a1,a2) -> do
  let (w1,w2) = hsplit w1w2 :: (HList w1, HList w2)
  gen1 <- splitGenerator p
  gen2 <- splitGenerator p
  let (eb1bwd1,eb2bwd2) =
        (fst $ runCuda gen1 $ forwardBackward l1 (W w1) a1,
         fst $ runCuda gen2 $ forwardBackward l2 (W w2) a2)
        `using` parTuple2 rdeepseq rdeepseq
  (b1,bwd1) <- embedExcept eb1bwd1
  (b2,bwd2) <- embedExcept eb2bwd2
  return ((b1,b2), \(b1',b2') -> let ((W w1',a1'), (W w2', a2')) =
                                       (bwd1 b1', bwd2 b2')
                                       `using` parTuple2 rdeepseq rdeepseq
                                 in (W $ hconcat w1' w2', (a1', a2')))

dataPar :: forall t m d w1 w2 b1 s a1 b2 a2
        . (MonadCuda m, Concat w1 w2,
           NFData b1, NFData (Weights s w1,a1),
           NFData b2, NFData (Weights s w2, a2),
           DeviceTransfer (Weights s w1) (Weights s w2),
           DeviceTransfer (Weights s w2) (Weights s w1),
           Device d,
           FixShape (Weights s w1), Floating (FixScalar (Weights s w1)))
        => Proxy d
        -> Layer Cuda s w1 a1 b1
        -> Layer Cuda s w2 a2 b2
        -> Layer m s w1 (a1, a2) (b1, b2)
dataPar p l1 l2 = Layer $ \w1 a1a2 -> do
  w2 <- transfer w1 :: m (Weights s w2)
  let w1w2 = W $ hconcat (unWeights w1) (unWeights w2)
  (b1b2, bwdb1b2) <- forwardBackward (layerPar p l1 l2) w1w2 a1a2
  return (b1b2, \b1b2' -> let (W w1w2', a1a2') = bwdb1b2 b1b2'
                              (w1', w2') = hsplit w1w2' :: (HList w1, HList w2)
                              w' = unsafeRunExcept
                                   $ (liftVec (\_w1 _w2 -> 0.5 *^ (_w1 ^+^ _w2))
                                      (W w1') (transfer' (W w2' :: Weights s w2))
                                      :: Either IncompatibleShape (Weights s w1))
                          in (w', a1a2'))
