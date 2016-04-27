{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies, DataKinds, BangPatterns #-}
module Test.DeepBanana.Layer.CUDA where

import Control.Category
import Foreign.C
import Test.Hspec
import Data.VectorSpace
import Control.Monad
import Control.Monad.Trans
import GHC.TypeLits
import Data.Proxy

import DeepBanana
import DeepBanana.Prelude

import Config
import Test.DeepBanana.Layer.NumericGrad

test_cuda_layers :: Spec
test_cuda_layers = do
  test_convolution
  test_bias
  test_dropout
  test_sumRows
  test_softmax
  test_replicateAsCols
  test_log
  test_mlrCost
  test_pooling2d
  test_batchNormalization

test_dropout :: Spec
test_dropout = describe "DeepBanana.Layer.Cuda.dropout" $ do
  let input = tensorFromList' (1:.1:.8:.8:.Z) [1..8*8] :: Tensor TestDevice (Dim 4) CFloat
      gen = generator 42
  it "does nothing for drop_proba = 0" $ do
    let (out,_) = runCudaEx gen $ forward (dropout 0) (W Z) input
    out `shouldBe` input
  it "returns all zeros for drop_proba = 1" $ do
    let (out,_) = runCudaEx gen $ forward (dropout 1) (W Z) input
    tensorToList out `shouldBe` (take (8*8) $ repeat 0)

test_sumRows :: Spec
test_sumRows = describe "DeepBanana.Layer.Cuda.sumRows" $ do
  let x = tensorFromList' (2:.2:.Z) [1,2,3,4] :: Tensor TestDevice (Dim 2) CFloat
      expected = [3,7] :: [CFloat]
  it "returns the right result for a simple example" $ do
    let (actual,_) = runCudaEx (generator 42) $ forward sumRows (W Z) x
    tensorToList actual `shouldBe` expected

test_convolution :: Spec
test_convolution = describe "DeepBanana.Layer.Cuda.convolution2d" $ do
  it "has a working forward pass" $ do
    let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm convolution_bwd_data_algo_0 convolution_bwd_filter_algo_0
        filter = tensorFromList' (1:.1:.3:.3:.Z) [0, 1, 0,
                                           1, 0, 1,
                                           0, 1, 0] :: Tensor TestDevice (Dim 4) CFloat
        img = tensorFromList' (1:.1:.3:.3:.Z) [1, 2, 3,
                                        4, 5, 6,
                                        7, 8, 9] :: Tensor TestDevice (Dim 4) CFloat
        expected_out = tensorFromList' (1:.1:.3:.3:.Z) [6,  9,  8,
                                                 13, 20, 17,
                                                 12, 21, 14] :: Tensor TestDevice (Dim 4) CFloat
    let (actual,_) = runCudaEx (generator 42) $ forward conv (W $ (:.) filter Z) img
    tensorToList actual `shouldBe` tensorToList expected_out
  it "has a correct backward pass" $ do
    let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm convolution_bwd_data_algo_0 convolution_bwd_filter_algo_0
        x = normal (1:.1:.8:.8:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        y = normal (1:.1:.8:.8:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        w = do
          w' <- normal (1:.1:.3:.3:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
          return $ W $ w' :. Z
    runCudaTEx (generator 42) $ check_backward conv w x y
    return ()

test_bias :: Spec
test_bias = describe "DeepBanana.Layer.CUDA.bias" $ do
  it "has a correct forward pass" $ do
    let x = tensorFromList' (1:.2:.2:.2:.Z)
            $ [1,2,3,4,5,6,7,8] :: Tensor TestDevice (Dim 4) CFloat
        y = tensorFromList' (2:.Z) [1,2] :: Tensor TestDevice (Dim 1) CFloat
        expected = tensorFromList' (1:.2:.2:.2:.Z)
                   $ [2,3,4,5,7,8,9,10] :: Tensor TestDevice (Dim 4) CFloat
        (actual,_) = runCudaEx (generator 42)  $ forward bias (W $ (:.) y Z) x
    actual `shouldBe` expected
  it "has a corect backward pass" $ do
    let x = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        y = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)
        w = do
          w' <- normal (3:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 1) CFloat)
          return $ W $ w' :. Z
    runCudaTEx (generator 42)
      $ check_backward bias w x y
    return ()
        

splitEvery :: Int -> [a] -> [[a]]
splitEvery n [] = []
splitEvery n xs = let (start,rest) = splitAt n xs in
  [start] ++ splitEvery n rest

naive_softmax :: Tensor TestDevice (Dim 2) CFloat -> Tensor TestDevice (Dim 2) CFloat
naive_softmax t =
  tensorFromList' (shape t) $ concat $ fmap softmaxLine $ splitEvery m $ tensorToList t
  where
    [n,m] = dimensions $ shape t
    softmaxLine xs = let denom = sum $ fmap exp xs in
                      fmap (\x -> exp x / denom) xs

naive_mlrCost :: Tensor TestDevice (Dim 2) CFloat -> Tensor TestDevice (Dim 2) CFloat -> CFloat
naive_mlrCost l x =
  let softmaxed = naive_softmax x
      [n,m] = dimensions $ shape x
      labels = tensorToList l
  in
   (-1/fromIntegral n)
   * (sum
      $ fmap log
      $ fmap sum
      $ splitEvery m
      $ zipWith (*) labels (tensorToList softmaxed))

test_softmax :: Spec
test_softmax = describe "softmax" $ do
  it "returns the right result for a simple example" $ do
    let x = tensorFromList' (3:.3:.Z) [0, 1, 0,
                                       1, 0, 1,
                                       1, 1, 1] :: Tensor TestDevice (Dim 2) CFloat
        expected = tensorFromList' (3:.3:.Z) [
          exp 0/(exp 0 + exp 1 + exp 0), exp 1/(exp 0 + exp 1 + exp 0), exp 0/(exp 0 + exp 1 + exp 0),
          exp 1/(exp 1 + exp 0 + exp 1), exp 0/(exp 1 + exp 0 + exp 1), exp 1/(exp 1 + exp 0 + exp 1),
          1/3, 1/3, 1/3] :: Tensor TestDevice (Dim 2) CFloat
        (actual,_) = runCudaEx (generator 42)
                     $ forward (softmax softmax_accurate softmax_mode_instance) (W Z) x
    actual `shouldBe` expected
  it "has a numerically correct backward pass" $ do
   runCudaTEx (generator 42) $ check_backward (softmax softmax_accurate softmax_mode_instance) (return $ W Z)
           (normal (8:.8:.Z) 5 1 :: CudaT IO (Tensor TestDevice (Dim 2) CFloat))
           (normal (8:.8:.Z) 5 1 :: CudaT IO (Tensor TestDevice (Dim 2) CFloat))
   return ()

test_replicateAsCols :: Spec
test_replicateAsCols = describe "DeepBanana.Layer.Cuda.replicateAsCols" $ do
  let x = tensorFromList' (4:.Z) [1,2,3,4] :: Tensor TestDevice (Dim 1) CFloat
      expected = [1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4]
  it "returns the right result for a simple example" $ do
    let (actual,_) = runCudaEx (generator 42) $ forward (replicateAsCols 5) (W Z) x
    tensorToList actual `shouldBe` expected
  it "has a backward pass close to the numerical backward pass" $ do
    runCudaTEx (generator 42) $ check_backward (replicateAsCols 5) (return $ W $ Z)
           (normal (56:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 1) CFloat))
           (normal (56:.5:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 2) CFloat))
    return ()

test_log :: Spec
test_log = describe "DeepBanana.Layer.Cuda.llog" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    let x = do
          x' <- uniform $ fixed (Proxy :: Proxy [8,8])
          return $ x' + 10E-5 :: CudaT IO (Tensor TestDevice (Fixed [8,8]) CFloat)
    runCudaTEx (generator 42) $ check_backward llog (return $ W Z) x x
    return ()

test_mlrCost :: Spec
test_mlrCost = describe "DeepBanana.Layer.Cuda.mlrCost" $ do
  let labels = tensorFromList' (8:.8:.Z) [1,0,0,0,1,1,1,0,
                                   0,1,1,1,1,1,0,1,
                                   1,1,1,1,1,1,0,0,
                                   0,1,0,0,0,0,1,0,
                                   1,0,0,0,1,1,1,0,
                                   0,1,1,1,1,1,0,1,
                                   1,1,1,1,1,1,0,0,
                                   0,1,0,0,0,0,1,0] :: Tensor TestDevice (Dim 2) (CFloat)
      x = tensorFromList' (8:.8:.Z) [1..8*8] :: Tensor TestDevice (Dim 2) (CFloat)
      expected = naive_mlrCost labels x
  it "returns the same results as a naive CPU implementation" $ do
    let (actual,_) = runCudaEx (generator 42) $ forward (mlrCost (8:.8:.Z) >+> toScalar) (W Z) (labels,x)
    actual `shouldBe` expected
  it "has an analytic gradient close to the numeric gradient" $ do
    runCudaTEx (generator 42) $ check_backward (mlrCost (8:.8:.Z)) (return $ W Z) (return (labels,x)) (return (ones' (8:.Z)) :: CudaT IO (Tensor TestDevice (Dim 1) CFloat))
    return ()

test_pooling2d :: Spec
test_pooling2d = describe "DeepBanana.Layer.Cuda.pooling2d" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    runCudaTEx (generator 42) $ check_backward (pooling2d (2,2) (0,0) (2,2) pooling_max) (return $ W Z) (normal (1:.1:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat)) (normal (1:.1:.2:.2:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat))
    return ()

test_batchNormalization :: Spec
test_batchNormalization = describe "DeepBanana.Layer.CUDA.CuDNN.batchNormalization" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    runCudaTEx (generator 42)
      $ check_backward (batchNormalization batchnorm_spatial 10E-5)
      (do
          (scale,bias) <- pure (,)
                          <*> normal (1:.5:.1:.1:.Z) 0 0.1
                          <*> normal (1:.5:.1:.1:.Z) 0 0.1
          return $ W $ scale:.bias:.Z)
      (normal (2:.5:.3:.3:.Z) 0 0.1)
      (normal (2:.5:.3:.3:.Z) 0 0.1 :: CudaT IO (Tensor TestDevice (Dim 4) CFloat))
    return ()
