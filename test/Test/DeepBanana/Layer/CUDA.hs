{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies, DataKinds #-}
module Test.DeepBanana.Layer.CUDA where

import Prelude hiding (id, (.))
import Control.Category
import Foreign.C
import Test.Hspec
import Data.VectorSpace
import Control.Monad
import Control.Monad.Trans
import GHC.TypeLits
import Data.Proxy

import DeepBanana
import DeepBanana.Layer.CUDA

import Test.DeepBanana.Layer.NumericGrad

test_cuda_layers :: Spec
test_cuda_layers = do
  test_convolution
  test_bias
  test_linear
  test_dropout
  test_sumRows
  test_sumCols
  test_softmax
  test_replicateAsCols
  test_log
  test_mlrCost
  test_pooling2d

test_dropout :: Spec
test_dropout = describe "DeepBanana.Layer.Cuda.dropout" $ do
  let input = tensorFromList' (1:.1:.8:.8:.Z) [1..8*8] :: Tensor 4 CFloat
  it "does nothing for drop_proba = 0" $ do
    let (out,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward (dropout 0) (W Z) input
    tensorToList out `shouldBe` tensorToList input
  it "returns all zeros for drop_proba = 1" $ do
    let (out,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward (dropout 1) (W Z) input
    tensorToList out `shouldBe` (take (8*8) $ repeat 0)

test_linear :: Spec
test_linear = describe "DeepBanana.Layer.Cuda.linear" $ do
  it "returns the right result for a simple example" $ do
    let
      x = tensorFromList' (2:.2:.Z) [1,2,3,4] :: Tensor 2 CFloat
      y = tensorFromList' (2:.2:.Z) [5,6,7,8] :: Tensor 2 CFloat
      expected = [19,22,43,50] :: [CFloat]
    let (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward linear (W $ (:.) y Z) x
    tensorToList actual `shouldBe` expected
  it "has analytic gradient close to numeric gradient" $ do
    let x = normal (16:.5:.Z) 0 0.1 :: CudaT IO (Tensor 2 CFloat)
        y = normal (16:.6:.Z) 0 0.1 :: CudaT IO (Tensor 2 CFloat)
        w = do
          w' <- normal (5:.6:.Z) 0 0.1 :: CudaT IO (Tensor 2 CFloat)
          return $ W $ (:.) w' Z
    runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward linear w x y
    return ()

test_sumCols :: Spec
test_sumCols = describe "DeepBanana.Layer.Cuda.sumCols" $ do
  let x = tensorFromList' (2:.2:.Z) [1,2,3,4] :: Tensor 2 CFloat
      expected = [4,6] :: [CFloat]
  it "returns the right result for a simple example" $ do
    let (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward sumCols (W Z) x
    tensorToList actual `shouldBe` expected

test_sumRows :: Spec
test_sumRows = describe "DeepBanana.Layer.Cuda.sumRows" $ do
  let x = tensorFromList' (2:.2:.Z) [1,2,3,4] :: Tensor 2 CFloat
      expected = [3,7] :: [CFloat]
  it "returns the right result for a simple example" $ do
    let (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward sumRows (W Z) x
    tensorToList actual `shouldBe` expected

test_convolution :: Spec
test_convolution = describe "DeepBanana.Layer.Cuda.convolution2d" $ do
  it "has a working forward pass" $ do
    let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
        filter = tensorFromList' (1:.1:.3:.3:.Z) [0, 1, 0,
                                           1, 0, 1,
                                           0, 1, 0] :: Tensor 4 CFloat
        img = tensorFromList' (1:.1:.3:.3:.Z) [1, 2, 3,
                                        4, 5, 6,
                                        7, 8, 9] :: Tensor 4 CFloat
        expected_out = tensorFromList' (1:.1:.3:.3:.Z) [6,  9,  8,
                                                 13, 20, 17,
                                                 12, 21, 14] :: Tensor 4 CFloat
    let (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward conv (W $ (:.) filter Z) img
    tensorToList actual `shouldBe` tensorToList expected_out
  it "has a correct backward pass" $ do
    let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
        x = normal (1:.1:.8:.8:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat)
        y = normal (1:.1:.8:.8:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat)
        w = do
          w' <- normal (1:.1:.3:.3:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat)
          return $ W $ w' :. Z
    runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward conv w x y
    return ()

test_bias :: Spec
test_bias = describe "DeepBanana.Layer.CUDA.bias" $ do
  it "has a correct forward pass" $ do
    let x = tensorFromList' (1:.2:.2:.2:.Z)
            $ [1,2,3,4,5,6,7,8] :: Tensor 4 CFloat
        y = tensorFromList' (2:.Z) [1,2] :: Tensor 1 CFloat
        expected = tensorFromList' (1:.2:.2:.2:.Z)
                   $ [2,3,4,5,7,8,9,10] :: Tensor 4 CFloat
        (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42)  $ forward bias (W $ (:.) y Z) x
    actual `shouldBe` expected
  it "has a corect backward pass" $ do
    let x = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat)
        y = normal (1:.3:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat)
        w = do
          w' <- normal (3:.Z) 0 0.1 :: CudaT IO (Tensor 1 CFloat)
          return $ W $ w' :. Z
    runCudaTEx (createGenerator rng_pseudo_default 42)
      $ check_backward bias w x y
    return ()
        

splitEvery :: Int -> [a] -> [[a]]
splitEvery n [] = []
splitEvery n xs = let (start,rest) = splitAt n xs in
  [start] ++ splitEvery n rest

naive_softmax :: Tensor 2 CFloat -> Tensor 2 CFloat
naive_softmax t =
  tensorFromList' (shape t) $ concat $ fmap softmaxLine $ splitEvery m $ tensorToList t
  where
    [n,m] = dimensions $ shape t
    softmaxLine xs = let denom = sum $ fmap exp xs in
                      fmap (\x -> exp x / denom) xs

naive_mlrCost :: Tensor 2 CFloat -> Tensor 2 CFloat -> CFloat
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
  it "CPU and Cuda naive softmaxes return the same results" $ do
    let ((actual, expected),_) = runCudaEx (createGenerator rng_pseudo_default 42) $ do
          x <- normal (8:.8:.Z) 0 0.1 :: Cuda (Tensor 2 CFloat)
          act <- forward (softmax (8:.8:.Z)) (W Z) x
          return (act, naive_softmax x)
    when (not $ allClose expected actual) $ do
      liftIO $ expectationFailure $
        "Naive and Cuda algorithm return different results:\nnaive: "
        ++ show expected ++ "\ncudnn: " ++ show actual :: IO ()
  it "has a numerically correct backward pass" $ do
   runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward (softmax (8:.8:.Z)) (return $ W Z)
           (normal (8:.8:.Z) 0 0.1 :: CudaT IO (Tensor 2 CFloat))
           (normal (8:.8:.Z) 0 0.1 :: CudaT IO (Tensor 2 CFloat))
   return ()

test_replicateAsCols :: Spec
test_replicateAsCols = describe "DeepBanana.Layer.Cuda.replicateAsCols" $ do
  let x = tensorFromList' (4:.Z) [1,2,3,4] :: Tensor 1 CFloat
      expected = [1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4]
  it "returns the right result for a simple example" $ do
    let (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward (replicateAsCols 5) (W Z) x
    tensorToList actual `shouldBe` expected
  it "has a backward pass close to the numerical backward pass" $ do
    runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward (replicateAsCols 5) (return $ W $ Z)
           (normal (56:.Z) 0 0.1 :: CudaT IO (Tensor 1 CFloat))
           (normal (56:.5:.Z) 0 0.1 :: CudaT IO (Tensor 2 CFloat))
    return ()

test_log :: Spec
test_log = describe "DeepBanana.Layer.Cuda.llog" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    let x = uniform (8:.8:.Z) >>= return . (10E-5 +) :: CudaT IO (Tensor 2 CFloat)
        y = uniform (8:.8:.Z) >>= return . (10E-5 +) :: CudaT IO (Tensor 2 CFloat)
    runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward llog (return $ W Z) x y
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
                                   0,1,0,0,0,0,1,0] :: Tensor 2 (CFloat)
      x = tensorFromList' (8:.8:.Z) [1..8*8] :: Tensor 2 (CFloat)
      expected = naive_mlrCost labels x
  it "returns the same results as a naive CPU implementation" $ do
    let (actual,_) = runCudaEx (createGenerator rng_pseudo_default 42) $ forward (mlrCost (8:.8:.Z) >+> toScalar) (W Z) (labels,x)
    actual `shouldBe` expected
  it "has an analytic gradient close to the numeric gradient" $ do
    runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward (mlrCost (8:.8:.Z)) (return $ W Z) (return (labels,x)) (return 1 :: CudaT IO (Tensor 1 CFloat))
    return ()

test_pooling2d :: Spec
test_pooling2d = describe "DeepBanana.Layer.Cuda.pooling2d" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    runCudaTEx (createGenerator rng_pseudo_default 42) $ check_backward (pooling2d (2,2) (0,0) (2,2) pooling_max) (return $ W Z) (normal (1:.1:.4:.4:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat)) (normal (1:.1:.2:.2:.Z) 0 0.1 :: CudaT IO (Tensor 4 CFloat))
    return ()
