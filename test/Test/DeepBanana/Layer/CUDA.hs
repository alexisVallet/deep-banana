{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts, TypeFamilies, DataKinds #-}
module Test.DeepBanana.Layer.CUDA where

import Prelude hiding (id, (.))
import Control.Category
import Foreign.C
import Test.Hspec
import Data.VectorSpace
import Control.Monad
import Control.Monad.Trans
import Data.HList.HList
import GHC.TypeLits
import Data.Proxy

import DeepBanana
import DeepBanana.Layer.CUDA

import Test.DeepBanana.Layer.NumericGrad

test_cuda_layers :: Spec
test_cuda_layers = do
  test_convolution
  test_linear
  test_dropout
  test_sumRows
  test_sumCols
  test_softmax
  test_replicateAsCols
  test_log
  test_mlrCost
  test_pooling2d

runCUDAExp :: CULLong -> CUDA a -> IO a
runCUDAExp seed action = do
  case runCUDA seed action of
   Left err -> do
     expectationFailure $ "Error running a CUDA action: " ++ err
     return undefined
   Right res -> return res

test_dropout :: Spec
test_dropout = describe "DeepBanana.Layer.CUDA.dropout" $ do
  let input = fromList (1:.1:.8:.8:.Z) [1..8*8] :: Tensor 4 CFloat
  it "does nothing for drop_proba = 0" $ do
    out <- runCUDAExp 42 $ forward (dropout 0) (HLS HNil) input
    toList out `shouldBe` toList input
  it "returns all zeros for drop_proba = 1" $ do
    out <- runCUDAExp 42 $ forward (dropout 1) (HLS HNil) input
    toList out `shouldBe` (take (8*8) $ repeat 0)

test_linear :: Spec
test_linear = describe "DeepBanana.Layer.CUDA.linear" $ do
  it "returns the right result for a simple example" $ do
    let
      x = fromList (2:.2:.Z) [1,2,3,4] :: Tensor 2 CFloat
      y = fromList (2:.2:.Z) [5,6,7,8] :: Tensor 2 CFloat
      expected = [19,22,43,50] :: [CFloat]
    actual <- runCUDAExp 42 $ forward linear (HLS $ HCons y HNil) x
    toList actual `shouldBe` expected
  it "has analytic gradient close to numeric gradient" $ do
    let x = normal (16:.5:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat)
        y = normal (16:.6:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat)
        w = do
          w' <- normal (5:.6:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat)
          return $ HLS $ HCons w' HNil
    exp <- runCUDAExp 42 $ check_backward linear w x y
    exp

test_sumCols :: Spec
test_sumCols = describe "DeepBanana.Layer.CUDA.sumCols" $ do
  let x = fromList (2:.2:.Z) [1,2,3,4] :: Tensor 2 CFloat
      expected = [4,6] :: [CFloat]
  it "returns the right result for a simple example" $ do
    actual <- runCUDAExp 42 $ forward sumCols (HLS HNil) x
    toList actual `shouldBe` expected

test_sumRows :: Spec
test_sumRows = describe "DeepBanana.Layer.CUDA.sumRows" $ do
  let x = fromList (2:.2:.Z) [1,2,3,4] :: Tensor 2 CFloat
      expected = [3,7] :: [CFloat]
  it "returns the right result for a simple example" $ do
    actual <- runCUDAExp 42 $ forward sumRows (HLS HNil) x
    toList actual `shouldBe` expected

test_convolution :: Spec
test_convolution = describe "DeepBanana.Layer.CUDA.convolution2d" $ do
  it "has a working forward pass" $ do
    let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
        filter = fromList (1:.1:.3:.3:.Z) [0, 1, 0,
                                           1, 0, 1,
                                           0, 1, 0] :: Tensor 4 CFloat
        img = fromList (1:.1:.3:.3:.Z) [1, 2, 3,
                                        4, 5, 6,
                                        7, 8, 9] :: Tensor 4 CFloat
        expected_out = fromList (1:.1:.3:.3:.Z) [6,  9,  8,
                                                 13, 20, 17,
                                                 12, 21, 14] :: Tensor 4 CFloat
    actual <- runCUDAExp 42 $ forward conv (HLS $ HCons filter HNil) img
    toList actual `shouldBe` toList expected_out
  it "has a correct backward pass" $ do
    let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
        x = normal (1:.1:.8:.8:.Z) 0 0.1 :: CUDA (Tensor 4 CFloat)
        y = normal (1:.1:.8:.8:.Z) 0 0.1 :: CUDA (Tensor 4 CFloat)
        w = do
          w' <- normal (1:.1:.3:.3:.Z) 0 0.1 :: CUDA (Tensor 4 CFloat)
          return $ HLS $ w' `HCons` HNil
    exp <- runCUDAExp 42 $ check_backward conv w x y
    exp

splitEvery :: Int -> [a] -> [[a]]
splitEvery n [] = []
splitEvery n xs = let (start,rest) = splitAt n xs in
  [start] ++ splitEvery n rest

naive_softmax :: Tensor 2 CFloat -> Tensor 2 CFloat
naive_softmax t =
  fromList (shape t) $ concat $ fmap softmaxLine $ splitEvery m $ toList t
  where
    [n,m] = dimensions $ shape t
    softmaxLine xs = let denom = sum $ fmap exp xs in
                      fmap (\x -> exp x / denom) xs

naive_mlrCost :: Tensor 2 CFloat -> Tensor 2 CFloat -> CFloat
naive_mlrCost l x =
  let softmaxed = naive_softmax x
      [n,m] = dimensions $ shape x
      labels = toList l
  in
   (-1/fromIntegral n)
   * (sum
      $ fmap log
      $ fmap sum
      $ splitEvery m
      $ zipWith (*) labels (toList softmaxed))

test_softmax :: Spec
test_softmax = describe "softmax" $ do
  it "CPU and CUDA naive softmaxes return the same results" $ do
    (actual, expected) <- runCUDAExp 42 $ do
      x <- normal (8:.8:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat)
      act <- forward (softmax (8:.8:.Z)) (HLS HNil) x
      return (act, naive_softmax x)
    when (not $ allClose expected actual) $ do
      liftIO $ expectationFailure $
        "Naive and CUDA algorithm return different results:\nnaive: "
        ++ show expected ++ "\ncudnn: " ++ show actual
  it "has a numerically correct backward pass" $ do
    exp <- runCUDAExp 42 $ check_backward (softmax (8:.8:.Z)) (return $ HLS HNil)
           (normal (8:.8:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat))
           (normal (8:.8:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat))
    exp

test_replicateAsCols :: Spec
test_replicateAsCols = describe "DeepBanana.Layer.CUDA.replicateAsCols" $ do
  let x = fromList (4:.Z) [1,2,3,4] :: Tensor 1 CFloat
      expected = [1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2,
                  3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4]
  it "returns the right result for a simple example" $ do
    actual <- runCUDAExp 42 $ forward (replicateAsCols 5) (HLS HNil) x
    toList actual `shouldBe` expected
  it "has a backward pass close to the numerical backward pass" $ do
    exp <- runCUDAExp 42 $ check_backward (replicateAsCols 5) (return $ HLS $ HNil)
           (normal (56:.Z) 0 0.1 :: CUDA (Tensor 1 CFloat))
           (normal (56:.5:.Z) 0 0.1 :: CUDA (Tensor 2 CFloat))
    exp

test_log :: Spec
test_log = describe "DeepBanana.Layer.CUDA.log" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    let x = uniform (8:.8:.Z) >>= return . (10E-5 +) :: CUDA (Tensor 2 CFloat)
        y = uniform (8:.8:.Z) >>= return . (10E-5 +) :: CUDA (Tensor 2 CFloat)
    exp <- runCUDAExp 42 $ check_backward llog (return $ HLS HNil) x y
    exp

test_mlrCost :: Spec
test_mlrCost = describe "DeepBanana.Layer.CUDA.mlrCost" $ do
  let labels = fromList (8:.8:.Z) [1,0,0,0,1,1,1,0,
                                   0,1,1,1,1,1,0,1,
                                   1,1,1,1,1,1,0,0,
                                   0,1,0,0,0,0,1,0,
                                   1,0,0,0,1,1,1,0,
                                   0,1,1,1,1,1,0,1,
                                   1,1,1,1,1,1,0,0,
                                   0,1,0,0,0,0,1,0] :: Tensor 2 (CFloat)
      x = fromList (8:.8:.Z) [1..8*8] :: Tensor 2 (CFloat)
      expected = naive_mlrCost labels x
  it "returns the same results as a naive CPU implementation" $ do
    actual <- runCUDAExp 42 $ forward (mlrCost (8:.8:.Z) >+> toScalar) (HLS HNil) (labels,x)
    actual `shouldBe` expected
  it "has an analytic gradient close to the numeric gradient" $ do
    exp <- runCUDAExp 42 $ check_backward (mlrCost (8:.8:.Z)) (return $ HLS HNil) (return (labels,x)) (return 1)
    exp

test_pooling2d :: Spec
test_pooling2d = describe "DeepBanana.Layer.CUDA.pooling2d" $ do
  it "has an analytic gradient close to the numeric gradient" $ do
    exp <- runCUDAExp 42 $ check_backward (pooling2d (2,2) (0,0) (2,2) pooling_max) (return $ HLS HNil) (normal (1:.1:.4:.4:.Z) 0 0.1 :: CUDA (Tensor 4 CFloat)) (normal (1:.1:.2:.2:.Z) 0 0.1 :: CUDA (Tensor 4 CFloat))
    exp
