{-# LANGUAGE DataKinds, OverloadedStrings, FlexibleContexts, TypeFamilies #-}
module Main where

import Control.Arrow (first, second)
import qualified Data.List as L
import Data.Maybe (fromJust)
import qualified Data.Vector.Storable as VS
import DeepBanana hiding (first, second)
import DeepBanana.Prelude
import qualified Foreign.CUDA as CUDA
import MNIST
import qualified Pipes.Prelude as P
import System.Mem
import Vision.Image hiding (map, shape)
import Vision.Image.Storage.DevIL

type Weights = '[Tensor 4 CFloat, Tensor 4 CFloat, Tensor 4 CFloat, Tensor 4 CFloat] 

type Training = CudaT IO

main :: IO ()
main = do
  let train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
      train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
      test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
      test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
      mnist_dir = "data" </> "mnist"
      train_images_file = mnist_dir </> "train-images.ubyte.gz"
      train_labels_file = mnist_dir </> "train-labels.ubyte.gz"
      test_images_file = mnist_dir </> "test-images.ubyte.gz"
      test_labels_file = mnist_dir </> "test-labels.ubyte.gz"
      nb_labels = 10
      batch_size = 32
  emnist_train <- runExceptT $ P.toListM
                  $ load_mnist (Just train_images_url) (Just train_labels_url)
                  train_images_file train_labels_file
  emnist_val <- runExceptT $ P.toListM
                $ load_mnist (Just test_images_url) (Just test_labels_url)
                test_images_file test_labels_file
  let gen = createGenerator rng_pseudo_default 42
  case pure (,) <*> emnist_train <*> emnist_val of
   Left err -> ioError $ userError $ "Error loading the dataset: " ++ err
   Right (mnist_train,mnist_val) -> do
     (merr,_) <- runCudaT gen $ do
       w_0 <- init_weights nb_labels
       let
         nb_val_batches = length mnist_val `div` batch_size
         preprocessing =
           P.map (\(i,l) -> (i, [l]))
           >-> batch_images nb_labels batch_size
           >-> P.map (\(b,l) -> (VS.map grey_to_float b, l))
           >-> batch_to_gpu (batch_size:.1:.28:.28:.Z) (batch_size:.nb_labels:.Z)
           :: Pipe (Grey, Int) (Tensor 4 CFloat, Tensor 2 CFloat) Training ()
         validate (c,w) = do
           let sampleAccuracy = forever $ do
                 (b,l) <- await
                 confmat <- lift $ predict batch_size nb_labels w b
                 let rows:.cols:.Z = shape confmat
                     predAndGt =
                       zip
                       (splitEvery cols $ tensorToList confmat)
                       (splitEvery cols $ tensorToList l)
                 forM_ predAndGt $ \(predconfs,gtconfs) -> do
                   let (predl,conf) = argmax predconfs
                       (gtl,_) = argmax gtconfs
                   yield ([predl],[gtl])
           valAccuracy <- accuracy $ randomize mnist_val
                          >-> preprocessing
                          >-> sampleAccuracy
           putStrLn $ "Validation accuracy: " ++ pack (show valAccuracy)
         cost_grad' w b = cost_grad batch_size nb_labels w b
         optimize = sgd (rmsprop 0.001 0.9 0.1) cost_grad' w_0 :: Pipe (Tensor 4 CFloat, Tensor 2 CFloat) (CFloat, HLSpace CFloat Weights) Training ()
       runEffect
         $ forever (randomize mnist_train)
         >-> preprocessing
         >-> optimize
         >-> runEvery 1000 validate
         >-> print_info
     case merr of
      Left err -> throw err
      Right _ -> return ()

splitEvery :: Int -> [a] -> [[a]]
splitEvery n = L.unfoldr (\xs -> case splitAt n xs of
                                  ([],_) -> Nothing
                                  (pref,suff) -> Just (pref,suff))

argmax :: (Ord a) => [a] -> (Int, a)
argmax = L.maximumBy (\(i1,x1) (i2,x2) -> compare x1 x2) . zip [0..]

cudaToTraining :: Cuda a -> Training a
cudaToTraining = cudaHoist generalize

model :: Int -> Int -> Layer Cuda CFloat Weights (Tensor 4 CFloat) (Tensor 2 CFloat)
model batch_size nb_labels =
  let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
             >+> activation activation_relu
      pool = pooling2d (2,2) (1,1) (2,2) pooling_max
      pool_avg = pooling2d (3,3) (1,1) (3,3) pooling_average_count_include_padding in
  conv
  >+> pool -- 14*14
  >+> conv
  >+> pool -- 7*7
  >+> conv
  >+> pool -- 3*3
  >+> conv
  >+> pool_avg -- 1*1
  >+> lreshape (batch_size:.nb_labels:.Z)

nnet :: Int -> Int
     -> Layer Cuda CFloat Weights (Tensor 2 CFloat, Tensor 4 CFloat) CFloat
nnet batch_size nb_labels =
  let criteria = mlrCost (batch_size:.nb_labels:.Z) >+> toScalar in
   (id' *** model batch_size nb_labels) >+> criteria
  
cost_grad :: Int -> Int -> HLSpace CFloat Weights
          -> (Tensor 4 CFloat, Tensor 2 CFloat)
          -> Training (CFloat, HLSpace CFloat Weights)
cost_grad batch_size nb_labels w_t (batch,labels) = do
  (cost, bwd) <- cudaToTraining $ forwardBackward (nnet batch_size nb_labels) w_t (labels,batch)
  let (w', _) = bwd (1 :: CFloat)
  return (cost, w')

predict :: Int -> Int -> HLSpace CFloat Weights -> Tensor 4 CFloat -> Training (Tensor 2 CFloat)
predict batch_size nb_labels w_t batch = do
  cudaToTraining $ forward (model batch_size nb_labels) w_t batch

he_init :: Dim 4 -> Training (Tensor 4 CFloat)
he_init s@(_:.c:.fh:.fw:.Z) =
  normal s 0 (sqrt (2 / (fromIntegral c * fromIntegral fh * fromIntegral fw)))

init_weights :: Int -> Training (HLSpace CFloat Weights)
init_weights nb_labels = do
  x <- pure hBuild
       <*> he_init (32:.1:.3:.3:.Z)
       <*> he_init (64:.32:.3:.3:.Z)
       <*> he_init (128:.64:.3:.3:.Z)
       <*> he_init (10:.128:.3:.3:.Z)
  return $ HLS $ hEnd x

grey_to_float :: Word8 -> CFloat
grey_to_float i = (fromIntegral i - 128) / 255

-- Printing the current optimization state.
data InfoState = InfoState {
  rolling_cost :: Maybe CFloat
  }

print_info :: Consumer (CFloat, HLSpace CFloat Weights) Training ()
print_info = flip evalStateT (InfoState Nothing) $ forM_ [1..] $ \i -> do
  (cost, weights) <- lift $ await
  roll_cost <- do
    mcur_roll_cost <- gets rolling_cost
    case mcur_roll_cost of
     Nothing -> modify (\s -> s {rolling_cost = Just cost})
     Just cur_roll_cost ->
       modify (\s -> s {rolling_cost = Just $ cur_roll_cost * 0.9 + cost * 0.1})
    fmap fromJust $ gets rolling_cost
  when (i `rem` 50 == 0) $ do
    putStrLn $ "Iteration " ++ pack (show i) ++ " cost rolling average: "
      ++ pack (show roll_cost)
