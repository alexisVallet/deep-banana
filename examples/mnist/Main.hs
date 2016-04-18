{-# LANGUAGE DataKinds, PolyKinds, OverloadedStrings, FlexibleContexts, TypeFamilies #-}
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

type MNISTWeights d = '[
  Tensor d 4 CFloat, Tensor d 1 CFloat,
  Tensor d 4 CFloat, Tensor d 1 CFloat,
  Tensor d 4 CFloat, Tensor d 1 CFloat,
  Tensor d 4 CFloat, Tensor d 1 CFloat,
  Tensor d 4 CFloat, Tensor d 1 CFloat,
  Tensor d 4 CFloat, Tensor d 1 CFloat] 

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
      batch_size = 128
      batch_shape = batch_size:.1:.28:.28:.Z :: Dim 4
      labels_shape = batch_size:.nb_labels:.Z :: Dim 2
      dev1 = Proxy :: Proxy 1
      dev2 = Proxy :: Proxy 2
  putStrLn "Loading training set..."
  emnist_train <- runExceptT $ P.toListM
                  $ load_mnist (Just train_images_url) (Just train_labels_url)
                  train_images_file train_labels_file
  putStrLn "Loading validation set..."
  emnist_val <- runExceptT $ P.toListM
                $ load_mnist (Just test_images_url) (Just test_labels_url)
                test_images_file test_labels_file
  let gen = generator 42 :: Generator
  case pure (,) <*> emnist_train <*> emnist_val of
   Left err -> ioError $ userError $ "Error loading the dataset: " ++ err
   Right (mnist_train,mnist_val) -> do
     putStrLn "Starting training..."
     (merr,_) <- runCudaT gen $ do
       putStrLn "Initializing weights..."
       w_0 <- init_weights nb_labels :: CudaT IO (Weights CFloat (MNISTWeights 1))
       let
         nb_val_batches = length mnist_val `div` batch_size
         preprocessing :: Pipe
                          (Grey, Int)
                          (SVector CFloat, SVector CFloat)
                          (CudaT IO) ()
         preprocessing =
           P.map (\(i,l) -> (i, [l]))
           >-> batch_images nb_labels batch_size
           >-> P.map (\(b,l) -> (VS.map grey_to_float b, l))
         validate (c,w) = do
           let sampleAccuracy :: Pipe (Tensor 1 4 CFloat, Tensor 1 2 CFloat) ([Int],[Int]) (CudaT IO) ()
               sampleAccuracy = forever $ do
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
                          >-> batch_to_gpu dev1 batch_shape labels_shape
                          >-> sampleAccuracy
           putStrLn $ "Validation accuracy: " ++ pack (show valAccuracy)
         optimize =
           sgd
           (rmsprop 0.001 0.9 0.1)
           (cost_grad (Proxy :: Proxy 1) (Proxy :: Proxy 2) batch_size nb_labels)
           w_0
       putStrLn "Launching sgd..."
       runEffect
         $ forever (randomize mnist_train)
         >-> preprocessing
         >-> batch_to_multi_gpus dev1 dev2 batch_shape labels_shape
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

cudaToTraining :: Cuda a -> (CudaT IO) a
cudaToTraining = cudaHoist generalize

model :: (Device d) => Int -> Int -> Layer (Cuda) CFloat (MNISTWeights d) (Tensor d 4 CFloat) (Tensor d 2 CFloat)
model batch_size nb_labels =
  let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
             >+> bias
             >+> activation activation_relu
      conv_1 = convolution2d (0,0) (1,1) convolution_fwd_algo_implicit_gemm
               >+> bias
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
  >+> dropout 0.5
  >+> conv_1
  >+> dropout 0.5
  >+> conv_1
  >+> lreshape (batch_size:.nb_labels:.Z)

nnet :: (Device d)
     => Proxy d -> Int -> Int
     -> Layer (Cuda) CFloat (MNISTWeights d) (Tensor d 2 CFloat, Tensor d 4 CFloat) CFloat
nnet _ batch_size nb_labels =
  let criteria = mlrCost (batch_size:.nb_labels:.Z) >+> toScalar in
   (id' *** model batch_size nb_labels) >+> criteria

cost_grad :: (Device d1, Device d2) => Proxy d1 -> Proxy d2 -> Int -> Int
          -> Weights CFloat (MNISTWeights d1)
          -> ((Tensor d1 4 CFloat, Tensor d1 2 CFloat),
              (Tensor d2 4 CFloat, Tensor d2 2 CFloat))
          -> CudaT IO (CFloat, Weights CFloat (MNISTWeights d1))
cost_grad p1 p2 batch_size nb_labels w_t ((b1,l1),(b2,l2)) = do
  let net1 = nnet p1 batch_size nb_labels 
      net2 = nnet p2 batch_size nb_labels
      fullNet = (dataPar p1 net1 net2) >+> add >+> scale -< 0.5 
  (cost, bwd) <- cudaToTraining
                 $ forwardBackward fullNet w_t ((l1,b1),(l2,b2))
  let (w', _) = bwd (1 :: CFloat)
  return (cost, w')

predict :: (Device d) => Int -> Int -> Weights CFloat (MNISTWeights d) -> Tensor d 4 CFloat -> CudaT IO (Tensor d 2 CFloat)
predict batch_size nb_labels w_t batch = do
  cudaToTraining $ forward (model batch_size nb_labels) w_t batch

he_init :: (Device d) => Dim 4 -> CudaT IO (Tensor d 4 CFloat)
he_init s@(_:.c:.fh:.fw:.Z) =
  normal s 0 (sqrt (2 / (fromIntegral c * fromIntegral fh * fromIntegral fw)))

init_weights :: (Device d) => Int -> CudaT IO (Weights CFloat (MNISTWeights d))
init_weights nb_labels = do
  w_1 <- he_init (32:.1:.3:.3:.Z)
  b_1 <- zeros (32:.Z)
  w_2 <- he_init (64:.32:.3:.3:.Z)
  b_2 <- zeros (64:.Z)
  w_3 <- he_init (128:.64:.3:.3:.Z)
  b_3 <- zeros (128:.Z)
  w_4 <- he_init (256:.128:.3:.3:.Z)
  b_4 <- zeros (256:.Z)
  w_5 <- he_init (256:.256:.1:.1:.Z)
  b_5 <- zeros (256:.Z)
  w_6 <- he_init (10:.256:.1:.1:.Z)
  b_6 <- zeros (10:.Z)
  return $ W $ w_1:.b_1:.w_2:.b_2:.w_3:.b_3:.w_4:.b_4:.w_5:.b_5:.w_6:.b_6:.Z

grey_to_float :: Word8 -> CFloat
grey_to_float i = (fromIntegral i - 128) / 255

-- Printing the current optimization state.
data InfoState = InfoState {
  rolling_cost :: Maybe CFloat
  }

print_info :: Device d => Consumer (CFloat, Weights CFloat (MNISTWeights d)) (CudaT IO) ()
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
