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

type Device1 = 1
type Device2 = 2
type ConvWeights d a = SizedList' 3 (Tensor d (Dim 4) a)

conv :: (Device d, TensorScalar a, MonadCuda m)
     => Int -> Int -> Int
     -> InitLayer m a (ConvWeights d a) (Tensor d (Dim 4) a) (Tensor d (Dim 4) a)
conv inp out sz =
  convLayer >+> batchNorm >+> relu
  where
    convLayer =
      init
      (convolution2d (sz `div` 2, sz `div` 2) (1,1)
       convolution_fwd_algo_implicit_gemm
       convolution_bwd_data_algo_0
       convolution_bwd_filter_algo_0)
      (he_init (out:.inp:.sz:.sz:.Z))
    batchNorm =
      init (batchNormalization batchnorm_spatial 10E-5)
      (pure (<+>) <*> ones_w (1:.out:.1:.1:.Z) <*>  zeros_w (1:.out:.1:.1:.Z))
    relu = init' $ activation activation_relu

pool :: (Device d, MonadCuda m, TensorScalar a)
     => Int -> PoolingMode
     -> InitLayer m a '[] (Tensor d (Dim 4) a) (Tensor d (Dim 4) a)
pool sz mode = init' $ pooling2d (sz,sz) (0,0) (sz,sz) mode

type MNISTWeights d a = SizedList' 18 (Tensor d (Dim 4) a)

initModel :: (Device d, MonadCuda m, TensorScalar a)
          => Int -> Int
          -> InitLayer m a (MNISTWeights d a) (Tensor d (Dim 4) a) (Tensor d (Dim 2) a)
initModel batch_size nb_labels =
        conv 1 32 3
        >+> pool 2 pooling_max -- 14*14
        >+> conv 32 64 3
        >+> pool 2 pooling_max -- 7*7
        >+> conv 64 128 3
        >+> pool 2 pooling_max-- 3*3
        >+> conv 128 256 3
        >+> pool 3 pooling_average_count_include_padding -- 1*1
        >+> init' (dropout 0.5)
        >+> conv 256 512 1
        >+> init' (dropout 0.5)
        >+> conv 512 nb_labels 1
        >+> init' (lreshape (batch_size:.nb_labels:.Z))

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
      dev1 = Proxy :: Proxy Device1
      dev2 = Proxy :: Proxy Device2
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
       let model = initModel batch_size nb_labels
           trainModel =
             let baseNet = (id' *** layer model)
                           >+> mlrCost (batch_size:.nb_labels:.Z)
                           >+> toScalar
             in dataPar dev1 baseNet baseNet >+> add_ >+> scaleByCst_ 0.5
           cost_grad w ((b1,l1),(b2,l2)) = do
             ~(cost, bwd) <- forwardBackward trainModel w ((l1,b1),(l2,b2))
             return (cost, fst $ bwd (1 :: CFloat))
           predict w b = forward (layer model) w b
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
             let sampleAccuracy :: Pipe (Tensor 1 (Dim 4) CFloat, Tensor 1 (Dim 2) CFloat) ([Int],[Int]) (CudaT IO) ()
                 sampleAccuracy = forever $ do
                   (b,l) <- await
                   confmat <- lift $ predict w b
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
       w_0 <- initWeights model
       putStrLn "Launching sgd..."
       runEffect
         $ forever (randomize mnist_train)
         >-> preprocessing
         >-> batch_to_multi_gpus dev1 dev2 batch_shape labels_shape
         >-> sgd (momentum 0.05 0.9) cost_grad w_0
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

he_init :: (Device d, TensorScalar a, MonadCuda m)
        => Dim 4 -> m (Weights a '[Tensor d (Dim 4) a])
he_init s@(_:.c:.fh:.fw:.Z) = do
  t <- normal s 0 (sqrt (2 / (fromIntegral c * fromIntegral fh * fromIntegral fw)))
  return $ W $ t:.Z

zeros_w :: (Device d, TensorScalar a, MonadCuda m, Shape s)
        => s -> m (Weights a '[Tensor d s a])
zeros_w = fmap (\t -> W $ t:.Z) . zeros

ones_w :: (Device d, TensorScalar a, MonadCuda m, Shape s)
       => s -> m (Weights a '[Tensor d s a])
ones_w = fmap (\t -> W $ t:.Z) . ones

grey_to_float :: Word8 -> CFloat
grey_to_float i = (fromIntegral i - 128) / 255

-- Printing the current optimization state.
data InfoState = InfoState {
  rolling_cost :: Maybe CFloat
  }

print_info :: Consumer (CFloat, w) (CudaT IO) ()
print_info = flip evalStateT (InfoState Nothing) $ forM_ [1..] $ \i -> do
  (cost, _) <- lift $ await
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
