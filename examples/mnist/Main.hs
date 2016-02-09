{-# LANGUAGE FlexibleContexts, DataKinds, TypeFamilies, RankNTypes #-}
module Main where

import qualified Data.Vector.Storable as V
import Data.HList.HList
import Data.Maybe
import Data.Word
import DeepBanana hiding (load_mnist)
import DeepBanana.Layer.CUDA
import Control.DeepSeq
import Control.Monad
import Control.Monad.Except
import Control.Monad.Identity
import Control.Monad.Morph
import Control.Monad.State
import Control.Monad.Trans
import MNIST
import Pipes
import qualified Pipes.Prelude as P
import Prelude hiding ((.), id)
import System.FilePath
import System.Mem
import Vision.Image
import Vision.Image.Storage.DevIL

type Weights = '[Tensor 4 CFloat, Tensor 4 CFloat, Tensor 4 CFloat, Tensor 4 CFloat] 

type TrainingT m = VanillaT CFloat (CUDAT m)

type Training = TrainingT Identity

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
  case pure (,) <*> emnist_train <*> emnist_val of
   Left err -> ioError $ userError $ "Error loading the dataset: " ++ err
   Right (mnist_train,mnist_val) -> do
     merr <- runCUDAT 42 $ runVanilla 0.01 $ do
       w_0 <- lift $ init_weights nb_labels
       let
         nb_val_batches = fromIntegral $ (length mnist_val `div` batch_size) :: CFloat
         preprocessing =
           P.map (\(i,l) -> (i, [l]))
           >-> batch_images nb_labels batch_size
           >-> P.map (\(b,l) -> (V.map grey_to_float b, l))
           >-> batch_to_gpu (batch_size:.1:.28:.28:.Z) (batch_size:.nb_labels:.Z)
           :: Pipe (Grey, Int) (Tensor 4 CFloat, Tensor 2 CFloat) (TrainingT IO) ()
         cost_grad' w b = hoistBase generalize $ cost_grad batch_size nb_labels w b
         validate (_,w_t) = do
           liftIO $ putStrLn "Computing validation cost..."
           sum_cost <- P.sum $ forM_ mnist_val yield
                       >-> preprocessing
                       >-> P.mapM (fmap fst . cost_grad' w_t)
           liftIO $ putStrLn $ "Validation cost: " ++ show (sum_cost / nb_val_batches)
         optimize = sgd vanilla cost_grad' w_0 :: Pipe (Tensor 4 CFloat, Tensor 2 CFloat) (CFloat, HLSpace CFloat Weights) (TrainingT IO) ()
       runEffect
         $ forever (randomize mnist_train)
         >-> preprocessing
         >-> optimize
         >-> runEvery 100 (\x -> deepseq x $ liftIO $ performGC)
         >-> runEvery 1000 validate
         >-> print_info
     case merr of
      Left err -> ioError $ userError $ "Error during training: " ++ err
      Right _ -> return ()

hoistBase :: (Monad m, Monad n) => (forall a . m a -> n a) -> TrainingT m a -> TrainingT n a
hoistBase morph = hoist (hoist (hoist morph))

nnet :: Int -> Int
     -> Layer Training CFloat Weights (Tensor 2 CFloat, Tensor 4 CFloat) CFloat
nnet batch_size nb_labels =
  let conv = convolution2d (1,1) (1,1) convolution_fwd_algo_implicit_gemm
             >+> activation activation_relu
      pool = pooling2d (2,2) (1,1) (2,2) pooling_max
      pool_avg = pooling2d (3,3) (1,1) (3,3) pooling_average_count_include_padding
      model = conv
              >+> pool -- 14*14
              >+> conv
              >+> pool -- 7*7
              >+> conv
              >+> pool -- 3*3
              >+> conv
              >+> pool_avg -- 1*1
              >+> lreshape (batch_size:.nb_labels:.Z)
      criteria = mlrCost (batch_size:.nb_labels:.Z) >+> toScalar in
   (id' *** model) >+> criteria
  
cost_grad :: Int -> Int -> HLSpace CFloat Weights
          -> (Tensor 4 CFloat, Tensor 2 CFloat)
          -> Training (CFloat, HLSpace CFloat Weights)
cost_grad batch_size nb_labels w_t (batch,labels) = do
  (cost, bwd) <- forwardBackward (nnet batch_size nb_labels) w_t (labels,batch)
  let (w', _) = bwd (1 :: CFloat)
  return (cost, w')

init_weights :: (MonadState Generator m) => Int -> m (HLSpace CFloat Weights)
init_weights nb_labels = do
  let he_init s@(_:.c:.fh:.fw:.Z) =
        normal s 0 (sqrt (2 / (fromIntegral c * fromIntegral fh * fromIntegral fw)))
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

print_info :: Consumer (CFloat, HLSpace CFloat Weights) (TrainingT IO) ()
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
    liftIO $ putStrLn $ "Iteration " ++ show i
      ++ " cost rolling average: " ++ show roll_cost
