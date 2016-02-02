module DeepBanana.Data (
    load_mnist
  , load_mnist_lazy
  , lazy_image_loader  
  , randomize
  , map_pixels
  , batch_images
  , batch_to_gpu
  , runEvery
  , serializeTo
  , random_crop
  , resize_min_dim
  ) where

import Foreign.C
import System.IO
import System.FilePath
import Vision.Image as Friday hiding (read)
import Vision.Primitive hiding (Shape, Z, (:.))
import qualified Vision.Primitive as VP
import Vision.Image.Storage.DevIL
import Pipes hiding (Proxy)
import System.Directory
import Data.Maybe
import Control.Monad
import Control.Monad.Except
import Control.Monad.ST
import Control.Monad.Random
import Foreign.Storable
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as MV
import qualified Data.Vector as NSV
import qualified Data.Vector.Mutable as NSVM
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as BI
import Data.Serialize
import Data.Proxy
import qualified Data.Time as Time
import Data.IORef
import Control.DeepSeq
import Foreign.ForeignPtr
import Control.Applicative

import DeepBanana.Tensor as T

vecToBstring :: forall a . (Storable a) => V.Vector a -> B.ByteString
vecToBstring vec =
  let (afptr, o, l) = V.unsafeToForeignPtr vec
      asize = sizeOf (undefined :: a)
      bo = o * asize
      bl = l * asize
  in BI.fromForeignPtr (castForeignPtr afptr) bo bl      

bstringToVec :: forall a . (Storable a) => B.ByteString -> Maybe (V.Vector a)
bstringToVec bstr =
  let (bfptr, bo, bl) = BI.toForeignPtr bstr
      asize = sizeOf (undefined :: a)
      o = if bo `rem` asize == 0 then Just (bo `div` asize) else Nothing
      l = if bl `rem` asize == 0 then Just (bl `div` asize) else Nothing
  in pure (V.unsafeFromForeignPtr (castForeignPtr bfptr)) <*> o <*> l

-- Serializing for Friday image.
instance (Storable i)
         => Serialize (Manifest i) where
  put (Manifest (VP.Z VP.:. w VP.:. h) vec) = do
    put w
    put h
    put $ vecToBstring vec
  get = do
    w <- get
    h <- get
    bstr <- get
    case bstringToVec bstr of
     Nothing -> error "Could not convert back to an image to data alignment issues."
     Just vec -> return (Manifest (VP.Z VP.:. w VP.:. h) vec)

load_mnist :: (Convertible StorageImage i)
           => FilePath -> IO [(i, Int)]
load_mnist directory = do
  dirs <- getDirectoryContents directory >>= filterM (\d -> doesDirectoryExist $ directory </> d)
  imgs <- forM dirs $ \classname -> do
    imgfpaths <- getDirectoryContents (directory </> classname) 
    imgs' <- forM imgfpaths $ \imgfpath -> do
        eimg <-load Autodetect $ directory </> classname </> imgfpath
        case eimg of
          Left err -> return Nothing
          Right img -> return $ Just (img, read classname)
    return imgs'
  return $ fmap fromJust $ filter (\mi -> case mi of
                                      Nothing -> False
                                      _ -> True) $ Prelude.concat $ imgs

load_mnist_lazy :: (MonadIO m, Convertible StorageImage i)
                => FilePath -> Producer (i, Int) m ()
load_mnist_lazy directory = do
  dirs <- liftIO $ getDirectoryContents directory >>= filterM (\d -> doesDirectoryExist $ directory </> d)
  forM_ dirs $ \classname -> do
    imgfpaths <- liftIO $ getDirectoryContents (directory </> classname) 
    forM_ imgfpaths $ \imgfpath -> do
      eimg <- liftIO $ load Autodetect $ directory </> classname </> imgfpath
      case eimg of
       Left err -> do
         liftIO $ do
           putStrLn $ "Couldn't load image " ++ imgfpath
           putStrLn $ "\t" ++ show err
         return ()
       Right img -> yield (img, read classname)

while :: (Monad m) => (m Bool) -> m b -> m [b]
while pred action = do
  continue <- pred
  if continue then do
    x <- action
    xs <- while pred action
    return (x:xs)
  else return []

lazy_image_loader :: forall i m l . (Image i, Convertible StorageImage i, Storable (ImagePixel i), MonadIO m)
                  => Proxy i -> FilePath -> Pipe (FilePath, l) (Manifest (ImagePixel i), l) m ()
lazy_image_loader _ directory = forever $ do
  (fpath, labels) <- await
  eimg <- liftIO $
          (load Autodetect $ directory </> fpath
           :: IO (Either StorageError i))
  case eimg of
   Left err -> liftIO $ do
     putStrLn $ "Unable to load image " ++ fpath
     putStrLn $ show err
   Right img -> do
     let VP.Z VP.:. h VP.:. w = Friday.shape img
     if  h == 1  || w == 1
      then liftIO $ putStrLn $ "Image loaded as 1 by 1, skipping: " ++ fpath
      else do
       imgRes <- computeP img
       yield (imgRes, labels)
       return ()

shuffle :: MonadRandom m => [a] -> m [a]
shuffle xs = do
  let l = length xs
  rands_ <- getRandomRs (0, l-1)
  let
    rands = take l rands_
    ar = runST $ do
      ar <- NSV.unsafeThaw $ NSV.fromList xs
      forM_ (zip [0..(l-1)] rands) $ \(i, j) -> do
        vi <- NSVM.read ar i
        vj <- NSVM.read ar j
        NSVM.write ar j vi
        NSVM.write ar i vj
      NSV.unsafeFreeze ar
  return $ NSV.toList ar

randomize :: (MonadRandom m) => [a] -> Producer a m ()
randomize xs = do
  sxs <- lift $ shuffle xs
  forM_ sxs yield

-- Converts an image to a storable-based vector by casting and sharing the inner
-- data foreign pointer.
img_to_vec :: (Image i, Pixel (ImagePixel i), Storable (PixelChannel (ImagePixel i)))
           => i -> V.Vector (PixelChannel (ImagePixel i))
img_to_vec img =
  let Manifest (VP.Z VP.:. h VP.:. w) pixVec = compute img
      c = nChannels img
      (pixFptr, o, l) = V.unsafeToForeignPtr pixVec
  in V.unsafeFromForeignPtr (castForeignPtr pixFptr) (c*o) (c*l)

-- applies a pixel-wise operation to all images.
map_pixels :: (FunctorImage src dst, Monad m)
           => (ImagePixel src -> ImagePixel dst) -> Pipe src dst m ()
map_pixels f = forever $ await >>= yield . Friday.map f

samesize_concat :: (Storable a) => [V.Vector a] -> V.Vector a
samesize_concat vectors = runST $ do
  let size = V.length $ head vectors
      nbVectors = length vectors
      totalSize = nbVectors * size
  mres <- MV.new totalSize
  forM_ (zip [0..] vectors) $ \(i,v) -> do
    let mresslice = MV.unsafeSlice (i * size) size mres
    mv <- V.unsafeThaw v
    MV.unsafeCopy mresslice mv
  V.unsafeFreeze mres

batch_images :: (Image i, Storable (PixelChannel (ImagePixel i)),
                 Pixel (ImagePixel i), Monad m, TensorScalar a)
             => Int
             -> Int
             -> Pipe (i, [Int]) (V.Vector (PixelChannel (ImagePixel i)), V.Vector a) m ()
batch_images nb_labels batch_size = forever $ do
  imgAndLabels <- replicateM batch_size await
  let (images, labels) = unzip imgAndLabels
      VP.Z VP.:. h VP.:. w = Friday.shape $ head images
      c = nChannels $ head images
      oneHot ls = V.fromList [if i `elem` ls then 1 else 0 | i <- [0..nb_labels - 1]]
      imgVecs = fmap img_to_vec $ images
      batch = samesize_concat imgVecs
      lmatrix = V.concat $ fmap oneHot labels
  yield (batch, lmatrix)

batch_to_gpu :: (MonadIO m, MonadError String m, TensorScalar a,
                 Shape (Dim n1), Shape (Dim n2))
             => Dim n1
             -> Dim n2
             -> Pipe (V.Vector a, V.Vector a) (Tensor n1 a, Tensor n2 a) m ()
batch_to_gpu shp1 shp2 = forever $ do
  (batch, labels) <- await
  tbatch <- fromVector shp1 batch
  tlabels <- fromVector shp2 labels
  yield (tbatch, tlabels)

-- serializes inputs
runEvery :: (Monad m) => Int -> (a -> m ()) -> Pipe a a m c
runEvery n action = forever $ do
  replicateM (n - 1) $ await >>= yield
  x <- await
  lift $ action x
  yield x

serializeTo :: (MonadIO m, Serialize a) => FilePath -> a -> m ()
serializeTo fpath toSerialize = do
  liftIO $ B.writeFile fpath $ encode toSerialize

-- Dataset transformations
random_crop :: forall i l m . (Image i, Storable (ImagePixel i), MonadRandom m)
            => Proxy i -> Int -> Int -> Pipe (Manifest (ImagePixel i), l) (Manifest (ImagePixel i), l) m ()
random_crop _ width height = forever $ do
  (img,l) <- await
  let VP.Z VP.:. imgHeight VP.:. imgWidth = Friday.shape img
  if (imgWidth < width || imgHeight < height)
   then error $ "Image too small for cropping:\nimage size: " ++ show (imgHeight,imgWidth) ++ "\ncrop size: " ++ show (height,width)
   else do
    ix <- lift $ getRandomR (0,imgWidth-width)
    iy <- lift $ getRandomR (0,imgHeight-height)
    let croppedImg = crop (Rect ix iy width height) img :: Manifest (ImagePixel i)
    cropRes <- computeP croppedImg
    yield (cropRes, l)

resize_min_dim :: forall i l m
               . (Image i, Storable (ImagePixel i), MonadRandom m,
                  Integral (ImageChannel i), Interpolable (ImagePixel i))
               => Proxy i -> Int -> Pipe (Manifest (ImagePixel i), l) (Manifest (ImagePixel i), l) m ()
resize_min_dim _ min_dim = forever $ do
  (img,l) <- await
  let VP.Z VP.:. imgHeight VP.:. imgWidth = Friday.shape img
      size = if imgHeight > imgWidth
             then ix2 (round (realToFrac (imgHeight * min_dim) / realToFrac imgWidth :: Double)) min_dim
             else ix2 min_dim (round (realToFrac (imgWidth * min_dim) / realToFrac imgHeight :: Double))
      resizedImg = resize TruncateInteger size img :: Manifest (ImagePixel i)
  resizedImgRes <- computeP resizedImg
  yield (resizedImgRes, l)
