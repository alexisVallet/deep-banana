{-# LANGUAGE TypeFamilies #-}
module DeepBanana.Data (
    module DeepBanana.Data.Exception
  , lazy_image_loader  
  , randomize
  , randomLabelSubset
  , randomize_seq_length
  , map_pixels
  , batch_images
  , batch_images_pad_labels
  , batch_to_gpu
  , batch_labels_to_gpu
  , runEvery
  , serializeTo
  , random_crop
  , resize_min_dim
  , accuracy
  , precision
  , recall
  ) where

import Control.Monad.Random
import Data.List (unfoldr, transpose)
import qualified Data.List as L
import qualified Data.ByteString.Internal as BI
import qualified Data.Serialize as Serialize
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Data.Vector.Storable.Mutable as MSV
import qualified Data.Vector.Mutable as NSVM
import Foreign.ForeignPtr
import qualified Pipes.Prelude as P
import Vision.Image as Friday hiding (read)
import Vision.Image.Storage.DevIL
import Vision.Primitive hiding (Shape, Z, (:.))
import qualified Vision.Primitive as VP

import DeepBanana.Data.Exception
import DeepBanana.Device
import DeepBanana.Exception
import DeepBanana.Prelude
import DeepBanana.Tensor as T
import DeepBanana.Tensor.Exception

vecToBstring :: forall a . (Storable a) => SVector a -> ByteString
vecToBstring vec =
  let (afptr, o, l) = SV.unsafeToForeignPtr vec
      asize = mySizeOf (Proxy :: Proxy a)
      bo = o * asize
      bl = l * asize
  in BI.fromForeignPtr (castForeignPtr afptr) bo bl      

bstringToVec :: forall a . (Storable a) => ByteString -> Maybe (SVector a)
bstringToVec bstr =
  let (bfptr, bo, bl) = BI.toForeignPtr bstr
      asize = mySizeOf (Proxy :: Proxy a)
      o = if bo `rem` asize == 0 then Just (bo `div` asize) else Nothing
      l = if bl `rem` asize == 0 then Just (bl `div` asize) else Nothing
  in pure (SV.unsafeFromForeignPtr (castForeignPtr bfptr)) <*> o <*> l

-- Serializing for Friday image.
instance (Storable i)
         => Serialize.Serialize (Manifest i) where
  put (Manifest (VP.Z VP.:. w VP.:. h) vec) = do
    Serialize.put w
    Serialize.put h
    Serialize.put $ vecToBstring vec
  get = do
    w <- Serialize.get
    h <- Serialize.get
    bstr <- Serialize.get
    case bstringToVec bstr of
     Nothing -> error "Could not convert back to an image to data alignment issues."
     Just vec -> return (Manifest (VP.Z VP.:. w VP.:. h) vec)

lazy_image_loader :: forall i m l
                  . (Image i, Convertible StorageImage i, Storable (ImagePixel i),
                     MonadIO m)
                  => Proxy i -> FilePath
                  -> Pipe (FilePath, l) (Manifest (ImagePixel i), l) m ()
lazy_image_loader _ directory = forever $ do
  (fpath, labels) <- await
  eimg <- liftIO $
          (load Autodetect $ directory </> fpath
           :: IO (Either StorageError i))
  case eimg of
   Left err -> liftIO $ do
     print $ "Unable to load image " ++ fpath
     print $ err
   Right img -> do
     let VP.Z VP.:. h VP.:. w = Friday.shape img
     if  h == 1  || w == 1
      then liftIO $ print $ "Image loaded as 1 by 1, skipping: " ++ fpath
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
      ar <- V.unsafeThaw $ V.fromList xs
      forM_ (zip [0..(l-1)] rands) $ \(i, j) -> do
        vi <- NSVM.read ar i
        vj <- NSVM.read ar j
        NSVM.write ar j vi
        NSVM.write ar i vj
      V.unsafeFreeze ar
  return $ V.toList ar

randomLabelSubset :: (MonadRandom m) => Int -> Pipe (a, [l]) (a, [l]) m ()
randomLabelSubset subsetSize = forever $ do
  (s, labels) <- await
  rlabels <- lift $ shuffle labels
  yield (s, take subsetSize labels)

randomize :: (MonadRandom m) => [a] -> Producer a m ()
randomize xs = do
  sxs <- lift $ shuffle xs
  forM_ sxs yield

randomize_seq_length :: (MonadRandom m) => Int -> Int -> Pipe (a,[b]) (a,[b]) m ()
randomize_seq_length nb_batches batch_size = forever $ do
  macro_batch <- replicateM (nb_batches * batch_size) await
  let sorted_batch = sortOn (snd >>> length) macro_batch
      mini_batches = unfoldr (\xs -> case xs of
                          [] -> Nothing
                          _ -> Just $ splitAt batch_size xs) sorted_batch
  random_batches <- lift $ shuffle mini_batches
  forM_ random_batches $ \batch -> do
    random_batch <- lift $ shuffle batch
    forM_ random_batch yield

-- Converts an image to a storable-based vector by casting and sharing the inner
-- data foreign pointer.
img_to_vec :: (Image i, Pixel (ImagePixel i), Storable (PixelChannel (ImagePixel i)))
           => i -> SVector (PixelChannel (ImagePixel i))
img_to_vec img =
  let Manifest (VP.Z VP.:. h VP.:. w) pixVec = compute img
      c = nChannels img
      (pixFptr, o, l) = SV.unsafeToForeignPtr pixVec
  in SV.unsafeFromForeignPtr (castForeignPtr pixFptr) (c*o) (c*l)

-- applies a pixel-wise operation to all images.
map_pixels :: (FunctorImage src dst, Monad m)
           => (ImagePixel src -> ImagePixel dst) -> Pipe src dst m ()
map_pixels f = forever $ await >>= yield . Friday.map f

samesize_concat :: (Storable a) => [SVector a] -> SVector a
samesize_concat vectors = runST $ do
  let size = SV.length $ unsafeHead vectors
      nbVectors = length vectors
      totalSize = nbVectors * size
  mres <- MSV.new totalSize
  forM_ (zip [0..] vectors) $ \(i,v) -> do
    let mresslice = MSV.unsafeSlice (i * size) size mres
    mv <- SV.unsafeThaw v
    MSV.unsafeCopy mresslice mv
  SV.unsafeFreeze mres

-- Batches images, and multiple labels in a single label matrix L s.t.
-- L_i,j = 1 if i has label h else 0
batch_images :: (Image i, Storable (PixelChannel (ImagePixel i)),
                 Pixel (ImagePixel i), Monad m, TensorScalar a, MonadError t m,
                 Variant t EmptyBatch)
             => Int
             -> Int
             -> Pipe (i, [Int]) (SVector (PixelChannel (ImagePixel i)), SVector a) m ()
batch_images nb_labels batch_size = do
  when (batch_size <= 0) $ throwVariant
    $ emptyBatch "batch_images: cannot build batch with size smaller than 1!"
  forever $ do
    imgAndLabels <- replicateM batch_size await
    let (images, labels) = unzip imgAndLabels
        VP.Z VP.:. h VP.:. w = Friday.shape $ unsafeHead images
        c = nChannels $ unsafeHead images
        oneHot ls = SV.fromList [if i `elem` ls then 1 else 0 | i <- [0..nb_labels - 1]]
        imgVecs = fmap img_to_vec $ images
        batch = samesize_concat imgVecs
        lmatrix = SV.concat $ fmap oneHot labels
    yield (batch, lmatrix)

-- Batches images, and multiple labels as a batched sequences of labels.
-- The output sequence will have the length of the longest sequence in the
-- batch, and shorter sequences will be padded with all 0 vectors. Otherwise,
-- uses one-hot encoding.
batch_images_pad_labels :: forall i m t a
                        . (Image i, Storable (PixelChannel (ImagePixel i)),
                           Pixel (ImagePixel i), Monad m, TensorScalar a,
                           MonadError t m, Variant t EmptyBatch)
                        => Int
                        -> Int
                        -> Int
                        -> Pipe (i, [Int]) (SVector (PixelChannel (ImagePixel i)), [SVector a]) m ()
batch_images_pad_labels nb_labels batch_size pad_label = do
  when (batch_size <= 0) $ throwVariant
    $ emptyBatch "batch_images_pad_labels: cannot build batch with size smaller than 1!"
  forever $ do
    imgAndLabels <- fmap asList $ replicateM batch_size await
    let (images, labels) = unzip imgAndLabels
        VP.Z VP.:. h VP.:. w = Friday.shape $ unsafeHead images
        c = nChannels $ unsafeHead images
        imgVecs = fmap img_to_vec $ images
        batch = samesize_concat imgVecs
        longest_seq_len = maximumEx $ fmap length labels
        padded_labels = fmap (\l -> fmap Just l ++ take (longest_seq_len - length l) (repeat Nothing)) labels
        oneHot mi = runST $ do
          mres <- MSV.replicate nb_labels 0
          case mi of
           Nothing -> do
             MSV.write mres pad_label 1
             SV.unsafeFreeze mres
           Just i -> do
             MSV.write mres i 1
             SV.unsafeFreeze mres
        onehot_labels = fmap (\ls -> samesize_concat $ fmap oneHot ls)
                        $ transpose padded_labels
    yield (batch, onehot_labels)


batch_to_gpu :: (MonadIO m, MonadError t m, Variant t OutOfMemory,
                 Variant t IncompatibleSize, TensorScalar a,
                 Shape (Dim n1), Shape (Dim n2), Device d1, Device d2)
             => Dim n1
             -> Dim n2
             -> Pipe (SVector a, SVector a) (Tensor d1 n1 a, Tensor d2 n2 a) m ()
batch_to_gpu shp1 shp2 = forever $ do
  (batch, labels) <- await
  tbatch <- fromVector shp1 batch
  tlabels <- fromVector shp2 labels
  yield (tbatch, tlabels)

batch_labels_to_gpu :: (MonadIO m, MonadError t m, Variant t OutOfMemory,
                        Variant t IncompatibleSize, TensorScalar a,
                        Shape (Dim n1), Shape (Dim n2), Device d1, Device d2)
                    => Dim n1
                    -> Dim n2
                    -> Pipe (SVector a, [SVector a]) (Tensor d1 n1 a, [Tensor d2 n2 a]) m ()
batch_labels_to_gpu shp1 shp2 = forever $ do
  (batch, labels) <- await
  tbatch <- fromVector shp1 batch
  tlabels <- forM labels $ \ls -> fromVector shp2 ls
  yield (tbatch, tlabels)

-- serializes inputs
runEvery :: (Monad m) => Int -> (a -> m ()) -> Pipe a a m c
runEvery n action = forever $ do
  fmap asList $ replicateM (n - 1) $ await >>= yield
  x <- await
  lift $ action x
  yield x

serializeTo :: (MonadIO m, Serialize.Serialize a) => FilePath -> a -> m ()
serializeTo fpath toSerialize = do
  liftIO $ writeFile fpath $ Serialize.encode toSerialize

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

accuracy :: (Eq a, Floating b, Monad m) => Producer ([a],[a]) m () -> m b
accuracy predGtProd = do
  let sampleAcc (pred,gt) =
        fromIntegral (length (L.intersect pred gt)) / fromIntegral (length (L.union pred gt))
  acc <- P.sum $ predGtProd >-> P.map sampleAcc
  len <- P.length predGtProd
  return $ acc / fromIntegral len

precision :: (Eq a, Floating b, Monad m) => Producer ([a],[a]) m () -> m b
precision predGtProd = do
  let samplePrec (pred,gt) =
        fromIntegral (length (L.intersect pred gt)) / fromIntegral (length pred)
  prec <- P.sum $ predGtProd >-> P.map samplePrec
  len <- P.length predGtProd
  return $ prec / fromIntegral len

recall :: (Eq a, Floating b, Monad m) => Producer ([a],[a]) m () -> m b
recall predGtProd = do
  let sampleRec (pred,gt) =
        fromIntegral (length (L.intersect pred gt)) / fromIntegral (length gt)
  recall <- P.sum $ predGtProd >-> P.map sampleRec
  len <- P.length predGtProd
  return $ recall / fromIntegral len
