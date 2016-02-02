{-# LANGUAGE OverloadedStrings, FlexibleContexts #-}
module MNIST where

import Data.Binary.Get (runGet, getWord32be)
import DeepBanana
import Codec.Compression.GZip
import Control.Lens
import Control.Monad
import Control.Monad.Trans
import Control.Monad.Except
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import qualified Data.Attoparsec.ByteString as AP
import Network.Wreq
import Pipes
import Prelude hiding ((.), id)
import System.Directory
import System.FilePath
import Vision.Image as VI
import qualified Vision.Primitive as VP

loadOrDownload :: (MonadIO m, MonadError String m)
               => Maybe String -> FilePath -> m BS.ByteString
loadOrDownload murl fpath = do
  dirExists <- liftIO $ doesDirectoryExist $ takeDirectory fpath
  when (not dirExists) $ throwError $
    "Directory does not exist: " ++ show (takeDirectory fpath)
  fileExists <- liftIO $ doesFileExist fpath
  case (fileExists, murl) of
   (False, Nothing) -> throwError
                       $ "File not found and no download URL provided: " ++ fpath
   (False, Just url) -> liftIO $ do
     putStrLn $ "File not found: " ++ fpath ++ "\nDownloading from: " ++ url
     resp <- get url
     LBS.writeFile fpath $ resp ^. responseBody
     putStrLn $ "Finished downloading."
     return $ LBS.toStrict $ decompress $ resp ^. responseBody
   _ -> liftIO $ fmap (LBS.toStrict . decompress) $ LBS.readFile fpath

parseInt :: AP.Parser Int
parseInt = fmap (fromIntegral . runGet getWord32be . LBS.fromStrict) $ AP.take 4

ubyte_images :: AP.Parser [Grey]
ubyte_images = do
  magic_number <- parseInt
  when (magic_number /= 2051) $ fail $
    "Invalid magic number: " ++ show magic_number ++ ". Should be 2051 for image ubyte files. Did you try to parse a label file?"
  nb_images <- parseInt
  nb_rows <- parseInt
  nb_cols <- parseInt
  replicateM nb_images $ do
    imgbstring <- AP.take (nb_rows * nb_cols)
    return $ (fromFunction (VP.Z VP.:. nb_rows VP.:. nb_cols) $
              \(VP.Z VP.:. i VP.:. j) ->
              GreyPixel $ BS.index imgbstring (fromIntegral $ j + i * nb_cols)) :: AP.Parser Grey

ubyte_labels :: AP.Parser [Int]
ubyte_labels = do
  magic_number <- parseInt
  when (magic_number /= 2049) $ fail $
    "Invalid magic number: " ++ show magic_number ++ ". Should be 2049 for image ubyte files. Did you try to parse an image file?"
  nb_labels <- parseInt
  replicateM nb_labels $ fmap fromIntegral $ AP.anyWord8

load_mnist :: (MonadIO m, MonadError String m)
           => Maybe String -> Maybe String -> FilePath -> FilePath
           -> Producer (Grey, Int) m ()
load_mnist mimgurl mlabelurl imgfpath labelfpath = do
  images_ubyte <- loadOrDownload mimgurl imgfpath
  labels_ubyte <- loadOrDownload mlabelurl labelfpath
  images <- case AP.parse ubyte_images images_ubyte of
             AP.Fail _ ctxs err -> throwError
                              $ "Failed to parse ubyte images:\nContext: "
                              ++ show ctxs ++ "\nError: " ++ err
             AP.Done _ i -> return i
  labels <- case AP.parse ubyte_labels labels_ubyte of
             AP.Fail _ ctxs err -> throwError
                              $ "Failed to parse ubyte images:\nContext: "
                              ++ show ctxs ++ "\nError: " ++ err
             AP.Done _ l -> return l
  forM_ (zip images labels) $ yield
  return ()
