module Tatoeba (
    loadTatoeba
  , Sentence(..)
  ) where

import DeepBanana.Prelude hiding (takeWhile, skipWhile)

import Codec.Compression.BZip
import Control.Lens hiding (index)
import Data.Attoparsec.Text hiding (take)
import Data.Serialize as S
import qualified Network.Wreq as Wreq
import System.Directory
import System.FilePath (takeDirectory)

data Sentence = Sentence {
    sentenceId :: Int
  , lang :: Text
  , text :: Text
  } deriving (Eq, Ord, Show, Read)

instance Serialize Sentence where
  put s = do
    S.put $ sentenceId s
    S.put $ encodeUtf8 $ lang s
    S.put $ encodeUtf8 $ text s
  get = pure Sentence <*> S.get <*> fmap decodeUtf8 S.get <*> fmap decodeUtf8 S.get

parseSentences :: Parser [Sentence]
parseSentences = many $ do
  sentenceId <- decimal
  char '\t'
  lang <- takeTill (== '\t')
  char '\t'
  text <- takeTill isEndOfLine
  endOfLine
  return $ Sentence sentenceId lang text

parseLinks :: Parser (IntMap Int)
parseLinks = fmap mapFromList $ many $ do
  sentenceId <- decimal
  char '\t'
  transId <- decimal
  endOfLine
  return (sentenceId, transId)

loadTatoeba :: (MonadIO m, MonadError String m)
            => FilePath -> FilePath
            -> Producer (Sentence, Sentence) m ()
loadTatoeba sentFpath linksFpath = do
  links_raw <- readFile linksFpath
  sentences_raw <- readFile sentFpath
  putStrLn $ "links_raw length: " ++ pack (show (length links_raw))
  putStrLn $ "sentences_raw length: " ++ pack (show (length sentences_raw))
  putStrLn $ "links: " ++ take 50 links_raw
  putStrLn $ "sentences: " ++ take 50 sentences_raw
  links <- case parseOnly parseLinks links_raw of
            Left err -> throwError
                        $ "Failed to parse tatoeba links:\nError: " ++ err
            Right l -> return l
  sentences <- case parseOnly parseSentences sentences_raw of
                Left err -> throwError
                            $ "Failed to parse tatoeba sentences:\nError: " ++ err
                Right s -> return s
  let nb_links = length links
  putStrLn $ "links length: " ++ pack (show nb_links)
  putStrLn $ "sentences length: " ++ pack (show (length sentences))
  let sentMap = asIntMap $ mapFromList $ fmap (\s -> (sentenceId s, s)) sentences
  forM_ (mapToList links) $ \(sourceId, targetId) -> do
    case pure (,) <*> lookup sourceId sentMap <*> lookup targetId sentMap of
     Nothing -> return ()
     Just (source, target) -> yield (source, target)
