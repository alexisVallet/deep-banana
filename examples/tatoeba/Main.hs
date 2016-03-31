module Main where

import Data.List (nub)
import Data.Serialize (encode)
import DeepBanana.Prelude
import qualified Pipes.Prelude as P

import Tatoeba

main :: IO ()
main = do
  let rootDir = "data" </> "tatoeba"
      sentFpath = rootDir </> "sentences.csv"
      linksFpath = rootDir </> "links.csv"
      engToFraFpath = rootDir </> "engToFra"
  eEngToFra <- runExceptT
              $ P.toListM
              $ loadTatoeba sentFpath linksFpath
              >-> P.filter (\(s1,s2) -> lang s1 == "eng" && lang s2 == "fra")
  case eEngToFra of
   Left err -> error err
   Right engToFra -> do
     let nbSamples = length engToFra
         (eng, fra) = unzip engToFra
         engAlphaSize = length $ nub $ unpack $ concatMap text $ eng
         fraAlphaSize = length $ nub $ unpack $ concatMap text $ fra
     putStrLn "pouet!"
     putStrLn $ "Dataset size: " ++ pack (show nbSamples)
     writeFile engToFraFpath $ encode engToFra
