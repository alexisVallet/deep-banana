{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module DeepBanana.Data.Exception (
    EmptyBatch
  , emptyBatch
  ) where

import DeepBanana.Exception
import DeepBanana.Prelude

newtype EmptyBatch = EmptyBatch (WithStack String)
                     deriving (Eq, Show, Typeable, Exception)

emptyBatch :: String -> EmptyBatch
emptyBatch = EmptyBatch . withStack
