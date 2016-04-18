{-# LANGUAGE GeneralizedNewtypeDeriving, DeriveGeneric, ImplicitParams #-}
module DeepBanana.Data.Exception (
    EmptyBatch
  , emptyBatch
  ) where

import GHC.Stack

import DeepBanana.Exception
import DeepBanana.Prelude

newtype EmptyBatch = EmptyBatch (WithStack String)
                     deriving (Eq, Show, Typeable, Exception, Generic, NFData)

emptyBatch :: (?loc :: CallStack) => String -> EmptyBatch
emptyBatch = EmptyBatch . withStack
