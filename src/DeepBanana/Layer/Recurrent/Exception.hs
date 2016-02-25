{-# LANGUAGE ImplicitParams, GeneralizedNewtypeDeriving, DeriveDataTypeable #-}
module DeepBanana.Layer.Recurrent.Exception (
    IncompatibleLength
  , incompatibleLength
  ) where

import GHC.Stack

import DeepBanana.Exception
import DeepBanana.Prelude

newtype IncompatibleLength = IncompatibleLength (WithStack String)
                           deriving (Show, Eq, Typeable, Exception)

incompatibleLength :: (?loc :: CallStack) => String -> IncompatibleLength
incompatibleLength = IncompatibleLength . withStack
