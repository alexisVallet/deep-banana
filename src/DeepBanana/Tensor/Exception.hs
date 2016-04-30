{-# LANGUAGE GeneralizedNewtypeDeriving, ImplicitParams, DeriveGeneric #-}
{-|
Exception datatypes for Tensor functions.
-}
module DeepBanana.Tensor.Exception (
    OutOfMemory
  , outOfMemory
  , IncompatibleSize
  , incompatibleSize
  , IncompatibleShape
  , incompatibleShape
  , IncompatibleDevice
  , incompatibleDevice
  ) where

import GHC.Stack

import DeepBanana.Prelude
import DeepBanana.Exception

newtype OutOfMemory = OutOfMemory (WithStack String)
                      deriving (Eq, Show, Typeable, Exception, Generic, NFData)

outOfMemory :: (?loc :: CallStack) => String -> OutOfMemory
outOfMemory = OutOfMemory . withStack

newtype IncompatibleSize = IncompatibleSize (WithStack String)
                           deriving (Eq, Show, Typeable, Exception, Generic, NFData)

incompatibleSize :: (?loc :: CallStack) => String -> IncompatibleSize
incompatibleSize = IncompatibleSize . withStack

newtype IncompatibleShape = IncompatibleShape (WithStack String)
                            deriving (Eq, Show, Typeable, Exception, Generic, NFData)

incompatibleShape :: (?loc :: CallStack) => String -> IncompatibleShape
incompatibleShape = IncompatibleShape . withStack

newtype IncompatibleDevice = IncompatibleDevice (WithStack String)
                             deriving (Eq, Show, Typeable, Exception, Generic, NFData)

incompatibleDevice :: (?loc :: CallStack) => String -> IncompatibleDevice
incompatibleDevice = IncompatibleDevice . withStack
