{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module DeepBanana.Tensor.Exception (
    OutOfMemory
  , outOfMemory
  , IncompatibleSize
  , incompatibleSize
  , IncompatibleShape
  , incompatibleShape
  ) where


import DeepBanana.Prelude
import DeepBanana.Exception

newtype OutOfMemory = OutOfMemory (WithStack String)
                      deriving (Eq, Show, Typeable, Exception)

outOfMemory = OutOfMemory . withStack

newtype IncompatibleSize = IncompatibleSize (WithStack String)
                           deriving (Eq, Show, Typeable, Exception)

incompatibleSize = IncompatibleSize . withStack

newtype IncompatibleShape = IncompatibleShape (WithStack String)
                            deriving (Eq, Show, Typeable, Exception)

incompatibleShape = IncompatibleShape . withStack
