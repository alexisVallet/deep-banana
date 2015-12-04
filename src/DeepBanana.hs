module DeepBanana (
  -- Main modules.
    module DeepBanana.Tensor
  , module DeepBanana.Layer
  , module DeepBanana.Data
  , module DeepBanana.Optimize
  -- Convenience re-exports.
  , module GHC.TypeLits
  , module Data.Proxy
  , module Pipes
  ) where

import GHC.TypeLits (
    Nat
  , KnownNat
  , type (+)
  , type (-)
  , type (*)
  , type (^)
  , natVal
  , SomeNat(..)
  , someNatVal
  , sameNat
  , CmpNat
  )
import Data.Proxy
import Pipes hiding (Proxy)

import DeepBanana.Tensor
import DeepBanana.Layer
import DeepBanana.Data
import DeepBanana.Optimize
