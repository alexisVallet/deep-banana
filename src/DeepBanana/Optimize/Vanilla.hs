{-# LANGUAGE TemplateHaskell #-}
module DeepBanana.Optimize.Vanilla (
    VanillaT
  , runVanilla
  , vanilla
  ) where

import Control.Monad.Reader
import Data.VectorSpace
import Control.Lens

