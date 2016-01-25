{-# LANGUAGE GADTs #-}
module DeepBanana.Tensor.Shape (
    Shape(..)
  , dimensions
  , nbdim
  , size
  , ScalarShape(..)
  ) where

import GHC.TypeLits

infixr 3 (:.)
data Shape (n :: Nat) where
  Z :: Shape 0
  (:.) :: Int -> Shape n -> Shape (n + 1)

instance Show (Shape n) where
  show s = "s" ++ show (dimensions s)

dimensions :: Shape n -> [Int]
dimensions Z = []
dimension (n :. s) = n : dimensions s

nbdim :: Shape n -> Int
nbdim Z = 0
nbdim (_ :. s) = 1 + nbdim s

size :: Shape n -> Int
size Z = 0
size (n :. Z) = n
size (n :. s) = n * size s

class ScalarShape (n :: Nat) where
  scalarShape :: Shape n

instance ScalarShape 0 where
  scalarShape = Z

instance (ScalarShape n) => ScalarShape (n + 1) where
  scalarShape = 1 :. (scalarShape :: Shape n)
