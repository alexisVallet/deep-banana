{-# LANGUAGE GADTs, TypeFamilies, DataKinds, ImplicitParams, StandaloneDeriving #-}
module DeepBanana.Exception (
    WithStack
  , withStack
  , Coproduct(..)
  , Variant(..)
  , throwVariant
  , unsafeRunExcept
  , embedExcept
  , runExceptTAs
  ) where

import DeepBanana.Prelude
import Data.Typeable
import GHC.SrcLoc
import GHC.Stack

data Coproduct (l :: [*]) where
  Left' :: a -> Coproduct (a ': l)
  Right' :: Coproduct l -> Coproduct (a ': l)

class Variant t e where
  setVariant :: e -> t
  getVariant :: t -> Maybe e

instance Variant a a where
  setVariant = id
  getVariant = Just

instance (Variant (Coproduct b) e) => Variant (Coproduct (a ': b)) e where
  setVariant e = Right' $ setVariant e
  getVariant (Left' _) = Nothing
  getVariant (Right' b) = getVariant b

instance {-# OVERLAPPING #-} Variant (Coproduct (a ': b)) a where
  setVariant a = Left' a
  getVariant (Left' a) = Just a
  getVariant (Right' _) = Nothing

instance {-# OVERLAPPING #-} (Show a) => Show (Coproduct '[a]) where
  show (Left' a) = show a

instance (Show a, Show (Coproduct b)) => Show (Coproduct (a ': b)) where
  show (Left' a) = show a
  show (Right' b) = show b

deriving instance (Typeable a) => Typeable (Coproduct '[a])
deriving instance (Typeable a, Typeable (Coproduct b)) => Typeable (Coproduct (a ': b))

deriving instance Typeable '[]
deriving instance (Typeable a, Typeable b) => Typeable (a ': b)

instance (Show a, Typeable a) => Exception (Coproduct '[a])
instance (Show a, Show (Coproduct b), Typeable a, Typeable b, Typeable (Coproduct b))
         => Exception (Coproduct (a ': b))

throwVariant :: (MonadError t m,  Variant t e) => e -> m a
throwVariant = throwError . setVariant

data WithStack e = WithStack {
    exception :: e
  , callStack :: String
  } deriving (Eq, Typeable)

instance (Show e) => Show (WithStack e) where
  show wse = show (exception wse) ++ "\n" ++ callStack wse

instance (Eq e, Show e, Typeable e) => Exception (WithStack e)

withStack :: (?loc :: CallStack) => e -> WithStack e
withStack exception = WithStack exception (showCallStack ?loc)

instance (Variant t e) => Variant (WithStack t) e where
  setVariant = withStack . setVariant
  getVariant = getVariant . exception

unsafeRunExcept :: (Exception e) => Either e a -> a
unsafeRunExcept (Left err) = throw err
unsafeRunExcept (Right out) = out

embedExcept :: (MonadError e m) => Either e a -> m a
embedExcept ea = do
  case ea of
   Left err -> throwError err
   Right res -> return res

runExceptTAs :: Proxy t -> ExceptT t m a -> m (Either t a)
runExceptTAs _ action = runExceptT action
