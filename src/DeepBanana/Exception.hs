{-# LANGUAGE TypeFamilies, DataKinds, ImplicitParams #-}
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
import GHC.Stack

type family Coproduct l where
  Coproduct '[a] = a
  Coproduct (e ': l) = Either e (Coproduct l)

class Variant t e where
  setVariant :: e -> t
  getVariant :: t -> Maybe e

instance Variant a a where
  setVariant = id
  getVariant = Just

instance (Variant b e) => Variant (Either a b) e where
  setVariant e = Right $ setVariant e
  getVariant (Left _) = Nothing
  getVariant (Right b) = getVariant b

instance {-# OVERLAPPING #-} Variant (Either a b) a where
  setVariant a = Left a
  getVariant (Left a) = Just a
  getVariant (Right _) = Nothing

instance (Show a, Show b, Typeable a, Typeable b) => Exception (Either a b)

throwVariant :: (MonadError t m,  Variant t e) => e -> m a
throwVariant = throwError . setVariant

data WithStack e = WithStack {
    exception :: e
  , callStack :: String
  } deriving (Eq, Show, Typeable)

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
