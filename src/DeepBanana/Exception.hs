{-# LANGUAGE GADTs, TypeFamilies, DataKinds, ImplicitParams, StandaloneDeriving, DeriveGeneric #-}
{-|
A stab at an extensible exceptions system, used internally. To be displaced eventually by whatever "composable effects" library becomes the de facto standard in the Haskell community.
-}
module DeepBanana.Exception (
  -- * Extensible variants of exception datatypes
    Coproduct(..)
  , Variant(..)
  , throwVariant
  , catchVariant
  -- * Stack traces
  , WithStack
  , withStack
  -- * Managing exception monads
  , unsafeRunExcept
  , embedExcept
  , runExceptTAs
  -- * Specific handlers
  , attemptGCThenRetryOn
  ) where

import DeepBanana.Prelude
import Data.Typeable
import GHC.SrcLoc
import GHC.Stack
import System.Mem

-- | Coproduct of datatypes, indexed by a type-level lists of these datatypes.
-- For instance, @'Coproduct' '[a,b]@ is equivalent to @'Either' a b@.
data Coproduct (l :: [*]) where
  Left' :: a -> Coproduct (a ': l)
  Right' :: Coproduct l -> Coproduct (a ': l)

instance {-# OVERLAPPING #-} (NFData a) => NFData (Coproduct '[a]) where
  rnf (Left' x) = rnf x

instance (NFData a, NFData (Coproduct b)) => NFData (Coproduct (a ': b)) where
  rnf (Left' a) = rnf a
  rnf (Right' cpb) = rnf cpb

-- | Type class to deal with variant datatypes in a generic fashion.
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

-- | Throwing an exception when the error datatype is a variant including that datatype.
throwVariant :: (MonadError t m,  Variant t e) => e -> m a
throwVariant = throwError . setVariant

-- | Catching one possible exception type out of a variant.
catchVariant :: forall m t e a
             . (MonadError t m, Variant t e)
             => m a -> (e -> m a) -> m a
catchVariant action handler =
  let handler' t = case getVariant t :: Maybe e of
        Nothing -> throwError t
        Just e -> handler e
  in catchError action handler'

-- | Simple wrapper adding a call stack to an arbitrary exception.
data WithStack e = WithStack {
    exception :: e
  , callStack :: String
  } deriving (Eq, Typeable, Generic)

instance (NFData e) => NFData (WithStack e)

instance (Show e) => Show (WithStack e) where
  show wse = show (exception wse) ++ "\n" ++ callStack wse

instance (Eq e, Show e, Typeable e) => Exception (WithStack e)

withStack :: (?loc :: CallStack) => e -> WithStack e
withStack exception = WithStack exception (showCallStack ?loc)

instance (Variant t e) => Variant (WithStack t) e where
  setVariant = withStack . setVariant
  getVariant = getVariant . exception

-- | Gets the @'Right'@ part of a computation, throwing an imprecise exception
-- in case of @'Left'@.
unsafeRunExcept :: (Exception e) => Either e a -> a
unsafeRunExcept (Left err) = throw err
unsafeRunExcept (Right out) = out

-- | Embeds an @'Either'@ value into an arbitrary error monad.
embedExcept :: (MonadError e m) => Either e a -> m a
embedExcept ea = do
  case ea of
   Left err -> throwError err
   Right res -> return res

-- | Synonym for @'ExceptT'@ specifying the exception type via a @'Proxy'@. Useful to
-- disambiguate types for the compiler.
runExceptTAs :: Proxy t -> ExceptT t m a -> m (Either t a)
runExceptTAs _ action = runExceptT action

-- | Runs a computation, and in case of a specific exception being raised performs
-- garbage collection before attempting the computation again. This is useful when
-- allocating a lot of memory quickly on the GPU, as the garbage collector may not have
-- enough time to run by itself. Throws back the same exception if the second run fails
-- anyway. This is used anytime an allocation is performed on a GPU device, most notably
-- @'DeepBanana.Tensor.Mutable.emptyTensor'@ and other tensor creation functions.
attemptGCThenRetryOn :: forall m t e a
                     . (MonadIO m, MonadError t m, Variant t e)
                     => Proxy e -> m a -> m a
attemptGCThenRetryOn _ action = do
  let handler e = do
        liftIO $ performGC
        action
  action `catchVariant` (handler :: e -> m a)
