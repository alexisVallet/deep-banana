{-# LANGUAGE GADTs, TypeFamilies, DataKinds, ImplicitParams, StandaloneDeriving, DeriveGeneric #-}
module DeepBanana.Exception (
    WithStack
  , withStack
  , Coproduct(..)
  , Variant(..)
  , throwVariant
  , catchVariant
  , unsafeRunExcept
  , embedExcept
  , runExceptTAs
  , attemptGCThenRetryOn
  ) where

import DeepBanana.Prelude
import Data.Typeable
import GHC.SrcLoc
import GHC.Stack
import System.Mem

data Coproduct (l :: [*]) where
  Left' :: a -> Coproduct (a ': l)
  Right' :: Coproduct l -> Coproduct (a ': l)

instance {-# OVERLAPPING #-} (NFData a) => NFData (Coproduct '[a]) where
  rnf (Left' x) = rnf x

instance (NFData a, NFData (Coproduct b)) => NFData (Coproduct (a ': b)) where
  rnf (Left' a) = rnf a
  rnf (Right' cpb) = rnf cpb
                                      
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

catchVariant :: forall m t e a
             . (MonadError t m, Variant t e)
             => m a -> (e -> m a) -> m a
catchVariant action handler =
  let handler' t = case getVariant t :: Maybe e of
        Nothing -> throwError t
        Just e -> handler e
  in catchError action handler'

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

attemptGCThenRetryOn :: forall m t e a
                     . (MonadIO m, MonadError t m, Variant t e)
                     => Proxy e -> m a -> m a
attemptGCThenRetryOn _ action = do
  let handler e = do
        liftIO $ performGC
        action
  action `catchVariant` (handler :: e -> m a)
