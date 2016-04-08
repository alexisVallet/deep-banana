{-# LANGUAGE PolyKinds, MultiParamTypeClasses, TypeFamilies, UndecidableInstances, ScopedTypeVariables #-}
module DeepBanana.Device (
    module DeepBanana.Device.Monad
  ) where

import DeepBanana.Device.Monad (
    Device
  , DeviceM
  , runDeviceM
  , cudnnHandle
  , cublasHandle
  , deviceMallocArray
  )
