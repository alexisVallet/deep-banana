{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-|
FFI wrapper for CuRAND. Naming simply removes the curand or curand_ prefixes of wrapped functions. Pointers referring to device arrays are wrapped as @'DevicePtr'@, while those referring to host arrays as regular @'Ptr'@. Output types are wrapped as @'DeviceM'@ to ensure they are executed with a specific device. See <http://docs.nvidia.com/cuda/curand CuRAND's documentation> for documentation of each individual function.
-}
module DeepBanana.Device.CuRAND where

import Foreign
import Foreign.C
import Foreign.CUDA.Types
import System.IO.Unsafe

import DeepBanana.Device.Monad
import DeepBanana.Prelude hiding (Handle)

#include <curand.h>

-- Enums for status, rng, ordering, direction vector set, method.
-- Datatypes for generators, distributions.
newtype Status = Status {
  unStatus :: CInt
  } deriving (Show, Eq, Storable)

#{enum Status, Status
 , status_success = CURAND_STATUS_SUCCESS
 , status_version_mismatch = CURAND_STATUS_VERSION_MISMATCH
 , status_not_initialized = CURAND_STATUS_NOT_INITIALIZED
 , status_allocation_failed = CURAND_STATUS_ALLOCATION_FAILED
 , status_type_error = CURAND_STATUS_TYPE_ERROR
 , status_out_of_range = CURAND_STATUS_OUT_OF_RANGE
 , status_length_not_multiple = CURAND_STATUS_LENGTH_NOT_MULTIPLE
 , status_double_precision_required = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
 , status_launch_failure = CURAND_STATUS_LAUNCH_FAILURE
 , status_preexisting_failure = CURAND_STATUS_PREEXISTING_FAILURE
 , status_initialization_failed = CURAND_STATUS_INITIALIZATION_FAILED
 , status_arch_mismatch = CURAND_STATUS_ARCH_MISMATCH
 , status_internal_error = CURAND_STATUS_INTERNAL_ERROR
 }

newtype RngType = RngType {
  unRngType :: CInt
  } deriving (Show, Eq, Storable)

#{enum RngType, RngType
 , rng_test = CURAND_RNG_TEST
 , rng_pseudo_default = CURAND_RNG_PSEUDO_DEFAULT
 , rng_pseudo_xorwow = CURAND_RNG_PSEUDO_XORWOW
 , rng_pseudo_mrg32k3a = CURAND_RNG_PSEUDO_MRG32K3A
 , rng_pseudo_mtgp32 = CURAND_RNG_PSEUDO_MTGP32
 , rng_pseudo_mt19937 = CURAND_RNG_PSEUDO_MT19937
 , rng_pseudo_philox4_32_10 = CURAND_RNG_PSEUDO_PHILOX4_32_10
 , rng_quasi_default = CURAND_RNG_QUASI_DEFAULT
 , rng_quasi_sobol32 = CURAND_RNG_QUASI_SOBOL32
 , rng_quasi_scrambled_sobol32 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
 , rng_quasi_sobol64 = CURAND_RNG_QUASI_SOBOL64
 , rng_quasi_scrambled_sobol64 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
 }

newtype CuRANDOrdering = CuRANDOrdering {
  unCuRANDOrdering :: CInt
  } deriving (Show, Eq, Storable)

#{enum CuRANDOrdering, CuRANDOrdering
 , ordering_pseudo_best = CURAND_ORDERING_PSEUDO_BEST
 , ordering_pseudo_default = CURAND_ORDERING_PSEUDO_DEFAULT
 , ordering_pseudo_seeded = CURAND_ORDERING_PSEUDO_SEEDED
 , ordering_quasi_default = CURAND_ORDERING_QUASI_DEFAULT
 }

newtype DirectionVectorSet = DirectionVectorSet {
  unDirectionVectorSet :: CInt
  } deriving (Show, Eq, Storable)

newtype DirectionVectors32 = DirectionVectors32 {
  unDirectionVectors32 :: Ptr CUInt
  } deriving Storable

newtype DirectionVectors64 = DirectionVectors64 {
  unDirectionVectors64 :: Ptr CULLong
  }

#{enum DirectionVectorSet, DirectionVectorSet
 , direction_vectors_32_joekuo6 = CURAND_DIRECTION_VECTORS_32_JOEKUO6
 , scrambled_direction_vectors_32_joekuo6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
 , direction_vectors_64_joekuo6 = CURAND_DIRECTION_VECTORS_64_JOEKUO6
 , scrambled_direction_vectors_64_joekuo6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
 }

newtype Generator d = Generator {
  unGenerator :: Ptr ()
  } deriving Storable

newtype Distribution = Distribution {
  unDistribution :: Ptr CDouble
  } deriving (Show, Eq, Storable)

newtype DistributionShift = DistributionShift {
  unDistributionShift :: Ptr ()
  } deriving Storable

newtype DistributionM2Shift = DistributionM2Shift {
  unDistributionM2Shift :: Ptr ()
  } deriving Storable

newtype HistogramM2 = HistogramM2 {
  unHistogramM2 :: Ptr ()
  } deriving Storable

newtype HistogramM2K = HistogramM2K {
  unHistogramM2K :: Ptr CInt
  } deriving (Show, Eq, Storable)

newtype HistogramM2V = HistogramM2V {
  unHistogramM2V :: Ptr Distribution
  } deriving (Show, Eq, Storable)

newtype DiscreteDistribution = DiscreteDistribution {
  unDiscreteDistribution :: Ptr ()
  }

newtype Method = Method {
  unMethod :: CInt
  } deriving (Show, Eq, Storable)

#{enum Method, Method
 , choose_best = CURAND_CHOOSE_BEST
 , itr = CURAND_ITR
 , knuth = CURAND_KNUTH
 , hitr = CURAND_HITR
 , m1 = CURAND_M1
 , m2 = CURAND_M2
 , binary_search = CURAND_BINARY_SEARCH
 , discrete_gauss = CURAND_DISCRETE_GAUSS
 , rejection = CURAND_REJECTION
 , device_api = CURAND_DEVICE_API
 , fast_rejection = CURAND_FAST_REJECTION
 , _3rd  = CURAND_3RD
 , definition = CURAND_DEFINITION
 , poisson = CURAND_POISSON
 }

-- Random number generator manipulation functions.
foreign import ccall safe "curandCreateGenerator"
  createGenerator :: Ptr (Generator d) -> RngType -> DeviceM d Status

foreign import ccall safe "curandCreateGeneratorHost"
  createGeneratorHost :: Ptr (Generator d) -> RngType -> DeviceM d Status

foreign import ccall safe "curandDestroyGenerator"
  destroyGenerator :: Generator d -> DeviceM d Status

-- Library version.
foreign import ccall safe "curandGetVersion"
  getVersion :: Ptr CInt -> DeviceM d Status

-- Set the CUDA stream.
foreign import ccall safe "curandSetStream"
  setStream :: Generator d -> Stream -> DeviceM d Status

-- Seeding.
foreign import ccall safe "curandSetPseudoRandomGeneratorSeed"
  setPseudoRandomGeneratorSeed :: Generator d -> CULLong -> DeviceM d Status

foreign import ccall safe "curandSetGeneratorOffset"
  setGeneratorOffset :: Generator d -> CULLong -> DeviceM d Status

-- CuRANDOrdering.
foreign import ccall safe "curandSetGeneratorOrdering"
  setGeneratorOrdering :: Generator d -> CuRANDOrdering -> DeviceM d Status

-- Dimensions.
foreign import ccall safe "curandSetQuasiRandomGeneratorDimensions"
  setQuasiRandomGeneratorDimensions :: Generator d -> CUInt -> DeviceM d Status

-- Random number generation.
foreign import ccall safe "curandGenerate"
  generate :: Generator d -> DevicePtr CUInt -> CSize -> DeviceM d Status

foreign import ccall safe "curandGenerateLongLong"
  generateLongLong :: Generator d -> DevicePtr CULLong -> CSize -> DeviceM d Status

-- Uniform distribution.
foreign import ccall safe "curandGenerateUniform"
  generateUniform :: Generator d -> DevicePtr CFloat -> CSize -> DeviceM d Status

foreign import ccall safe "curandGenerateUniformDouble"
  generateUniformDouble :: Generator d -> DevicePtr CDouble -> CSize -> DeviceM d Status

-- Normal distribution.
foreign import ccall safe "curandGenerateNormal"
  generateNormal :: Generator d -> DevicePtr CFloat -> CSize -> CFloat -> CFloat
                 -> DeviceM d Status

foreign import ccall safe "curandGenerateNormalDouble"
  generateNormalDouble :: Generator d -> DevicePtr CDouble -> CSize -> CDouble -> CDouble
                       -> DeviceM d Status

-- Log-normal distribution.
foreign import ccall safe "curandGenerateLogNormal"
  generateLogNormal :: Generator d -> DevicePtr CFloat -> CSize -> CFloat -> CFloat
                    -> DeviceM d Status

foreign import ccall safe "curandGenerateLogNormalDouble"
  generateLogNormalDouble :: Generator d -> DevicePtr CDouble -> CSize -> CDouble
                          -> CDouble -> DeviceM d Status

-- Poisson distribution.
foreign import ccall safe "curandCreatePoissonDistribution"
  createPoissonDistribution :: CDouble -> Ptr DiscreteDistribution -> DeviceM d Status

foreign import ccall safe "curandDestroyDistribution"
  destroyDistribution :: DiscreteDistribution -> DeviceM d Status

foreign import ccall safe "curandGeneratePoisson"
  generatePoisson :: Generator d -> DevicePtr CUInt -> CSize -> CDouble -> DeviceM d Status

foreign import ccall safe "curandGeneratePoissonMethod"
  generatePoissonMethod :: Generator d -> DevicePtr CUInt -> CSize -> CDouble
                        -> DeviceM d Status

-- Setting up starting states.
foreign import ccall safe "curandGenerateSeeds"
  generateSeeds :: Generator d -> DeviceM d Status

foreign import ccall safe "curandGetDirectionVectors32"
  getDirectionVectors32 :: Ptr (Ptr DirectionVectors32) -> DirectionVectorSet
                        -> DeviceM d Status

foreign import ccall safe "curandGetScrambleConstants32"
  getScrambleConstants32 :: Ptr (Ptr CUInt) -> DeviceM d Status

foreign import ccall safe "curandGetDirectionVectors64"
  getDirectionVectors64 :: Ptr (Ptr DirectionVectors64) -> DirectionVectors64
                        -> DeviceM d Status

foreign import ccall safe "curandGetScrambleConstants64"
  getScrambleConstants64 :: Ptr (Ptr CULLong) -> DeviceM d Status
