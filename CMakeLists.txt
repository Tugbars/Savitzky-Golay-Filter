# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and languages
project(SavitzkyGolayFilter LANGUAGES C CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==============================================================================
# SIMD Configuration Options
#==============================================================================
option(ENABLE_AVX512 "Enable AVX-512 optimizations" OFF)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_SSE2 "Enable SSE2 optimizations" ON)
option(ENABLE_NATIVE "Enable -march=native (auto-detect CPU)" OFF)

# Ensure Cygwin runtime linkage for all targets
if (CYGWIN)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc -lcygwin")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lc -lcygwin")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -lc -lcygwin")
endif()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Fetch FFF (Fake Function Framework)
FetchContent_Declare(
    fff
    GIT_REPOSITORY https://github.com/meekrosoft/fff.git
    GIT_TAG master
)
FetchContent_MakeAvailable(fff)

# Include subdirectories
add_subdirectory(src/iterative)
add_subdirectory(test/iterative)

# Enable testing
enable_testing()