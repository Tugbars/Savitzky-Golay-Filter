# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and languages
project(SavitzkyGolayFilter LANGUAGES C CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# BUILD OPTIONS
# =============================================================================
option(USE_PARALLEL_SAVGOL "Build parallel (OpenMP) version of Savitzky-Golay filter" ON)

# Print selected build mode
if(USE_PARALLEL_SAVGOL)
    message(STATUS "Building PARALLEL Savitzky-Golay implementation (with OpenMP)")
else()
    message(STATUS "Building SEQUENTIAL Savitzky-Golay implementation")
endif()

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

# =============================================================================
# INCLUDE SUBDIRECTORIES
# =============================================================================
# Conditionally include the appropriate implementation
if(USE_PARALLEL_SAVGOL)
    add_subdirectory(src/parallel)
else()
    add_subdirectory(src/iterative)
endif()

# Always include tests (they should work with either implementation)
add_subdirectory(test/iterative)

# Enable testing
enable_testing()