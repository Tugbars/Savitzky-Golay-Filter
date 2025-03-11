# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(SavitzkyGolayFilter C)

# Set C standard (optional, adjust as needed)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


# Enable AVX
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mavx -mfma)
elseif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/arch:AVX)
endif()

# Include directories for header files
include_directories(${CMAKE_SOURCE_DIR}/include/iterative)

# Source files
set(SOURCES
    src/iterative/savgolFilter.c
    test/iterative/main.c
)

# Create the executable
add_executable(savgol_filter ${SOURCES})

# Optionally, specify output directory for the executable
set_target_properties(savgol_filter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)