# Define the source and include directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/iterative)

# List the source files
set(SOURCES
    ${SRC_DIR}/savgol_filter_vectorized.c
)

# List the header files
set(HEADERS
    ${INCLUDE_DIR}/savgolFilter.h
    ${SRC_DIR}/savgol_simd_ops.h
    ${SRC_DIR}/savgol_soa_convert.h
    ${SRC_DIR}/savgol_kernels.h
)

# Create a static library
add_library(savgolFilter STATIC ${SOURCES} ${HEADERS})

#==============================================================================
# SIMD Compiler Flags
#==============================================================================

# Base optimization flags
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(savgolFilter PRIVATE -O3 -ffast-math)
endif()

# SIMD-specific flags
if(ENABLE_NATIVE)
    # Auto-detect CPU features
    message(STATUS "SavGol: Using -march=native for automatic CPU detection")
    if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(savgolFilter PRIVATE -march=native)
    endif()
else()
    # Manual SIMD configuration
    if(ENABLE_AVX512)
        message(STATUS "SavGol: AVX-512 enabled")
        if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(savgolFilter PRIVATE -mavx512f -mavx512dq)
        elseif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
            target_compile_options(savgolFilter PRIVATE /arch:AVX512)
        endif()
    elseif(ENABLE_AVX2)
        message(STATUS "SavGol: AVX2 enabled")
        if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(savgolFilter PRIVATE -mavx2 -mfma)
        elseif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
            target_compile_options(savgolFilter PRIVATE /arch:AVX2)
        endif()
    elseif(ENABLE_SSE2)
        message(STATUS "SavGol: SSE2 enabled")
        if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(savgolFilter PRIVATE -msse2)
        elseif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
            target_compile_options(savgolFilter PRIVATE /arch:SSE2)
        endif()
    else()
        message(STATUS "SavGol: Scalar-only build (no SIMD)")
    endif()
endif()

# Set include directories for the library
target_include_directories(savgolFilter PUBLIC 
    ${INCLUDE_DIR}
    ${SRC_DIR}  # For SIMD headers
)

# Link the math library if needed
target_link_libraries(savgolFilter PRIVATE m)

# Ensure Cygwin runtime symbols are resolved
if (CYGWIN)
    target_link_libraries(savgolFilter PUBLIC c cygwin)
endif()