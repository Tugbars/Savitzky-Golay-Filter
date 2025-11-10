# Define the source and include directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/iterative)

# List the source files
set(SOURCES
    ${SRC_DIR}/savgolFilterMT.c
)

# List the header files (optional, but good practice for IDEs)
set(HEADERS
    ${INCLUDE_DIR}/savgolFilter.h
)

# Find OpenMP package
find_package(OpenMP)

# Create a static library
add_library(savgolFilterMT STATIC ${SOURCES} ${HEADERS})

# Enable AVX/FMA
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(savgolFilterMT PRIVATE -mavx -mfma)
elseif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
    target_compile_options(savgolFilterMT PRIVATE /arch:AVX)
endif()

# CRITICAL: Define SAVGOL_PARALLEL_BUILD to enable threaded API
target_compile_definitions(savgolFilterMT PUBLIC SAVGOL_PARALLEL_BUILD)

# Add OpenMP support if available
if (OpenMP_C_FOUND)
    target_compile_options(savgolFilterMT PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(savgolFilterMT PRIVATE OpenMP::OpenMP_C)
    message(STATUS "✓ OpenMP found (version ${OpenMP_C_VERSION}) - parallel version will use multithreading")
else()
    message(WARNING "✗ OpenMP NOT found - parallel version will compile but run sequentially!")
    message(WARNING "  Install OpenMP support for your compiler to enable parallelization")
endif()

# Set include directories for the library
target_include_directories(savgolFilterMT PUBLIC ${INCLUDE_DIR})

# Link the math library
target_link_libraries(savgolFilterMT PRIVATE m)

# Ensure Cygwin runtime symbols are resolved
if (CYGWIN)
    target_link_libraries(savgolFilterMT PUBLIC c cygwin)
endif()

# Add an alias for consistent naming
add_library(savgolFilter::MT ALIAS savgolFilterMT)