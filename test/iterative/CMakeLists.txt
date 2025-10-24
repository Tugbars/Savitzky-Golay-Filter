#==============================================================================
# Main Application Executable (DEFAULT TARGET)
#==============================================================================

# Define the application executable
add_executable(savgol_app
    main.cpp
)

# Set include directories for the application
target_include_directories(savgol_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include/iterative
    ${CMAKE_SOURCE_DIR}/src/iterative  # For SIMD headers
)

# Apply same SIMD flags as library
if(ENABLE_NATIVE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(savgol_app PRIVATE -march=native)
    endif()
elseif(ENABLE_AVX512)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(savgol_app PRIVATE -mavx512f -mavx512dq)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(savgol_app PRIVATE /arch:AVX512)
    endif()
elseif(ENABLE_AVX2)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(savgol_app PRIVATE -mavx2 -mfma)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(savgol_app PRIVATE /arch:AVX2)
    endif()
elseif(ENABLE_SSE2)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(savgol_app PRIVATE -msse2)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(savgol_app PRIVATE /arch:SSE2)
    endif()
endif()

# Link the application to the library
target_link_libraries(savgol_app
    savgolFilter
)

# Ensure Cygwin runtime symbols are resolved for the application
if (CYGWIN)
    target_link_libraries(savgol_app PRIVATE c cygwin)
endif()

#==============================================================================
# Google Test Executable (OPTIONAL - build with -DBUILD_GTESTS=ON)
#==============================================================================

option(BUILD_GTESTS "Build Google Test unit tests" OFF)

if(BUILD_GTESTS)
    message(STATUS "Building Google Test unit tests")
    
    # Define the test executable
    add_executable(test_savgolFilter
        test_savgolFilter.cpp
    )

    # Set include directories for the test executable
    target_include_directories(test_savgolFilter PRIVATE
        ${CMAKE_SOURCE_DIR}/include/iterative
        ${CMAKE_SOURCE_DIR}/src/iterative  # For SIMD headers
        ${fff_SOURCE_DIR}
    )

    # Apply same SIMD flags as library
    if(ENABLE_NATIVE)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_savgolFilter PRIVATE -march=native -Wno-attributes)
        endif()
    elseif(ENABLE_AVX512)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_savgolFilter PRIVATE -mavx512f -mavx512dq -Wno-attributes)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(test_savgolFilter PRIVATE /arch:AVX512)
        endif()
    elseif(ENABLE_AVX2)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_savgolFilter PRIVATE -mavx2 -mfma -Wno-attributes)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(test_savgolFilter PRIVATE /arch:AVX2)
        endif()
    elseif(ENABLE_SSE2)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_savgolFilter PRIVATE -msse2 -Wno-attributes)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(test_savgolFilter PRIVATE /arch:SSE2)
        endif()
    endif()

    # Link the test executable to the library and GoogleTest
    target_link_libraries(test_savgolFilter
        savgolFilter
        gtest_main
        gmock_main
    )

    # Ensure Cygwin runtime symbols are resolved for the test executable
    if (CYGWIN)
        target_link_libraries(test_savgolFilter PRIVATE c cygwin)
    endif()

    # Register the tests
    include(GoogleTest)
    gtest_discover_tests(test_savgolFilter)
    
else()
    message(STATUS "Skipping Google Test unit tests (use -DBUILD_GTESTS=ON to enable)")
endif()