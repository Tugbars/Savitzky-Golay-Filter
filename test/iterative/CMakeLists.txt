# Define the application executable
add_executable(savgol_app
    main.cpp
)

# Set include directories for the application
target_include_directories(savgol_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include/iterative
)

# Enable AVX for the application (optional)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(savgol_app PRIVATE -mavx -mfma)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(savgol_app PRIVATE /arch:AVX)
    target_link_options(savgol_app PRIVATE /NODEFAULTLIB:libcmt.lib)
endif()

# Link the application to the library
target_link_libraries(savgol_app
    savgolFilter
    gtest_main
    gmock_main
)

# Ensure Cygwin runtime symbols are resolved for the application
if (CYGWIN)
    target_link_libraries(savgol_app PRIVATE c cygwin)
endif()

# Define the test executable
add_executable(test_savgolFilter
    test_savgolFilter.cpp
)

# Set include directories for the test executable
target_include_directories(test_savgolFilter PRIVATE
    ${CMAKE_SOURCE_DIR}/include/iterative
    ${fff_SOURCE_DIR}
)

# Enable AVX for the test executable (optional)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_savgolFilter PRIVATE -mavx -mfma)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_savgolFilter PRIVATE /arch:AVX)
    target_link_options(test_savgolFilter PRIVATE /NODEFAULTLIB:libcmt.lib)
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
