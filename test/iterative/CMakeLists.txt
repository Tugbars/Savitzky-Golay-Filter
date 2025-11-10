# =============================================================================
# Test Configuration for Savitzky-Golay Filter
# =============================================================================

# Common include directory
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/iterative)

# Determine which library to link based on build configuration
if(USE_PARALLEL_SAVGOL)
    set(SAVGOL_LIBRARY savgolFilterMT)
    message(STATUS "Tests will link against: savgolFilterMT (parallel)")
else()
    set(SAVGOL_LIBRARY savgolFilter)
    message(STATUS "Tests will link against: savgolFilter (sequential)")
endif()

# =============================================================================
# Common Configuration Function
# =============================================================================
function(configure_savgol_target TARGET_NAME)
    # Set include directories
    target_include_directories(${TARGET_NAME} PRIVATE
        ${INCLUDE_DIR}
        ${fff_SOURCE_DIR}
    )
    
    # Enable AVX/FMA
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_NAME} PRIVATE -mavx -mfma)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${TARGET_NAME} PRIVATE /arch:AVX)
    endif()
    
    # Link libraries
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${SAVGOL_LIBRARY}
        gtest_main
        gmock_main
    )
endfunction()

# =============================================================================
# Build Targets
# =============================================================================

# Application executable
add_executable(savgol_app main.cpp)
configure_savgol_target(savgol_app)

# Test executable
add_executable(test_savgolFilter main.cpp)
configure_savgol_target(test_savgolFilter)

# Register tests with GoogleTest
include(GoogleTest)
gtest_discover_tests(test_savgolFilter)