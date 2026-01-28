# =============================================================================
# Test Configuration for Savitzky-Golay Filter
# =============================================================================

# Directories
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/iterative)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/iterative)

# Determine which library to link based on build configuration
# Fall back to sequential if parallel not available
if(USE_PARALLEL_SAVGOL AND TARGET savgolFilterMT)
    set(SAVGOL_LIBRARY savgolFilterMT)
    message(STATUS "Tests will link against: savgolFilterMT (parallel)")
elseif(TARGET savgolFilter)
    set(SAVGOL_LIBRARY savgolFilter)
    message(STATUS "Tests will link against: savgolFilter (sequential)")
else()
    message(FATAL_ERROR "No savgolFilter library target found. Build src/iterative first.")
endif()

# =============================================================================
# Streaming Library (build if not already defined elsewhere)
# =============================================================================

if(NOT TARGET savgolStream)
    add_library(savgolStream STATIC
        ${SRC_DIR}/savgol_stream.c
    )
    target_include_directories(savgolStream PUBLIC ${INCLUDE_DIR})
    target_link_libraries(savgolStream PUBLIC ${SAVGOL_LIBRARY})
endif()

# =============================================================================
# Standalone Test Executables
# =============================================================================

# Core unit tests
add_executable(test_savgol test_savgol.c)
target_include_directories(test_savgol PRIVATE ${INCLUDE_DIR})
target_link_libraries(test_savgol PRIVATE ${SAVGOL_LIBRARY})
if(UNIX)
    target_link_libraries(test_savgol PRIVATE m)
endif()

# Streaming filter tests  
add_executable(test_savgol_stream test_savgol_stream.c)
target_include_directories(test_savgol_stream PRIVATE ${INCLUDE_DIR})
target_link_libraries(test_savgol_stream PRIVATE savgolStream ${SAVGOL_LIBRARY})
if(UNIX)
    target_link_libraries(test_savgol_stream PRIVATE m)
endif()

# Benchmark / demo application
add_executable(test_savgol_main test_savgol_main.c)
target_include_directories(test_savgol_main PRIVATE ${INCLUDE_DIR})
target_link_libraries(test_savgol_main PRIVATE ${SAVGOL_LIBRARY})
if(UNIX)
    target_link_libraries(test_savgol_main PRIVATE m)
endif()

# =============================================================================
# CTest Registration
# =============================================================================

add_test(NAME SavgolCoreTests COMMAND test_savgol)
add_test(NAME SavgolStreamTests COMMAND test_savgol_stream)
add_test(NAME SavgolBenchmark COMMAND test_savgol_main)