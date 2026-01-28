# =============================================================================
# Savitzky-Golay Filter Library Build (src/iterative/CMakeLists.txt)
# =============================================================================

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/iterative)

# =============================================================================
# Core Filter Library (Sequential)
# =============================================================================

add_library(savgolFilter STATIC
    savgolFilter.c
)

target_include_directories(savgolFilter PUBLIC
    ${INCLUDE_DIR}
)

# =============================================================================
# Parallel Filter Library (Optional - requires OpenMP)
# =============================================================================

if(USE_PARALLEL_SAVGOL)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        add_library(savgolFilterMT STATIC
            savgolFilter.c
        )
        target_include_directories(savgolFilterMT PUBLIC ${INCLUDE_DIR})
        target_compile_definitions(savgolFilterMT PRIVATE USE_OPENMP)
        target_link_libraries(savgolFilterMT PUBLIC OpenMP::OpenMP_C)
        message(STATUS "Building parallel savgolFilterMT with OpenMP")
    else()
        message(WARNING "OpenMP not found, USE_PARALLEL_SAVGOL ignored")
    endif()
endif()

# =============================================================================
# Streaming Filter Library
# =============================================================================

add_library(savgolStream STATIC
    savgol_stream.c
)

target_include_directories(savgolStream PUBLIC
    ${INCLUDE_DIR}
)

# Link against sequential or parallel depending on config
if(USE_PARALLEL_SAVGOL AND TARGET savgolFilterMT)
    target_link_libraries(savgolStream PUBLIC savgolFilterMT)
else()
    target_link_libraries(savgolStream PUBLIC savgolFilter)
endif()

# =============================================================================
# Coefficient Export Tool
# =============================================================================

add_executable(savgol_export
    savgol_export.c
)

target_include_directories(savgol_export PRIVATE ${INCLUDE_DIR})
target_link_libraries(savgol_export PRIVATE savgolFilter)
if(UNIX)
    target_link_libraries(savgol_export PRIVATE m)
endif()

# =============================================================================
# 2D Filter Library
# =============================================================================

add_library(savgol2d STATIC
    savgol2d.c
)

target_include_directories(savgol2d PUBLIC ${INCLUDE_DIR})
if(UNIX)
    target_link_libraries(savgol2d PUBLIC m)
endif()