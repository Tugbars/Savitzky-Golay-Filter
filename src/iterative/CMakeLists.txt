# =============================================================================
# Savitzky-Golay Filter Library Build
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

target_compile_options(savgolFilter PRIVATE
    -Wall -Wextra -O2
)

# =============================================================================
# Streaming Filter Library
# =============================================================================

add_library(savgolStream STATIC
    savgol_stream.c
)

target_include_directories(savgolStream PUBLIC
    ${INCLUDE_DIR}
)

target_link_libraries(savgolStream PUBLIC
    savgolFilter
)

target_compile_options(savgolStream PRIVATE
    -Wall -Wextra -O2
)

# =============================================================================
# Installation (Optional)
# =============================================================================

# install(TARGETS savgolFilter savgolStream
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
# )
# 
# install(FILES
#     ${INCLUDE_DIR}/savgolFilter.h
#     ${INCLUDE_DIR}/savgol_stream.h
#     DESTINATION include
# )