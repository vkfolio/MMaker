# musicX Studio -- Phase 0 spike.
#
# Consumes vikui_cpp in-tree. VIKUI_DIR can point elsewhere; the default is the
# sibling checkout. vikui pins MSVC 14.51 / VS 2026, so use its own preset
# toolchain rather than whatever cmake picks up.
cmake_minimum_required(VERSION 3.24)
project(musicx_studio CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VIKUI_DIR "D:/osp/musicX/vikui_cpp" CACHE PATH "vikui_cpp checkout")
if(NOT EXISTS "${VIKUI_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "vikui_cpp not found at ${VIKUI_DIR} -- set -DVIKUI_DIR=")
endif()

# vikui declares its examples/gallery/tests unconditionally, so EXCLUDE_FROM_ALL
# is what keeps them out of this build -- only the `spike` target is asked for.
set(VIKUI_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${VIKUI_DIR}" "${CMAKE_BINARY_DIR}/vikui" EXCLUDE_FROM_ALL)

add_executable(spike
    src/main.cpp
    src/miniaudio_impl.c
)
target_include_directories(spike PRIVATE third_party)
target_link_libraries(spike PRIVATE vik_components)

if(WIN32)
    # miniaudio's WASAPI backend.
    target_link_libraries(spike PRIVATE ole32 winmm)
endif()
