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

# --- Phase 0 spike, kept as running evidence rather than as a product ---
add_executable(spike
    spike/main.cpp
    spike/miniaudio_impl.c
)
target_include_directories(spike PRIVATE third_party)
target_link_libraries(spike PRIVATE vik_components)

# --- the audio engine, deliberately free of any UI dependency -------------
# Keeping the mixer linkable on its own is what makes the Phase 1 gate a test
# rather than a listening session.
add_library(mx_audio STATIC
    src/audio/mixer.cpp
)
target_include_directories(mx_audio PUBLIC src third_party)

# --- the app --------------------------------------------------------------
add_executable(musicx
    src/app_main.cpp
    src/audio/device.cpp
    src/media/bounce.cpp
    src/net/api.cpp
    src/net/cache.cpp
    src/net/http.cpp
    src/media/decode.cpp
    src/media/peaks.cpp
    src/ui/arrangement.cpp
    src/miniaudio_impl.c
)
target_include_directories(musicx PRIVATE src third_party)
find_package(CURL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(musicx PRIVATE mx_audio vik_components
                      CURL::libcurl nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(musicx PRIVATE ole32 winmm)
endif()

# --- the gate: audio correctness under editing ---------------------------
add_executable(mixer_test tests/mixer_test.cpp)
target_link_libraries(mixer_test PRIVATE mx_audio)

enable_testing()
add_test(NAME mixer_gate COMMAND mixer_test)

if(WIN32)
    # miniaudio's WASAPI backend.
    target_link_libraries(spike PRIVATE ole32 winmm)
endif()
