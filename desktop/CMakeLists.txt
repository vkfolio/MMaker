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
    src/app/crash.cpp
    src/app/document.cpp
    src/app/settings.cpp
    src/net/api.cpp
    src/net/cache.cpp
    src/net/http.cpp
    src/media/decode.cpp
    src/media/peaks.cpp
    src/ui/arrangement.cpp
    src/ui/pianoroll.cpp
    src/audio/midi_in.cpp
    src/audio/recorder.cpp
    src/ai/sidecar.cpp
    src/miniaudio_impl.c
)
target_include_directories(musicx PRIVATE src third_party)
find_package(CURL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(musicx PRIVATE mx_audio vik_components
                      CURL::libcurl nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(musicx PRIVATE ole32 winmm crypt32 dbghelp)

    # 8 MB of stack instead of the 1 MB default.
    #
    # A declarative UI builds its tree out of temporaries, and MSVC reserves
    # frame space for every branch of a function whether it runs or not. render()
    # plus an open modal was enough to overflow 1 MB, which shows up as
    # STATUS_FATAL_USER_CALLBACK_EXCEPTION -- a crash with no stack, inside a
    # panel that may not even be visible. Splitting functions helped and did not
    # remove the ceiling; this does, and costs nothing but address space.
    target_link_options(musicx PRIVATE /STACK:8388608)

    # Symbols in Release. A crash log full of raw addresses names
    # nothing, and /DEBUG costs only a .pdb beside the exe.
    target_compile_options(musicx PRIVATE $<$<CONFIG:Release>:/Zi>)
    target_link_options(musicx PRIVATE $<$<CONFIG:Release>:/DEBUG /OPT:REF /OPT:ICF>)
endif()

# --- the gate: audio correctness under editing ---------------------------
add_executable(zoom_check tests/zoom_check.cpp src/ui/arrangement.cpp
               src/media/peaks.cpp)
target_include_directories(zoom_check PRIVATE src third_party)
target_link_libraries(zoom_check PRIVATE vik_components)
add_test(NAME zoom_anchor COMMAND zoom_check)

add_executable(pianoroll_check tests/pianoroll_check.cpp src/ui/pianoroll.cpp)
target_include_directories(pianoroll_check PRIVATE src third_party)
target_link_libraries(pianoroll_check PRIVATE vik_components)
add_test(NAME pianoroll_coords COMMAND pianoroll_check)

# The editing contract: split arithmetic, trim bounds, id uniqueness. No UI and
# no audio -- session.h is header-only for exactly this reason.
add_executable(edit_check tests/edit_check.cpp src/media/peaks.cpp)
target_include_directories(edit_check PRIVATE src third_party)
target_link_libraries(edit_check PRIVATE vik_components)
add_test(NAME clip_editing COMMAND edit_check)

add_executable(mixer_test tests/mixer_test.cpp)
target_link_libraries(mixer_test PRIVATE mx_audio)

enable_testing()
add_test(NAME mixer_gate COMMAND mixer_test)

if(WIN32)
    # miniaudio's WASAPI backend.
    target_link_libraries(spike PRIVATE ole32 winmm)
endif()
