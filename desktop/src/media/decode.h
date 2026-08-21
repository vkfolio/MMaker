// Decoding, on a worker, straight to the device's rate.
//
// The device-rate fork the plan forces here is taken this way: audio is
// resampled once, at decode, so the mixer never carries per-clip resampler
// state. That state is the part that is genuinely hard to keep click-free, and
// keeping it out of the real-time path is worth the cost -- which is that
// changing the default output device to a different rate invalidates every
// decoded buffer. `source_rate` is kept so that recovery is a re-decode rather
// than a re-download.

#pragma once

#include <filesystem>
#include <string>

#include "audio/graph.h"
#include "peaks.h"

namespace mx {

struct Media {
    BufferRef   buffer;
    Peaks       peaks;
    std::string error;          // empty on success
    double      seconds = 0.0;
    uint32_t    source_rate = 0;

    bool ok() const { return buffer != nullptr && error.empty(); }
};

/// Decodes to interleaved float at `device_rate`, then builds the pyramid.
/// Blocking; call it from a worker.
Media decode_file(const std::filesystem::path& path, uint32_t device_rate,
                  uint32_t channels = 2);

}  // namespace mx
