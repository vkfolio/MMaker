// Offline bounce: render the session to a file, faster than real time.
//
// This is deliberately the *same* mixer the device drives, not a second
// rendering path. The plan calls export "a second driver that constrains the
// whole RT design", and that only holds if both drivers share the code -- a
// separate offline mixer would be free to disagree with what people hear, and
// would prove nothing about it.
//
// The only difference is that offline has no listener to hide a discontinuity
// from, so the ramps that exist for that purpose start settled.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "audio/mixer.h"
#include "session.h"

namespace mx {

struct BounceResult {
    bool        ok = false;
    std::string error;
    int64_t     frames = 0;
    double      seconds = 0.0;
    double      render_ms = 0.0;      // wall clock spent mixing
    double      realtime_ratio = 0.0; // >1 means faster than real time
    float       peak = 0.0f;
    bool        clipped = false;
};

/// Mixes the whole session into a 32-bit float WAV at `path`.
///
/// `tail_seconds` keeps any fade-out from being truncated at the last clip's
/// end. Nothing is normalised: an export that quietly changes the level is an
/// export you cannot compare against anything.
BounceResult bounce(Session& session, const std::filesystem::path& path,
                    double tail_seconds = 0.25);

}  // namespace mx
