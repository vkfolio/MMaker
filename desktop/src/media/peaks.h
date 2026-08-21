// The waveform pyramid.
//
// Drawing a zoomed-out track from raw PCM means touching every sample every
// frame, so the peaks are precomputed once at decode into a mip pyramid: level
// 0 summarises 256 frames per bucket, each level above halves the resolution.
// The drawing code picks the level whose bucket is closest to one pixel.
//
// Size: two floats per bucket per channel, and the levels above level 0 sum to
// one more level, so ~0.135 MB per minute per channel. Negligible against the
// PCM it summarises.
//
// The floor matters and is easy to miss: a level-0 bucket is 5.3 ms at 48 kHz,
// so anything needing sample accuracy -- trimming a clip edge, drawing at full
// zoom -- must fall back to the raw PCM rather than reading the pyramid.

#pragma once

#include <cstdint>
#include <vector>

namespace mx {

struct PeakLevel {
    int64_t            frames_per_bucket = 256;
    std::vector<float> lo;    // interleaved by channel: [ch0 ch1 ch0 ch1 ...]
    std::vector<float> hi;
    int64_t            buckets = 0;
};

struct Peaks {
    uint32_t               channels = 2;
    std::vector<PeakLevel> levels;

    /// Level whose buckets are no larger than `frames_per_pixel`, so the
    /// drawing never has to average buckets down -- only up.
    int level_for(double frames_per_pixel) const {
        int best = 0;
        for (int i = 0; i < static_cast<int>(levels.size()); ++i) {
            if (static_cast<double>(levels[i].frames_per_bucket) <= frames_per_pixel)
                best = i;
            else
                break;
        }
        return best;
    }

    /// Below this, the pyramid is too coarse and the caller must read PCM.
    static constexpr int64_t kBaseBucket = 256;
};

/// Builds the pyramid from interleaved float PCM. Runs on a worker; never on
/// the UI thread and certainly never on the audio thread.
Peaks build_peaks(const float* samples, int64_t frames, uint32_t channels);

}  // namespace mx
