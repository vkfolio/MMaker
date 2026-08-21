#include "peaks.h"

#include <algorithm>

namespace mx {

Peaks build_peaks(const float* samples, int64_t frames, uint32_t channels) {
    Peaks p;
    p.channels = channels;
    if (!samples || frames <= 0 || channels == 0) return p;

    const int64_t ch = channels;

    // Level 0 straight from PCM.
    PeakLevel base;
    base.frames_per_bucket = Peaks::kBaseBucket;
    base.buckets = (frames + base.frames_per_bucket - 1) / base.frames_per_bucket;
    base.lo.assign(static_cast<size_t>(base.buckets * ch), 0.0f);
    base.hi.assign(static_cast<size_t>(base.buckets * ch), 0.0f);

    for (int64_t b = 0; b < base.buckets; ++b) {
        const int64_t from = b * base.frames_per_bucket;
        const int64_t to   = std::min(frames, from + base.frames_per_bucket);
        for (int64_t c = 0; c < ch; ++c) {
            float lo = 0.0f, hi = 0.0f;
            bool first = true;
            for (int64_t i = from; i < to; ++i) {
                const float v = samples[i * ch + c];
                if (first) { lo = hi = v; first = false; }
                else { lo = std::min(lo, v); hi = std::max(hi, v); }
            }
            base.lo[static_cast<size_t>(b * ch + c)] = lo;
            base.hi[static_cast<size_t>(b * ch + c)] = hi;
        }
    }
    p.levels.push_back(std::move(base));

    // Each level above is built from the one below, not from PCM: halving an
    // existing level is O(n) over a shrinking array, so the whole pyramid costs
    // barely more than level 0 alone.
    while (p.levels.back().buckets > 2) {
        const PeakLevel& src = p.levels.back();
        PeakLevel up;
        up.frames_per_bucket = src.frames_per_bucket * 2;
        up.buckets = (src.buckets + 1) / 2;
        up.lo.assign(static_cast<size_t>(up.buckets * ch), 0.0f);
        up.hi.assign(static_cast<size_t>(up.buckets * ch), 0.0f);

        for (int64_t b = 0; b < up.buckets; ++b) {
            const int64_t a0 = b * 2;
            const int64_t a1 = std::min(src.buckets - 1, a0 + 1);
            for (int64_t c = 0; c < ch; ++c) {
                const auto i0 = static_cast<size_t>(a0 * ch + c);
                const auto i1 = static_cast<size_t>(a1 * ch + c);
                up.lo[static_cast<size_t>(b * ch + c)] = std::min(src.lo[i0], src.lo[i1]);
                up.hi[static_cast<size_t>(b * ch + c)] = std::max(src.hi[i0], src.hi[i1]);
            }
        }
        p.levels.push_back(std::move(up));
    }

    return p;
}

}  // namespace mx
