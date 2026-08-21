#include <cstdio>
#include <cmath>
#include "ui/arrangement.h"
int main() {
    mx::View v;
    v.frames_per_pixel = 512.0;
    v.scroll_frames = 100000;
    const float cursor = 640.0f;
    int bad = 0;
    for (int i = 0; i < 12; ++i) {
        const int64_t before = v.frame_at(cursor);
        v.zoom_about(cursor, i % 2 ? 1.2 : 1.0 / 1.2, 4.0, 262144.0);
        const int64_t after = v.frame_at(cursor);
        const int64_t drift = std::llabs(after - before);
        // Within one pixel's worth of frames is exact for integer scrolling.
        if (drift > static_cast<int64_t>(v.frames_per_pixel) + 1) ++bad;
        std::printf("  fpp %9.1f  frame under cursor %8lld -> %8lld  drift %lld\n",
                    v.frames_per_pixel, (long long)before, (long long)after,
                    (long long)drift);
    }
    std::printf("%s\n", bad ? "FAIL cursor anchor drifts" : "PASS cursor stays anchored");
    return bad ? 1 : 0;
}
