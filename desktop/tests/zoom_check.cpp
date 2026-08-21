// The timeline's coordinate contract.
//
// Two properties, both of which were broken in shipped builds and neither of
// which is visible in a screenshot until someone clicks:
//
//   round trip -- a click at window x must put the playhead at window x. Mouse
//                 events arrive in window coordinates and the view maps frames
//                 to canvas-relative pixels, so every handler has to subtract
//                 the canvas origin. Missing that offsets the playhead by the
//                 width of the sidebar and track headers.
//
//   anchoring  -- zooming about the cursor must leave the frame under the
//                 cursor under the cursor. Otherwise the content slides away
//                 while you zoom.

#include <cmath>
#include <cstdio>
#include <string>

#include "ui/arrangement.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) ++failures;
    std::printf("  %-46s %s\n", what, ok ? "PASS" : "FAIL");
}

}  // namespace

int main() {
    std::printf("\ntimeline coordinates\n%s\n", std::string(66, '=').c_str());

    // A canvas that does not start at the window's left edge -- which is the
    // whole point, since the sidebar and headers sit to its left.
    const float canvas_x = 380.0f;

    {
        int bad = 0;
        for (double fpp : {8.0, 64.0, 512.0, 4096.0, 65536.0}) {
            mx::View v;
            v.frames_per_pixel = fpp;
            v.scroll_frames = 1'000'000;
            for (float window_x : {380.0f, 500.0f, 913.0f, 1400.0f}) {
                const int64_t frame = v.frame_at(window_x - canvas_x);
                const float drawn = canvas_x + v.x_of(frame);
                if (std::fabs(drawn - window_x) > 1.0f) {
                    ++bad;
                    std::printf("     fpp %8.0f  clicked %.0f  drew %.1f\n",
                                fpp, window_x, drawn);
                }
            }
        }
        check(bad == 0, "click at x draws the playhead at x");
    }

    {
        mx::View v;
        v.frames_per_pixel = 512.0;
        v.scroll_frames = 100'000;
        const float cursor = 640.0f;
        int bad = 0;
        for (int i = 0; i < 12; ++i) {
            const int64_t before = v.frame_at(cursor);
            v.zoom_about(cursor, i % 2 ? 1.2 : 1.0 / 1.2, 4.0, 262144.0);
            const int64_t after = v.frame_at(cursor);
            if (std::llabs(after - before) >
                static_cast<int64_t>(v.frames_per_pixel) + 1)
                ++bad;
        }
        check(bad == 0, "zoom keeps the frame under the cursor");
    }

    {
        // Right-to-left drags must mean the same range as left-to-right.
        mx::Selection sel;
        sel.track = 1;
        sel.from = 96'000;
        sel.to = 24'000;
        check(sel.begin() == 24'000 && sel.end() == 96'000 && sel.active(),
              "a backwards drag selects the same range");
    }

    {
        // Lane hit-testing is the inverse of lane drawing.
        int bad = 0;
        for (int i = 0; i < 6; ++i) {
            const float top = mx::track_top(i);
            if (mx::track_at(top + 1.0f, 6) != i) ++bad;
            if (mx::track_at(top + mx::kTrackHeight - 1.0f, 6) != i) ++bad;
        }
        check(bad == 0, "lane hit-test inverts lane layout");
        check(mx::track_at(4.0f, 6) == -1, "the ruler is not a lane");
    }

    std::printf("%s\n", std::string(66, '=').c_str());
    if (failures == 0) {
        std::printf("ALL PASS\n\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n\n", failures);
    return 1;
}
