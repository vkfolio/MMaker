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
        // Lane hit-testing is the inverse of lane drawing -- at every scroll
        // offset and every lane height, not just the default pair. This is the
        // property that keeps a click on track 3 from editing track 5.
        int bad = 0;
        for (float scroll : {0.0f, 37.0f, 260.0f}) {
            for (float h : {mx::kTrackHeightMin, 84.0f, mx::kTrackHeightMax}) {
                mx::View v;
                v.scroll_y_px = scroll;
                v.track_h = h;
                for (int i = 0; i < 6; ++i) {
                    const float top = v.track_top(i);
                    if (top < mx::kRulerHeight) continue;   // scrolled above
                    if (v.track_at(top + 1.0f, 6) != i) ++bad;
                    if (v.track_at(top + v.track_h - 1.0f, 6) != i) ++bad;
                }
            }
        }
        check(bad == 0, "lane hit-test inverts lane layout at any scroll/height");

        mx::View v;
        check(v.track_at(4.0f, 6) == -1, "the ruler is not a lane");
        v.scroll_y_px = 500.0f;
        check(v.track_at(4.0f, 6) == -1, "the ruler is not a lane when scrolled");
    }

    {
        // The header column is drawn as elements offset by -scroll_y_px while
        // the lanes are drawn on a canvas at track_top(). Those are two
        // different code paths that must produce the same number, and a
        // disagreement is invisible until you scroll.
        mx::View v;
        v.scroll_y_px = 137.0f;
        v.track_h = 64.0f;
        int bad = 0;
        for (int i = 0; i < 8; ++i) {
            const float header_top = mx::kRulerHeight +
                                     static_cast<float>(i) * (v.track_h + mx::kTrackGap) -
                                     v.scroll_y_px;
            if (std::fabs(header_top - v.track_top(i)) > 0.001f) ++bad;
        }
        check(bad == 0, "track headers and lanes agree on every top edge");
    }

    {
        // Zooming track height keeps what is under the cursor under it, the
        // same contract horizontal zoom already has.
        mx::View v;
        v.track_h = 84.0f;
        v.scroll_y_px = 200.0f;
        const float cursor = 300.0f;
        const int before = v.track_at(cursor, 40);
        v.zoom_tracks_about(cursor, 1.5f);
        const int after = v.track_at(cursor, 40);
        check(before == after, "track-height zoom keeps the lane under the cursor");
    }

    {
        // Scrolling stops at the last track rather than into empty space.
        mx::View v;
        v.track_h = 84.0f;
        // A viewport shorter than the content, or there is nothing to clamp:
        // 4 lanes of 84 px come to 372 px, so 600 px of window shows them all.
        v.scroll_y_px = 100000.0f;
        v.clamp_y(4, 200.0f);
        const float content = v.content_height(4);
        check(std::fabs(v.scroll_y_px - (content - 200.0f)) < 0.001f,
              "vertical scroll clamps to the content");

        v.scroll_y_px = -50.0f;
        v.clamp_y(4, 200.0f);
        check(v.scroll_y_px == 0.0f, "vertical scroll cannot go above the top");

        // A session shorter than the window does not scroll at all.
        v.scroll_y_px = 400.0f;
        v.clamp_y(1, 900.0f);
        check(v.scroll_y_px == 0.0f, "a short session does not scroll");
    }

    std::printf("%s\n", std::string(66, '=').c_str());
    if (failures == 0) {
        std::printf("ALL PASS\n\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n\n", failures);
    return 1;
}
