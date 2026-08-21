// The arrangement surface.
//
// One canvas for content, real elements for chrome. Two rules from the plan are
// load-bearing here and neither is negotiable later:
//
//   * No matrix. Zoom is anisotropic -- horizontal only -- and a scaled CTM
//     blurs the 1 px playhead and re-rasterises every glyph per zoom step. Pixel
//     rects are computed analytically instead.
//   * Hit-testing is the inverse function, never a test against drawn shapes.
//     That is precisely what makes one canvas cheaper than thousands of
//     elements: nothing is retained to hit-test against.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "session.h"

#include <include/core/SkCanvas.h>
#include <include/core/SkFont.h>
#include <include/core/SkPaint.h>
#include <include/core/SkPath.h>
#include <include/core/SkPathBuilder.h>
#include <include/core/SkPoint.h>

namespace mx {

constexpr float kRulerHeight = 28.0f;
constexpr float kTrackHeight = 84.0f;
constexpr float kTrackGap    = 2.0f;

/// The one place the timeline<->pixel mapping is defined. Everything else --
/// drawing, hit-testing, scroll clamping -- goes through these two functions,
/// so they cannot drift apart.
struct View {
    double  frames_per_pixel = 512.0;
    int64_t scroll_frames    = 0;

    float x_of(int64_t frame) const {
        return static_cast<float>((frame - scroll_frames) / frames_per_pixel);
    }
    int64_t frame_at(float x) const {
        return scroll_frames + static_cast<int64_t>(std::llround(x * frames_per_pixel));
    }

    /// Frames per pixel that fits `frames` into `width_px`, with a little air.
    static double fit(int64_t frames, float width_px) {
        if (frames <= 0 || width_px <= 1.0f) return 512.0;
        return static_cast<double>(frames) / (width_px * 0.96);
    }

    /// Zoom about a fixed pixel, so the frame under the cursor stays put --
    /// the behaviour that makes wheel-zoom feel attached to the content.
    void zoom_about(float x, double factor, double min_fpp, double max_fpp) {
        const int64_t anchor = frame_at(x);
        frames_per_pixel = std::clamp(frames_per_pixel * factor, min_fpp, max_fpp);
        scroll_frames = anchor - static_cast<int64_t>(std::llround(x * frames_per_pixel));
        if (scroll_frames < 0) scroll_frames = 0;
    }
};

inline float track_top(int index) {
    return kRulerHeight + static_cast<float>(index) * (kTrackHeight + kTrackGap);
}

/// Which track lane a y coordinate falls in, or -1. The inverse function again.
inline int track_at(float y, int track_count) {
    if (y < kRulerHeight) return -1;
    const int i = static_cast<int>((y - kRulerHeight) / (kTrackHeight + kTrackGap));
    return (i >= 0 && i < track_count) ? i : -1;
}

/// A time range on one track. This is what every AI tool acts on -- the plan
/// calls it "the defining interaction" -- so it is a first-class thing the
/// drawing and the tools both read, not state hidden inside a mouse handler.
struct Selection {
    TrackId track = 0;
    int64_t from  = 0;
    int64_t to    = 0;

    bool active() const { return track != 0 && to > from; }
    void clear() { track = 0; from = to = 0; }
    /// Ordered bounds, so a right-to-left drag means the same as left-to-right.
    int64_t begin() const { return std::min(from, to); }
    int64_t end() const { return std::max(from, to); }
};

struct Palette {
    SkColor bg        = SkColorSetRGB(0x14, 0x16, 0x1d);
    SkColor lane      = SkColorSetRGB(0x1a, 0x1d, 0x26);
    SkColor lane_alt  = SkColorSetRGB(0x17, 0x1a, 0x22);
    SkColor grid      = SkColorSetRGB(0x26, 0x2b, 0x36);
    SkColor grid_bar  = SkColorSetRGB(0x35, 0x3c, 0x4a);
    SkColor ruler     = SkColorSetRGB(0x1b, 0x1e, 0x27);
    SkColor text      = SkColorSetRGB(0x8d, 0x94, 0xa3);
    SkColor playhead  = SkColorSetRGB(0xff, 0x6b, 0x4a);
    SkColor selection = SkColorSetARGB(0x38, 0xff, 0xd5, 0x8a);
    SkColor selection_edge = SkColorSetARGB(0xcc, 0xff, 0xd5, 0x8a);
};

struct DrawStats {
    int64_t columns_drawn = 0;
    int     clips_drawn   = 0;
    // Split so the cost can be attributed rather than guessed at: `build` is
    // our own per-column arithmetic, `submit` is what Skia does with it.
    double  build_ms  = 0.0;
    double  submit_ms = 0.0;
};

/// Draws the ruler, lanes, clips and playhead.
///
/// `playhead` is the *heard* position, not the rendered one -- drawing the
/// render position puts the line visibly ahead of the sound.
DrawStats draw_arrangement(SkCanvas* c, const SkRect& bounds, Session& session,
                           const View& view, int64_t playhead, ClipId selected,
                           const Selection& range = {}, const Palette& pal = {});

}  // namespace mx
