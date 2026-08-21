#include "arrangement.h"

#include <chrono>
#include <format>

namespace mx {
namespace {

/// One vertical line per pixel column, from the pyramid level whose buckets are
/// nearest a pixel. Reading a coarser level and stretching it is what makes a
/// zoomed-out track cheap; reading a finer one and averaging would put the cost
/// back.
int64_t draw_waveform(SkCanvas* c, const SkRect& lane, float origin_x,
                      const Clip& clip, const Source& src, const View& view,
                      SkColor color, DrawStats& stats) {
    if (!src.loaded() || src.peaks.levels.empty()) return 0;
    const auto t_build = std::chrono::steady_clock::now();

    // View coordinates are canvas-relative; lane rects are in window
    // coordinates. Mixing the two silently shifts every waveform by the width
    // of whatever sits to the left of the canvas -- here the track headers,
    // which is why the audio played correctly while the drawing read from the
    // wrong place.
    const float x_from = std::max(lane.left(), origin_x + view.x_of(clip.start_frame));
    const float x_to   = std::min(lane.right(),
                                  origin_x + view.x_of(clip.start_frame + clip.length));
    if (x_to <= x_from) return 0;

    const int level_index = src.peaks.level_for(view.frames_per_pixel);
    const PeakLevel& level = src.peaks.levels[level_index];
    const int64_t ch = src.peaks.channels;

    const float mid  = lane.centerY();
    const float half = lane.height() * 0.42f;

    std::vector<SkPoint> lines;
    lines.reserve(static_cast<size_t>((x_to - x_from) * 2.0f) + 4);

    for (float x = std::floor(x_from); x < x_to; x += 1.0f) {
        // Timeline -> source frame -> bucket. Analytic the whole way; no
        // transform is ever pushed onto the canvas.
        const int64_t t0 = view.frame_at(x - origin_x);
        const int64_t t1 = view.frame_at(x + 1.0f - origin_x);
        const int64_t s0 = clip.source_offset + (t0 - clip.start_frame);
        const int64_t s1 = clip.source_offset + (t1 - clip.start_frame);
        if (s1 <= 0) continue;

        int64_t b0 = s0 / level.frames_per_bucket;
        int64_t b1 = std::max(b0 + 1, s1 / level.frames_per_bucket);
        b0 = std::clamp<int64_t>(b0, 0, level.buckets - 1);
        b1 = std::clamp<int64_t>(b1, b0 + 1, level.buckets);

        float lo = 0.0f, hi = 0.0f;
        for (int64_t b = b0; b < b1; ++b) {
            for (int64_t k = 0; k < ch; ++k) {
                lo = std::min(lo, level.lo[static_cast<size_t>(b * ch + k)]);
                hi = std::max(hi, level.hi[static_cast<size_t>(b * ch + k)]);
            }
        }

        // A silent column still deserves a mark, or a quiet passage looks like
        // a hole in the clip rather than quiet audio.
        float y0 = mid - hi * half;
        float y1 = mid - lo * half;
        if (y1 - y0 < 1.0f) { y0 = mid - 0.5f; y1 = mid + 0.5f; }

        lines.push_back(SkPoint::Make(x + 0.5f, y0));
        lines.push_back(SkPoint::Make(x + 0.5f, y1));
    }

    const auto t_submit = std::chrono::steady_clock::now();
    stats.build_ms += std::chrono::duration<double, std::milli>(t_submit - t_build).count();
    if (lines.empty()) return 0;

    SkPaint paint;
    paint.setColor(color);
    paint.setStrokeWidth(1.0f);
    paint.setAntiAlias(false);      // 1 px verticals; AA only blurs them
    c->drawPoints(SkCanvas::kLines_PointMode,
                  SkSpan<const SkPoint>(lines.data(), lines.size()), paint);
    stats.submit_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_submit).count();
    return static_cast<int64_t>(lines.size() / 2);
}

}  // namespace

DrawStats draw_arrangement(SkCanvas* c, const SkRect& bounds, Session& session,
                           const View& view, int64_t playhead, ClipId selected,
                           const Palette& pal) {
    DrawStats stats;

    SkPaint fill;
    fill.setAntiAlias(false);
    fill.setColor(pal.bg);
    c->drawRect(bounds, fill);

    const int track_count = static_cast<int>(session.tracks.size());

    // --- lanes ---------------------------------------------------------------
    for (int i = 0; i < track_count; ++i) {
        const float top = bounds.top() + track_top(i);
        if (top > bounds.bottom()) break;
        fill.setColor(i % 2 ? pal.lane_alt : pal.lane);
        c->drawRect(SkRect::MakeLTRB(bounds.left(), top, bounds.right(),
                                     top + kTrackHeight), fill);
    }

    // --- bar grid ------------------------------------------------------------
    // The grid is stepped in bars, and thinned out rather than drawn and
    // overdrawn: at 100 px per bar every bar gets a line, at 3 px per bar only
    // every 16th does.
    const int64_t fpb = std::max<int64_t>(1, session.frames_per_bar());
    const double px_per_bar = fpb / view.frames_per_pixel;
    int bar_step = 1;
    while (px_per_bar * bar_step < 24.0) bar_step *= 2;

    SkPaint line;
    line.setAntiAlias(false);
    line.setStrokeWidth(1.0f);

    SkFont font;
    font.setSize(11.0f);
    // A default SkFont carries no typeface unless something has installed a
    // font manager. That is true in the window, and not true when drawing
    // offscreen -- and asking Skia to draw text without one is a crash, not an
    // empty string. Labels are the only thing lost by skipping.
    const bool can_draw_text = font.getTypeface() != nullptr;
    SkPaint text_paint;
    text_paint.setColor(pal.text);
    text_paint.setAntiAlias(true);

    const int64_t first_bar = view.scroll_frames / fpb;
    const float lanes_bottom =
        bounds.top() + track_top(std::max(0, track_count)) - kTrackGap;

    for (int64_t bar = first_bar - (first_bar % bar_step);; bar += bar_step) {
        const float x = bounds.left() + view.x_of(bar * fpb);
        if (x > bounds.right()) break;
        if (x < bounds.left() - 1.0f) continue;

        const bool major = (bar % (bar_step * 4)) == 0;
        line.setColor(major ? pal.grid_bar : pal.grid);
        c->drawLine(std::floor(x) + 0.5f, bounds.top() + kRulerHeight,
                    std::floor(x) + 0.5f, lanes_bottom, line);
    }

    // --- ruler ---------------------------------------------------------------
    fill.setColor(pal.ruler);
    c->drawRect(SkRect::MakeLTRB(bounds.left(), bounds.top(), bounds.right(),
                                 bounds.top() + kRulerHeight), fill);

    for (int64_t bar = first_bar - (first_bar % bar_step);; bar += bar_step) {
        const float x = bounds.left() + view.x_of(bar * fpb);
        if (x > bounds.right()) break;
        if (x < bounds.left() - 40.0f) continue;

        line.setColor(pal.grid_bar);
        c->drawLine(std::floor(x) + 0.5f, bounds.top() + kRulerHeight - 7.0f,
                    std::floor(x) + 0.5f, bounds.top() + kRulerHeight, line);

        // Bars are 1-based on screen, as every DAW displays them.
        if (!can_draw_text) continue;
        const std::string label = std::to_string(bar + 1);
        c->drawSimpleText(label.c_str(), label.size(), SkTextEncoding::kUTF8,
                          std::floor(x) + 4.0f, bounds.top() + 13.0f, font, text_paint);
    }

    // --- clips ---------------------------------------------------------------
    // Drawn in passes by paint type -- every fill, then every waveform, then
    // every border, then every label -- rather than clip by clip.
    //
    // This is not tidiness. Interleaving antialiased round-rects with non-AA
    // point batches makes Skia break its batch and flush at each switch, and
    // the flush dominates: measured per-clip, drawPoints cost ~0.87 ms per
    // *call* almost independently of how many points it carried.
    struct Visible {
        const Clip*   clip;
        const Source* src;
        SkRect        lane;
        uint32_t      color;
    };
    std::vector<Visible> visible;
    visible.reserve(session.clips.size());

    for (const Clip& clip : session.clips) {
        const int index = [&] {
            for (int i = 0; i < track_count; ++i)
                if (session.tracks[i].id == clip.track_id) return i;
            return -1;
        }();
        if (index < 0) continue;

        const float x0 = bounds.left() + view.x_of(clip.start_frame);
        const float x1 = bounds.left() + view.x_of(clip.start_frame + clip.length);
        if (x1 < bounds.left() || x0 > bounds.right()) continue;

        const float top = bounds.top() + track_top(index);
        const SkRect lane = SkRect::MakeLTRB(std::max(bounds.left(), x0), top + 2.0f,
                                             std::min(bounds.right(), x1),
                                             top + kTrackHeight - 2.0f);
        if (lane.width() <= 0.0f) continue;

        visible.push_back(Visible{&clip, session.find_source(clip.source_id), lane,
                                  session.tracks[index].color});
    }

    // pass 1: bodies
    fill.setAntiAlias(false);
    for (const Visible& v : visible) {
        fill.setColor(SkColorSetARGB(0x38, (v.color >> 16) & 0xff,
                                     (v.color >> 8) & 0xff, v.color & 0xff));
        c->drawRoundRect(v.lane, 3.0f, 3.0f, fill);
    }

    // pass 2: waveforms
    for (const Visible& v : visible) {
        if (!v.src) continue;
        const SkColor wave = SkColorSetARGB(0xdd, (v.color >> 16) & 0xff,
                                            (v.color >> 8) & 0xff, v.color & 0xff);
        stats.columns_drawn +=
            draw_waveform(c, v.lane, bounds.left(), *v.clip, *v.src, view, wave, stats);
    }

    // pass 3: borders
    {
        SkPaint border;
        border.setStyle(SkPaint::kStroke_Style);
        border.setAntiAlias(true);
        for (const Visible& v : visible) {
            const bool sel = v.clip->id == selected;
            border.setStrokeWidth(sel ? 2.0f : 1.0f);
            border.setColor(sel ? SkColorSetRGB(0xff, 0xd5, 0x8a)
                                : SkColorSetARGB(0x88, 0xff, 0xff, 0xff));
            c->drawRoundRect(v.lane, 3.0f, 3.0f, border);
        }
    }

    // pass 4: labels
    text_paint.setColor(SkColorSetARGB(0xcc, 0xff, 0xff, 0xff));
    for (const Visible& v : visible) {
        if (!can_draw_text) break;
        if (v.lane.width() <= 40.0f || v.clip->name.empty()) continue;
        c->save();
        c->clipRect(v.lane);
        c->drawSimpleText(v.clip->name.c_str(), v.clip->name.size(),
                          SkTextEncoding::kUTF8, v.lane.left() + 5.0f,
                          v.lane.top() + 13.0f, font, text_paint);
        c->restore();
    }
    text_paint.setColor(pal.text);
    stats.clips_drawn = static_cast<int>(visible.size());

    // --- playhead ------------------------------------------------------------
    const float px = bounds.left() + view.x_of(playhead);
    if (px >= bounds.left() && px <= bounds.right()) {
        line.setColor(pal.playhead);
        line.setStrokeWidth(1.0f);
        c->drawLine(std::floor(px) + 0.5f, bounds.top(),
                    std::floor(px) + 0.5f, bounds.bottom(), line);

        SkPathBuilder head;
        head.moveTo(std::floor(px) - 5.0f, bounds.top());
        head.lineTo(std::floor(px) + 6.0f, bounds.top());
        head.lineTo(std::floor(px) + 0.5f, bounds.top() + 8.0f);
        head.close();
        fill.setColor(pal.playhead);
        fill.setAntiAlias(true);
        c->drawPath(head.detach(), fill);
        fill.setAntiAlias(false);
    }

    return stats;
}

}  // namespace mx
