#include "arrangement.h"

#include <chrono>
#include <cstring>
#include <format>

namespace mx {
namespace {

/// One vertical line per pixel column, from the pyramid level whose buckets are
/// nearest a pixel. Reading a coarser level and stretching it is what makes a
/// zoomed-out track cheap; reading a finer one and averaging would put the cost
/// back.
/// One vertical line per pixel column, per channel.
///
/// Two channels drawn in their own bands rather than summed into one shape.
/// ACE Studio draws stereo this way and it is not decoration: a summed
/// waveform hides the thing you most want to see on a stem -- whether the two
/// sides differ at all.
int64_t draw_waveform(SkCanvas* c, const SkRect& body, float origin_x,
                      const Clip& clip, const Source& src, const View& view,
                      SkColor color, DrawStats& stats) {
    if (!src.loaded() || src.peaks.levels.empty()) return 0;
    const auto t_build = std::chrono::steady_clock::now();

    // View coordinates are canvas-relative; lane rects are in window
    // coordinates. Mixing the two silently shifts every waveform by the width
    // of whatever sits to the left of the canvas.
    const float x_from = std::max(body.left(), origin_x + view.x_of(clip.start_frame));
    const float x_to   = std::min(body.right(),
                                  origin_x + view.x_of(clip.start_frame + clip.length));
    if (x_to <= x_from) return 0;

    const int level_index = src.peaks.level_for(view.frames_per_pixel);
    const PeakLevel& level = src.peaks.levels[level_index];
    const int64_t channels = std::max<int64_t>(1, src.peaks.channels);
    const int64_t shown = std::min<int64_t>(channels, 2);

    const float band = body.height() / static_cast<float>(shown);
    std::vector<SkPoint> lines;
    lines.reserve(static_cast<size_t>((x_to - x_from) * 2.0f * shown) + 8);

    for (int64_t ch = 0; ch < shown; ++ch) {
        const float mid = body.top() + band * (ch + 0.5f);
        const float half = band * 0.44f;

        for (float x = std::floor(x_from); x < x_to; x += 1.0f) {
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
                const auto i = static_cast<size_t>(b * channels + ch);
                lo = std::min(lo, level.lo[i]);
                hi = std::max(hi, level.hi[i]);
            }

            // A silent column still deserves a mark, or a quiet passage looks
            // like a hole in the clip rather than quiet audio.
            float y0 = mid - hi * half;
            float y1 = mid - lo * half;
            if (y1 - y0 < 1.0f) { y0 = mid - 0.5f; y1 = mid + 0.5f; }

            lines.push_back(SkPoint::Make(x + 0.5f, y0));
            lines.push_back(SkPoint::Make(x + 0.5f, y1));
        }
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
                           const Selection& range, const Palette& pal) {
    DrawStats stats;

    SkPaint fill;
    fill.setAntiAlias(false);
    fill.setColor(pal.bg);
    c->drawRect(bounds, fill);

    const int track_count = static_cast<int>(session.tracks.size());

    // --- lanes ---------------------------------------------------------------
    for (int i = 0; i < track_count; ++i) {
        const float top = bounds.top() + view.track_top(i);
        if (top > bounds.bottom()) break;
        if (top + view.track_h < bounds.top() + kRulerHeight) continue;
        fill.setColor(i % 2 ? pal.lane_alt : pal.lane);
        c->drawRect(SkRect::MakeLTRB(bounds.left(), top, bounds.right(),
                                     top + view.track_h), fill);
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
        bounds.top() + view.track_top(std::max(0, track_count)) - kTrackGap;

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

        const float top = bounds.top() + view.track_top(index);
        const SkRect lane = SkRect::MakeLTRB(std::max(bounds.left(), x0), top + 2.0f,
                                             std::min(bounds.right(), x1),
                                             top + view.track_h - 2.0f);
        if (lane.width() <= 0.0f) continue;

        visible.push_back(Visible{&clip, session.find_source(clip.source_id), lane,
                                  session.tracks[index].color});
    }

    // A clip is a solid coloured block with a title strip along the top and the
    // waveform below it, the way ACE Studio draws them. The title travels with
    // the clip rather than living in the track header, which is what makes a
    // lane of several clips readable -- each one says what produced it.
    const float kTitle = 15.0f;

    // pass 1: bodies
    fill.setAntiAlias(false);
    for (const Visible& v : visible) {
        const uint8_t r = (v.color >> 16) & 0xff, g = (v.color >> 8) & 0xff,
                      b = v.color & 0xff;
        // Body first, then a brighter strip on top of it.
        fill.setColor(SkColorSetARGB(0x55, r, g, b));
        c->drawRect(v.lane, fill);
        fill.setColor(SkColorSetARGB(0xff, r, g, b));
        c->drawRect(SkRect::MakeLTRB(v.lane.left(), v.lane.top(),
                                     v.lane.right(),
                                     std::min(v.lane.bottom(), v.lane.top() + kTitle)),
                    fill);
    }

    // pass 2: waveforms, below the title strip
    for (const Visible& v : visible) {
        if (!v.src) continue;
        SkRect body = v.lane;
        body.fTop = std::min(body.fBottom, body.fTop + kTitle);
        if (body.height() <= 2.0f) continue;
        // Near-black over the clip colour, as in the reference: the waveform
        // reads as a shape cut out of the block rather than a line drawn on it.
        const SkColor wave = SkColorSetARGB(0xdd, 0x0d, 0x0f, 0x14);
        stats.columns_drawn +=
            draw_waveform(c, body, bounds.left(), *v.clip, *v.src, view, wave, stats);
    }

    // pass 3: borders
    {
        SkPaint border;
        border.setStyle(SkPaint::kStroke_Style);
        border.setAntiAlias(true);
        for (const Visible& v : visible) {
            const bool sel = v.clip->id == selected;
            if (!sel) continue;          // only the selected clip is outlined
            border.setStrokeWidth(2.0f);
            border.setColor(SkColorSetRGB(0xff, 0xd5, 0x8a));
            c->drawRect(v.lane, border);
        }
    }

    // pass 4: labels
    text_paint.setColor(SkColorSetARGB(0xcc, 0xff, 0xff, 0xff));
    // Dark text on the bright title strip, not white on colour: the strips are
    // fully saturated, and white on yellow is unreadable.
    text_paint.setColor(SkColorSetARGB(0xee, 0x10, 0x12, 0x18));
    for (const Visible& v : visible) {
        if (!can_draw_text) break;
        if (v.lane.width() <= 40.0f) continue;
        c->save();
        c->clipRect(SkRect::MakeLTRB(v.lane.left(), v.lane.top(),
                                     v.lane.right() - 3.0f, v.lane.top() + kTitle));
        if (!v.clip->name.empty())
            c->drawSimpleText(v.clip->name.c_str(), v.clip->name.size(),
                              SkTextEncoding::kUTF8, v.lane.left() + 5.0f,
                              v.lane.top() + 11.0f, font, text_paint);

        // Version badge, right-aligned in the title strip. Only once there is
        // more than one: a clip that has never been through a tool has nothing
        // to compare against, and a permanent "v1/1" is noise.
        if (v.src && v.src->versions.size() > 1 && v.lane.width() > 96.0f) {
            const std::string badge = "v" + std::to_string(v.src->current + 1) +
                                      "/" + std::to_string(v.src->versions.size());
            const float width = font.measureText(badge.c_str(), badge.size(),
                                                 SkTextEncoding::kUTF8);
            const SkRect pill = SkRect::MakeLTRB(
                v.lane.right() - width - 12.0f, v.lane.top() + 2.0f,
                v.lane.right() - 3.0f, v.lane.top() + kTitle - 2.0f);
            SkPaint chip;
            chip.setAntiAlias(true);
            chip.setColor(SkColorSetARGB(0x66, 0x10, 0x12, 0x18));
            c->drawRoundRect(pill, 3.0f, 3.0f, chip);
            c->drawSimpleText(badge.c_str(), badge.size(), SkTextEncoding::kUTF8,
                              pill.left() + 4.5f, v.lane.top() + 11.0f, font,
                              text_paint);
        }
        c->restore();
    }
    text_paint.setColor(pal.text);
    stats.clips_drawn = static_cast<int>(visible.size());

    // --- empty state ---------------------------------------------------------
    // The reference says "Double click to start, or select a template" in the
    // middle of an empty canvas. Same idea, different next step: here the
    // content comes from the pod, so the sentence points at that.
    if (visible.empty() && can_draw_text) {
        SkFont big;
        big.setSize(13.0f);
        const char* line = session.tracks.empty()
            ? "Pick a project on the left, or start musicx with a folder of stems"
            : "No audio on the timeline yet";
        SkPaint hint;
        hint.setAntiAlias(true);
        hint.setColor(SkColorSetARGB(0x88, 0x8d, 0x94, 0xa3));
        const float width = big.measureText(line, std::strlen(line),
                                            SkTextEncoding::kUTF8);
        c->drawSimpleText(line, std::strlen(line), SkTextEncoding::kUTF8,
                          bounds.centerX() - width * 0.5f, bounds.centerY(),
                          big, hint);
    }

    // --- selection ----------------------------------------------------------
    // Over the waveform rather than under it: what is selected has to stay
    // legible against whatever it is covering, and a wash beneath the clip body
    // disappears entirely on a dense stem.
    if (range.active()) {
        int index = -1;
        for (int i = 0; i < track_count; ++i)
            if (session.tracks[i].id == range.track) { index = i; break; }
        if (index >= 0) {
            const float x0 = bounds.left() + view.x_of(range.begin());
            const float x1 = bounds.left() + view.x_of(range.end());
            const float top = bounds.top() + view.track_top(index);
            const SkRect box = SkRect::MakeLTRB(
                std::max(bounds.left(), x0), top,
                std::min(bounds.right(), x1), top + view.track_h);
            if (box.width() > 0.0f) {
                fill.setColor(pal.selection);
                fill.setAntiAlias(false);
                c->drawRect(box, fill);

                // Hard edges: a soft wash alone makes it impossible to tell
                // exactly what will be regenerated.
                line.setColor(pal.selection_edge);
                line.setStrokeWidth(1.0f);
                if (x0 >= bounds.left())
                    c->drawLine(std::floor(x0) + 0.5f, top,
                                std::floor(x0) + 0.5f, top + view.track_h, line);
                if (x1 <= bounds.right())
                    c->drawLine(std::floor(x1) + 0.5f, top,
                                std::floor(x1) + 0.5f, top + view.track_h, line);
            }
        }
    }

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
