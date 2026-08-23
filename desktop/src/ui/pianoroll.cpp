#include "pianoroll.h"

#include <include/core/SkFont.h>
#include <include/core/SkPaint.h>

namespace mx {
namespace {

/// Width of the grab band at a note's right edge, in pixels. Fixed in pixels
/// rather than frames so a short note stays resizable when zoomed out and a
/// long one does not become mostly-handle when zoomed in.
constexpr float kResizeBand = 6.0f;

struct Palette {
    SkColor bg          = SkColorSetRGB(0x18, 0x1b, 0x23);
    SkColor row_white   = SkColorSetRGB(0x1e, 0x22, 0x2c);
    SkColor row_black   = SkColorSetRGB(0x17, 0x1a, 0x22);
    SkColor row_c       = SkColorSetRGB(0x22, 0x27, 0x33);
    SkColor grid        = SkColorSetRGB(0x26, 0x2b, 0x36);
    SkColor grid_bar    = SkColorSetRGB(0x39, 0x41, 0x51);
    SkColor ruler       = SkColorSetRGB(0x1b, 0x1e, 0x27);
    SkColor key_white   = SkColorSetRGB(0xe8, 0xea, 0xef);
    SkColor key_black   = SkColorSetRGB(0x23, 0x27, 0x31);
    SkColor key_edge    = SkColorSetRGB(0x11, 0x13, 0x19);
    SkColor note        = SkColorSetRGB(0x6b, 0x63, 0xe8);
    SkColor note_sel    = SkColorSetRGB(0x9b, 0x93, 0xff);
    SkColor note_edge   = SkColorSetRGB(0x3a, 0x33, 0xa0);
    SkColor text        = SkColorSetRGB(0x8d, 0x94, 0xa3);
    SkColor lyric       = SkColorSetRGB(0xff, 0xff, 0xff);
    SkColor playhead    = SkColorSetRGB(0xff, 0x6b, 0x4a);
};

const char* kNames[12] = {"C",  "C#", "D",  "D#", "E",  "F",
                          "F#", "G",  "G#", "A",  "A#", "B"};

}  // namespace

std::string pitch_name(uint8_t pitch) {
    // MIDI 60 is C4, so the octave number is pitch/12 - 1.
    const int octave = static_cast<int>(pitch) / 12 - 1;
    return std::string(kNames[pitch % 12]) + std::to_string(octave);
}

int note_at(const Score& score, const PianoView& view, float x, float y) {
    // Backwards, so the note drawn last -- on top -- is the one you grab.
    for (int i = static_cast<int>(score.notes.size()) - 1; i >= 0; --i) {
        const SkRect r = view.rect_of(score.notes[static_cast<size_t>(i)]);
        if (r.contains(x, y)) return i;
    }
    return -1;
}

NoteGrab grab_kind(const Score& score, const PianoView& view, int index, float x) {
    if (index < 0 || index >= static_cast<int>(score.notes.size()))
        return NoteGrab::Move;
    const SkRect r = view.rect_of(score.notes[static_cast<size_t>(index)]);
    // Never let the handle swallow the whole note: a note narrower than twice
    // the band would otherwise be impossible to move, only to resize.
    const float band = std::min(kResizeBand, r.width() * 0.4f);
    return (x >= r.right() - band) ? NoteGrab::ResizeEnd : NoteGrab::Move;
}

PianoStats draw_pianoroll(SkCanvas* c, const SkRect& bounds, const Score& score,
                          const PianoView& view, int selected, int64_t playhead,
                          int64_t frames_per_bar) {
    PianoStats stats;
    const Palette pal;

    SkPaint fill;
    fill.setAntiAlias(false);
    fill.setColor(pal.bg);
    c->drawRect(bounds, fill);

    c->save();
    c->clipRect(bounds);

    // --- pitch rows ---------------------------------------------------------
    // Only the rows that are actually on screen. An 88-key keyboard at 40 px a
    // row is 3500 px tall, so drawing all of it would be mostly waste.
    const float grid_left = bounds.left() + kKeyboardWidth;
    for (int p = kPitchMax; p >= kPitchMin; --p) {
        const auto pitch = static_cast<uint8_t>(p);
        const float top = bounds.top() + view.y_of(pitch);
        if (top + view.row_h < bounds.top() + kPianoRulerHeight) continue;
        if (top > bounds.bottom()) break;

        fill.setColor(pitch % 12 == 0 ? pal.row_c
                                      : (is_black_key(pitch) ? pal.row_black
                                                             : pal.row_white));
        c->drawRect(SkRect::MakeLTRB(grid_left, top, bounds.right(),
                                     top + view.row_h),
                    fill);
        ++stats.rows_drawn;
    }

    SkPaint line;
    line.setAntiAlias(false);
    line.setStrokeWidth(1.0f);

    SkFont font;
    font.setSize(9.0f);
    // A default SkFont has no typeface unless a font manager was installed --
    // true in the window, not true offscreen. Asking Skia to draw text without
    // one is a crash, not an empty string.
    const bool can_draw_text = font.getTypeface() != nullptr;
    SkPaint text_paint;
    text_paint.setAntiAlias(true);

    // --- bar grid -----------------------------------------------------------
    const int64_t fpb = std::max<int64_t>(1, frames_per_bar);
    const double px_per_bar = fpb / view.frames_per_pixel;
    const int64_t beats = 4;
    const double px_per_beat = px_per_bar / static_cast<double>(beats);

    const int64_t first_bar = view.scroll_frames / fpb;
    for (int64_t bar = first_bar;; ++bar) {
        const float x = grid_left + view.x_of(bar * fpb);
        if (x > bounds.right()) break;
        if (x >= grid_left) {
            line.setColor(pal.grid_bar);
            c->drawLine(std::floor(x) + 0.5f, bounds.top() + kPianoRulerHeight,
                        std::floor(x) + 0.5f, bounds.bottom(), line);
        }
        if (px_per_beat >= 6.0) {
            line.setColor(pal.grid);
            for (int64_t b = 1; b < beats; ++b) {
                const float bx = x + static_cast<float>(px_per_beat * b);
                if (bx < grid_left || bx > bounds.right()) continue;
                c->drawLine(std::floor(bx) + 0.5f,
                            bounds.top() + kPianoRulerHeight,
                            std::floor(bx) + 0.5f, bounds.bottom(), line);
            }
        }
    }

    // --- notes --------------------------------------------------------------
    SkPaint edge;
    edge.setAntiAlias(true);
    edge.setStyle(SkPaint::kStroke_Style);
    edge.setStrokeWidth(1.0f);
    edge.setColor(pal.note_edge);

    for (size_t i = 0; i < score.notes.size(); ++i) {
        const Note& n = score.notes[i];
        SkRect r = view.rect_of(n);
        r.offset(bounds.left() + kKeyboardWidth, bounds.top());
        if (r.right() < grid_left || r.left() > bounds.right()) continue;
        if (r.bottom() < bounds.top() || r.top() > bounds.bottom()) continue;

        r.fBottom -= 1.0f;
        fill.setAntiAlias(true);
        fill.setColor(static_cast<int>(i) == selected ? pal.note_sel : pal.note);
        c->drawRoundRect(r, 2.0f, 2.0f, fill);
        c->drawRoundRect(r, 2.0f, 2.0f, edge);
        fill.setAntiAlias(false);
        ++stats.notes_drawn;

        // The lyric belongs on the note, not in a separate lane: what a note
        // sings is a property of the note, and reading them apart is how you
        // put a syllable on the wrong pitch.
        if (can_draw_text && !n.lyric.empty() && r.width() > 14.0f &&
            r.height() > 8.0f) {
            c->save();
            c->clipRect(r);
            text_paint.setColor(pal.lyric);
            c->drawSimpleText(n.lyric.c_str(), n.lyric.size(),
                              SkTextEncoding::kUTF8, r.left() + 3.0f,
                              r.bottom() - (r.height() - 7.0f) * 0.5f, font,
                              text_paint);
            c->restore();
        }
    }

    // --- playhead -----------------------------------------------------------
    if (playhead >= 0) {
        const float x = grid_left + view.x_of(playhead);
        if (x >= grid_left && x <= bounds.right()) {
            line.setColor(pal.playhead);
            c->drawLine(std::floor(x) + 0.5f, bounds.top(),
                        std::floor(x) + 0.5f, bounds.bottom(), line);
        }
    }

    // --- keyboard gutter, drawn last so notes never cover it ----------------
    fill.setColor(pal.bg);
    c->drawRect(SkRect::MakeLTRB(bounds.left(), bounds.top(), grid_left,
                                 bounds.bottom()),
                fill);

    for (int p = kPitchMax; p >= kPitchMin; --p) {
        const auto pitch = static_cast<uint8_t>(p);
        const float top = bounds.top() + view.y_of(pitch);
        if (top + view.row_h < bounds.top() + kPianoRulerHeight) continue;
        if (top > bounds.bottom()) break;

        const bool black = is_black_key(pitch);
        fill.setColor(black ? pal.key_black : pal.key_white);
        const float right = black ? grid_left - 18.0f : grid_left;
        c->drawRect(SkRect::MakeLTRB(bounds.left(), top, right,
                                     top + view.row_h - 1.0f),
                    fill);

        // Label the Cs only. Every row labelled is unreadable at small heights
        // and redundant at large ones.
        if (can_draw_text && !black && pitch % 12 == 0 && view.row_h >= 9.0f) {
            const std::string name = pitch_name(pitch);
            text_paint.setColor(pal.key_edge);
            c->drawSimpleText(name.c_str(), name.size(), SkTextEncoding::kUTF8,
                              bounds.left() + 3.0f, top + view.row_h - 3.0f,
                              font, text_paint);
        }
    }

    // --- ruler, over everything --------------------------------------------
    fill.setColor(pal.ruler);
    c->drawRect(SkRect::MakeLTRB(bounds.left(), bounds.top(), bounds.right(),
                                 bounds.top() + kPianoRulerHeight),
                fill);
    line.setColor(pal.grid_bar);
    c->drawLine(bounds.left(), bounds.top() + kPianoRulerHeight - 0.5f,
                bounds.right(), bounds.top() + kPianoRulerHeight - 0.5f, line);

    if (can_draw_text) {
        text_paint.setColor(pal.text);
        for (int64_t bar = first_bar;; ++bar) {
            const float x = grid_left + view.x_of(bar * fpb);
            if (x > bounds.right()) break;
            if (x < grid_left) continue;
            const std::string label = std::to_string(bar + 1);
            c->drawSimpleText(label.c_str(), label.size(), SkTextEncoding::kUTF8,
                              x + 3.0f, bounds.top() + kPianoRulerHeight - 6.0f,
                              font, text_paint);
        }
    }

    c->restore();
    return stats;
}

}  // namespace mx
