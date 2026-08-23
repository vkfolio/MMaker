// The note editor.
//
// Second surface, same contract as the arrangement (ui/arrangement.h): one
// canvas for content, real elements for chrome, analytic rects, and hit-testing
// by inverse function rather than against retained shapes.
//
// The one rule carried over from the vertical-scroll work, and it is the whole
// reason `View` looks the way it does now: **both mappings live in the struct,
// as members.** A free function over a constant row height was fine right up
// until something scrolled, and then every caller that forgot the offset
// silently edited the wrong row. Here that would mean typing a lyric onto the
// wrong note, which is worse: it is a data change, not a display glitch.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "session.h"

#include <include/core/SkCanvas.h>
#include <include/core/SkRect.h>

namespace mx {

constexpr float kPianoRulerHeight = 22.0f;
constexpr float kKeyboardWidth    = 56.0f;
constexpr float kRowHeightMin     = 6.0f;
constexpr float kRowHeightMax     = 40.0f;

constexpr uint8_t kPitchMin = 21;   // A0
constexpr uint8_t kPitchMax = 108;  // C8

/// Is this pitch a black key? The pattern repeats every octave, so one table.
inline bool is_black_key(uint8_t pitch) {
    switch (pitch % 12) {
        case 1: case 3: case 6: case 8: case 10: return true;
        default: return false;
    }
}

/// Note name for the keyboard gutter, e.g. "C4" or "A#3".
std::string pitch_name(uint8_t pitch);

/// The piano roll's coordinate system.
///
/// Time uses exactly the same arithmetic as `View`, so zooming feels identical
/// in both surfaces and a frame means the same thing in each. Pitch is the new
/// axis: high pitches at the top, which is the one convention every editor
/// shares and the one that is instantly wrong if inverted.
struct PianoView {
    double  frames_per_pixel = 512.0;
    int64_t scroll_frames    = 0;
    float   row_h            = 14.0f;
    float   scroll_y_px      = 0.0f;

    float x_of(int64_t frame) const {
        return static_cast<float>((frame - scroll_frames) / frames_per_pixel);
    }
    int64_t frame_at(float x) const {
        return scroll_frames + static_cast<int64_t>(std::llround(x * frames_per_pixel));
    }

    /// Top edge of a pitch's row. Pitch counts up, y counts down.
    float y_of(uint8_t pitch) const {
        const int from_top = static_cast<int>(kPitchMax) - static_cast<int>(pitch);
        return kPianoRulerHeight + static_cast<float>(from_top) * row_h - scroll_y_px;
    }

    /// The inverse: which pitch a y coordinate lands on, clamped to the
    /// keyboard. Returns 0 above the ruler, which is not a note row.
    uint8_t pitch_at(float y) const {
        if (y < kPianoRulerHeight) return 0;
        const float local = y - kPianoRulerHeight + scroll_y_px;
        const int from_top = static_cast<int>(std::floor(local / row_h));
        const int pitch = static_cast<int>(kPitchMax) - from_top;
        return static_cast<uint8_t>(std::clamp(pitch, static_cast<int>(kPitchMin),
                                               static_cast<int>(kPitchMax)));
    }

    float content_height() const {
        return kPianoRulerHeight +
               static_cast<float>(kPitchMax - kPitchMin + 1) * row_h;
    }

    void clamp_y(float viewport_h) {
        const float max_scroll = std::max(0.0f, content_height() - viewport_h);
        scroll_y_px = std::clamp(scroll_y_px, 0.0f, max_scroll);
    }

    /// The rectangle a note occupies, in surface coordinates.
    SkRect rect_of(const Note& n) const {
        const float x0 = x_of(n.start);
        const float x1 = x_of(n.start + n.length);
        const float y0 = y_of(n.pitch);
        return SkRect::MakeLTRB(x0, y0, std::max(x1, x0 + 2.0f), y0 + row_h);
    }

    /// Centre the view on the notes there are, or on middle C if there are
    /// none. Opening onto an empty region of an 88-key keyboard reads as a
    /// broken editor.
    void frame_notes(const Score& score, float viewport_h) {
        int lo = 60, hi = 60;
        if (!score.notes.empty()) {
            lo = hi = score.notes.front().pitch;
            for (const auto& n : score.notes) {
                lo = std::min(lo, static_cast<int>(n.pitch));
                hi = std::max(hi, static_cast<int>(n.pitch));
            }
        }
        const float mid = (y_of(static_cast<uint8_t>(hi)) +
                           y_of(static_cast<uint8_t>(lo)) + row_h) * 0.5f;
        scroll_y_px += mid - viewport_h * 0.5f;
        clamp_y(viewport_h);
    }
};

/// Which note is under a point, or -1. Later notes win, so a note drawn on top
/// of another is the one you grab.
int note_at(const Score& score, const PianoView& view, float x, float y);

/// Where along a note's width a grab landed. Resizing from the right edge is
/// the convention; a fixed pixel band means it stays grabbable at any zoom.
enum class NoteGrab { Move, ResizeEnd };
NoteGrab grab_kind(const Score& score, const PianoView& view, int index, float x);

struct PianoStats {
    int notes_drawn = 0;
    int rows_drawn  = 0;
};

/// Draws the roll into `bounds`. `selected` is an index into `score.notes`, or
/// -1. `playhead` is in clip-relative frames, or negative to hide it.
PianoStats draw_pianoroll(SkCanvas* canvas, const SkRect& bounds,
                          const Score& score, const PianoView& view,
                          int selected, int64_t playhead,
                          int64_t frames_per_bar);

}  // namespace mx
