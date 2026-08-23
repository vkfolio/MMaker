// The piano roll's coordinate contract.
//
// The arrangement's version of this test exists because a lane hit-test that
// disagreed with lane drawing meant clicking one track and editing another.
// Here the same class of bug is worse: a pitch hit-test that disagrees with
// pitch drawing means typing a lyric onto the wrong note, which changes the
// document rather than just the picture.
//
// So the properties are the same three, on the new axis:
//
//   round trip -- the pitch under y is the pitch drawn at y, at every row
//                 height and every scroll offset
//   containment -- a note's own rect contains its own hit-test point, which is
//                 what makes "click the note you can see" true
//   grab zones  -- the resize band never swallows a note whole, or short notes
//                 could be resized but never moved

#include <cmath>
#include <cstdio>
#include <string>

#include "ui/pianoroll.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) ++failures;
    std::printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
}

mx::Note note(int64_t start, int64_t length, uint8_t pitch,
              const char* lyric = "") {
    mx::Note n;
    n.start = start;
    n.length = length;
    n.pitch = pitch;
    n.lyric = lyric;
    return n;
}

}  // namespace

int main() {
    std::printf("\npiano roll coordinates\n%s\n", std::string(72, '=').c_str());

    {
        // Pitch <-> y inverts, at every row height and scroll offset.
        int bad = 0;
        for (float row : {mx::kRowHeightMin, 14.0f, mx::kRowHeightMax}) {
            for (float scroll : {0.0f, 91.0f, 700.0f}) {
                mx::PianoView v;
                v.row_h = row;
                v.scroll_y_px = scroll;
                for (int p = mx::kPitchMin; p <= mx::kPitchMax; ++p) {
                    const auto pitch = static_cast<uint8_t>(p);
                    const float top = v.y_of(pitch);
                    if (top < mx::kPianoRulerHeight) continue;  // scrolled off
                    if (v.pitch_at(top + 1.0f) != pitch) ++bad;
                    if (v.pitch_at(top + row - 1.0f) != pitch) ++bad;
                }
            }
        }
        check(bad == 0, "pitch hit-test inverts pitch layout");
    }

    {
        // High pitches at the top. Inverting this is instantly wrong to anyone
        // who has used any editor, and trivially easy to do by accident.
        mx::PianoView v;
        check(v.y_of(72) < v.y_of(60), "higher pitches sit higher");
        check(v.pitch_at(mx::kPianoRulerHeight + 1.0f) > v.pitch_at(2000.0f),
              "y increases as pitch falls");
    }

    {
        // The ruler is not a note row -- the same rule the arrangement has.
        mx::PianoView v;
        check(v.pitch_at(2.0f) == 0, "the ruler is not a pitch row");
        v.scroll_y_px = 400.0f;
        check(v.pitch_at(2.0f) == 0, "the ruler is not a pitch row when scrolled");
    }

    {
        // A note's rect contains the point its own hit-test should find.
        mx::Score score;
        score.notes.push_back(note(0, 24'000, 60, "twin"));
        score.notes.push_back(note(24'000, 24'000, 64, "kle"));
        score.notes.push_back(note(48'000, 48'000, 67, "lit"));

        mx::PianoView v;
        v.frames_per_pixel = 512.0;
        int bad = 0;
        for (size_t i = 0; i < score.notes.size(); ++i) {
            const SkRect r = v.rect_of(score.notes[i]);
            const float cx = (r.left() + r.right()) * 0.5f;
            const float cy = (r.top() + r.bottom()) * 0.5f;
            if (mx::note_at(score, v, cx, cy) != static_cast<int>(i)) ++bad;
        }
        check(bad == 0, "each note is found at the centre of its own rect");
        check(mx::note_at(score, v, 4000.0f, 40.0f) == -1,
              "empty space finds no note");
    }

    {
        // Overlapping notes: the one drawn last is on top, so it is the one
        // you grab. Otherwise a note you cannot see steals the click.
        mx::Score score;
        score.notes.push_back(note(0, 48'000, 60));
        score.notes.push_back(note(0, 48'000, 60));
        mx::PianoView v;
        const SkRect r = v.rect_of(score.notes[0]);
        check(mx::note_at(score, v, (r.left() + r.right()) * 0.5f,
                          (r.top() + r.bottom()) * 0.5f) == 1,
              "the topmost of two stacked notes wins");
    }

    {
        // Resize band: grabbable at the right edge, and never so wide that a
        // short note cannot be moved at all.
        mx::Score score;
        score.notes.push_back(note(0, 48'000, 60));   // wide
        score.notes.push_back(note(0, 600, 62));      // narrower than the band

        mx::PianoView v;
        v.frames_per_pixel = 512.0;

        const SkRect wide = v.rect_of(score.notes[0]);
        check(mx::grab_kind(score, v, 0, wide.right() - 1.0f) ==
                  mx::NoteGrab::ResizeEnd,
              "the right edge of a note resizes it");
        check(mx::grab_kind(score, v, 0, wide.left() + 4.0f) == mx::NoteGrab::Move,
              "the body of a note moves it");

        const SkRect tiny = v.rect_of(score.notes[1]);
        check(mx::grab_kind(score, v, 1, tiny.left() + 0.1f) == mx::NoteGrab::Move,
              "a very short note can still be moved, not only resized");
    }

    {
        // Time behaves exactly as the arrangement's View does, so a frame means
        // the same thing in both surfaces and zoom feels identical.
        mx::PianoView v;
        v.frames_per_pixel = 256.0;
        v.scroll_frames = 500'000;
        int bad = 0;
        for (float x : {0.0f, 13.0f, 400.0f, 1600.0f})
            if (std::fabs(v.x_of(v.frame_at(x)) - x) > 1.0f) ++bad;
        check(bad == 0, "time round-trips through frame_at/x_of");
    }

    {
        // Opening onto an empty stretch of an 88-key keyboard reads as broken,
        // so the view frames whatever notes exist.
        mx::Score score;
        score.notes.push_back(note(0, 24'000, 84));
        score.notes.push_back(note(24'000, 24'000, 88));

        mx::PianoView v;
        v.frame_notes(score, 400.0f);
        const float y84 = v.y_of(84);
        check(y84 > mx::kPianoRulerHeight && y84 < 400.0f,
              "framing brings the notes on screen");

        mx::Score empty;
        mx::PianoView v2;
        v2.frame_notes(empty, 400.0f);
        const float middle_c = v2.y_of(60);
        check(middle_c > mx::kPianoRulerHeight && middle_c < 400.0f,
              "an empty score frames middle C");
    }

    {
        // Scroll clamps to the keyboard, both ends.
        mx::PianoView v;
        v.scroll_y_px = 99999.0f;
        v.clamp_y(300.0f);
        check(std::fabs(v.scroll_y_px - (v.content_height() - 300.0f)) < 0.001f,
              "vertical scroll clamps to the keyboard");
        v.scroll_y_px = -80.0f;
        v.clamp_y(300.0f);
        check(v.scroll_y_px == 0.0f, "vertical scroll cannot go above the top");
    }

    {
        // Note naming, since it is what the gutter shows and an off-by-one
        // octave is the classic MIDI mistake.
        check(mx::pitch_name(60) == "C4", "MIDI 60 is C4");
        check(mx::pitch_name(69) == "A4", "MIDI 69 is A4");
        check(mx::pitch_name(61) == "C#4", "MIDI 61 is C#4");
        check(mx::is_black_key(61) && !mx::is_black_key(60),
              "black keys are the black keys");
    }

    std::printf("%s\n", std::string(72, '=').c_str());
    if (failures == 0) {
        std::printf("ALL PASS\n\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n\n", failures);
    return 1;
}
