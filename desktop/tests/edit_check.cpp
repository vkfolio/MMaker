// The editing contract.
//
// These operations move a window onto audio; they never touch the audio. So
// the properties worth asserting are the ones that make a cut inaudible and a
// trim reversible:
//
//   conservation -- a split leaves two clips that together read exactly what
//                   the one did, same source frames, no gap and no overlap
//   containment  -- a trim reveals or hides material, it does not slide the
//                   take under the window
//   safety       -- no operation may produce a clip that cannot be grabbed
//                   again: zero length, inverted, or reading before its source
//
// The split arithmetic is the part worth guarding hardest. source_offset is
// invisible on screen -- a wrong one plays the right length of the wrong audio,
// which sounds like a bad take rather than like a bug.

#include <cstdio>
#include <string>

#include "audio/midi_in.h"
#include "session.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) ++failures;
    std::printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
}

/// A session with one 1000-frame clip starting at 500, reading from frame 200
/// of its source -- a deliberately non-zero offset, since an offset of zero
/// hides exactly the arithmetic errors this is here to catch.
mx::Session make_session(mx::ClipId* out) {
    mx::Session s;
    auto& track = s.add_track("t", 0xff4f8ef7);
    auto& src = s.add_source("x.wav", "x");
    auto& clip = s.add_clip(track.id, src.id, 500, 1000);
    clip.source_offset = 200;
    *out = clip.id;
    return s;
}

}  // namespace

int main() {
    std::printf("\nclip editing\n%s\n", std::string(72, '=').c_str());

    {
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        mx::Clip* right = s.split_clip(id, 900);
        check(right != nullptr, "a split inside the clip produces a second clip");

        const mx::Clip* left = s.find_clip(id);
        check(left && left->start_frame == 500 && left->length == 400,
              "the left half keeps the start and loses the remainder");
        check(right && right->start_frame == 900 && right->length == 600,
              "the right half starts at the cut and keeps the rest");
        check(left && right &&
                  left->start_frame + left->length == right->start_frame,
              "the halves meet exactly: no gap, no overlap");
        // The whole point: the right half must read further into the source by
        // exactly as much as the left half consumed.
        check(right && right->source_offset == 200 + 400,
              "the right half reads on from where the left half stopped");
        check(left && right && left->source_offset == 200,
              "the left half still reads from the original offset");
        check(left && right && left->length + right->length == 1000,
              "no audio is gained or lost across the cut");
        check(right && right->source_id == left->source_id &&
                  right->track_id == left->track_id,
              "the halves share the source and the track");
    }

    {
        // A cut outside, or flush with an edge, would make a clip of zero
        // length: it draws as a sliver and can never be grabbed again.
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        check(s.split_clip(id, 500) == nullptr, "a cut at the very start is refused");
        check(s.split_clip(id, 1500) == nullptr, "a cut at the very end is refused");
        check(s.split_clip(id, 100) == nullptr, "a cut before the clip is refused");
        check(s.split_clip(id, 9000) == nullptr, "a cut after the clip is refused");
        check(s.clips.size() == 1, "a refused cut leaves the clip alone");
    }

    {
        // Notes are shared by reference, so two halves would edit one score.
        mx::Session s;
        auto& t = s.add_track("v", 0xff4f8ef7);
        auto& c = s.add_score_clip(t.id, "voice", "Voice", 0, 1000);
        check(s.split_clip(c.id, 500) == nullptr, "a note clip is not split");
    }

    {
        // Trimming the start slides the window, not the audio: what is under
        // any given frame of the timeline must not move.
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        s.trim_start(id, 700);
        const mx::Clip* c = s.find_clip(id);
        check(c && c->start_frame == 700 && c->length == 800,
              "trimming the start shortens from the left");
        check(c && c->source_offset == 200 + 200,
              "trimming the start reveals later source material");
        // The frame at timeline 900 read source 200+(900-500)=600 before, and
        // must still read 600 after.
        check(c && c->source_offset + (900 - c->start_frame) == 600,
              "audio under the timeline does not move when the start is trimmed");
    }

    {
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        s.trim_end(id, 1200);
        const mx::Clip* c = s.find_clip(id);
        check(c && c->start_frame == 500 && c->length == 700,
              "trimming the end shortens from the right");
        check(c && c->source_offset == 200, "trimming the end leaves the offset alone");
        s.trim_end(id, 400);   // behind its own start
        c = s.find_clip(id);
        check(c && c->length >= 1, "a clip cannot be trimmed to nothing");
    }

    {
        // Reading before the start of a source would be silence at best.
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        s.trim_start(id, 0);
        const mx::Clip* c = s.find_clip(id);
        check(c && c->source_offset >= 0, "a trim cannot read before the source starts");
        check(c && c->length >= 1, "a trim cannot invert the clip");
    }

    {
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        const mx::Clip* copy = s.duplicate_clip(id);
        check(copy && copy->id != id, "a duplicate is a new clip");
        check(copy && copy->start_frame == 1500,
              "a duplicate lands where the original ends");
        check(copy && copy->source_offset == 200 && copy->length == 1000,
              "a duplicate reads the same audio");
    }

    {
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        check(s.remove_clip(id), "a clip can be removed");
        check(s.clips.empty(), "removing takes it off the timeline");
        check(!s.remove_clip(id), "removing it twice is refused, not a crash");
    }

    {
        // Deleting a track must not leave clips pointing at nothing: the mixer
        // would look up a track id that no longer exists.
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        const mx::TrackId t = s.clips.front().track_id;
        s.remove_track(t);
        check(s.tracks.empty() && s.clips.empty(),
              "removing a track takes its clips with it");
    }

    {
        // Ids must never be reused: a saved document referring to clip 3 must
        // not resolve to a different clip after an edit.
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        s.split_clip(id, 900);
        const mx::ClipId after_split = s.next_clip;
        s.remove_clip(id);
        s.duplicate_clip(s.clips.front().id);
        check(s.next_clip > after_split, "ids keep climbing across delete and copy");
        bool unique = true;
        for (size_t i = 0; i < s.clips.size(); ++i)
            for (size_t j = i + 1; j < s.clips.size(); ++j)
                if (s.clips[i].id == s.clips[j].id) unique = false;
        check(unique, "every clip id is distinct");
    }

    {
        // clip_at_frame is what a split with no selection uses to find its
        // target, so its edges must match the drawn ones.
        mx::ClipId id = 0;
        mx::Session s = make_session(&id);
        const mx::TrackId t = s.clips.front().track_id;
        check(s.clip_at_frame(t, 499) == nullptr, "just before the clip finds nothing");
        check(s.clip_at_frame(t, 500) != nullptr, "the first frame is inside");
        check(s.clip_at_frame(t, 1499) != nullptr, "the last frame is inside");
        check(s.clip_at_frame(t, 1500) == nullptr, "one past the end is outside");
    }

    {
        // MIDI placement. The driver stamps events as they arrive and the UI
        // drains a whole block at once, so this arithmetic is what decides
        // whether a chord stays a chord.
        const uint32_t rate = 48000;
        const int64_t anchor = 100000;

        check(mx::midi_frame_of(500, 500, anchor, rate) == anchor,
              "the newest note lands on the anchor");
        // 100 ms earlier is 4800 frames earlier at 48 kHz.
        check(mx::midi_frame_of(400, 500, anchor, rate) == anchor - 4800,
              "an earlier note lands earlier, by its own timestamp");
        check(mx::midi_frame_of(450, 500, anchor, rate) == anchor - 2400,
              "half the delay is half the distance");
        check(mx::midi_frame_of(400, 500, anchor, rate) <
                  mx::midi_frame_of(450, 500, anchor, rate),
              "an arpeggio keeps its order");
        check(mx::midi_frame_of(500, 500, anchor, rate) ==
                  mx::midi_frame_of(500, 500, anchor, rate),
              "notes struck together land together");
        check(mx::midi_frame_of(0, 5000, 100, rate) == 0,
              "a note cannot be placed before the start");
    }

    std::printf("%s\n", std::string(72, '=').c_str());
    if (failures == 0) {
        std::printf("ALL PASS\n\n");
        return 0;
    }
    std::printf("%d FAILURE(S)\n\n", failures);
    return 1;
}
