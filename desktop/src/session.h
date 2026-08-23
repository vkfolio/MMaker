// The local document.
//
// Deliberately *not* a mirror of the server's Project. The server is a render
// service, not a document store: mix state exists on both sides, its saves are
// whole-file rewrites, and a long job finishing can clobber an edit made while
// it ran. Mirroring buys those liabilities for nothing.
//
// So the session keeps two things: the local arrangement, which is the truth
// for local playback, and *provenance* -- enough to say where a piece of audio
// came from and to fetch it again. Nothing more.

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio/graph.h"
#include "media/peaks.h"

namespace mx {

using SourceId = uint32_t;

/// One rendered state of a source. A tool appends one rather than replacing
/// what was there, which is what makes a result comparable: without a previous
/// version to switch back to, a successful regenerate and a no-op look
/// identical unless you happen to replay the exact region.
struct SourceVersion {
    std::string           id;          // server version id
    std::string           audio;       // project-relative path
    std::string           op;          // "import", "repaint", "vary", ...
    std::filesystem::path local_path;  // in the content cache
    BufferRef             buffer;
    Peaks                 peaks;
};

/// Where a piece of audio came from, and every version of it fetched so far.
///
/// The cache is keyed by project + version + path, so it stays correct even
/// while the server's id guarantees are weak.
struct Source {
    SourceId    id = 0;
    std::string project_id;
    std::string stem_id;
    std::string version_id;
    std::string url;
    std::string sha256;

    std::filesystem::path local_path;
    std::string           label;

    BufferRef buffer;      // decoded, at device rate -- the current version
    Peaks     peaks;
    uint32_t  source_rate = 0;

    /// Every version fetched this session, oldest first, and which one is
    /// showing. Switching is instant because each is already decoded.
    std::vector<SourceVersion> versions;
    int                        current = 0;

    bool loaded() const { return buffer != nullptr; }

    /// Records a version and makes it current.
    void adopt(SourceVersion v) {
        // A tool re-run with the same result should not stack duplicates.
        for (size_t i = 0; i < versions.size(); ++i) {
            if (!v.id.empty() && versions[i].id == v.id) {
                current = static_cast<int>(i);
                show(current);
                return;
            }
        }
        versions.push_back(std::move(v));
        current = static_cast<int>(versions.size()) - 1;
        show(current);
    }

    /// Switches which version the timeline plays and draws.
    void show(int index) {
        if (versions.empty()) return;
        current = std::clamp(index, 0, static_cast<int>(versions.size()) - 1);
        buffer = versions[current].buffer;
        peaks = versions[current].peaks;
        version_id = versions[current].id;
    }
};

using ScoreId = uint32_t;

/// One sung or played note.
///
/// Times are in frames, like everything else in the session, not in ticks. One
/// time unit in the document and conversions only at the edges is what keeps a
/// note, a clip and the playhead from disagreeing after a tempo change.
struct Note {
    int64_t     start  = 0;      // relative to the clip, not the timeline
    int64_t     length = 0;
    uint8_t     pitch  = 60;     // MIDI note number, 60 = C4
    std::string lyric;           // one syllable; empty for an instrument
};

/// The notes behind a clip, plus who should perform them.
///
/// This is the second input representation in the app: ACE-Step renders audio
/// from prompts, and has no notion of a note. A score is what the singing and
/// instrument engines take instead, so it lives beside `Source` rather than
/// inside it.
struct Score {
    ScoreId           id = 0;
    std::vector<Note> notes;
    std::string       voice_id;             // which voice or instrument
    std::string       voice_label;          // for the clip title
    std::string       language = "en";
    std::string       key = "C Major";

    /// Frames from the clip start to the end of the last note.
    int64_t length_frames() const {
        int64_t end = 0;
        for (const auto& n : notes) end = std::max(end, n.start + n.length);
        return end;
    }
};

struct Clip {
    ClipId      id = 0;
    TrackId     track_id = 0;
    SourceId    source_id = 0;
    /// A clip is backed by audio *or* by notes, never both.
    ///
    /// A score clip carries no source until it has been rendered, and
    /// `build_graph` already skips clips whose source is not loaded -- so it is
    /// silent with no change to the mixer at all. When the render arrives it
    /// lands as an ordinary `Source` on this same clip, which is what lets
    /// playback, offline bounce and the version badge keep working untouched.
    ScoreId     score_id = 0;

    int64_t start_frame   = 0;
    int64_t length        = 0;
    int64_t source_offset = 0;
    int64_t fade_in       = 0;
    int64_t fade_out      = 0;
    float   gain          = 1.0f;
    std::string name;
};

struct Track {
    TrackId     id = 0;
    std::string name;
    float       gain   = 1.0f;
    float       pan    = 0.0f;
    bool        muted  = false;
    bool        soloed = false;
    uint32_t    color  = 0xff4f8ef7;
};

struct Session {
    std::string title = "Untitled";
    double      bpm = 120.0;
    int         beats_per_bar = 4;
    uint32_t    rate = 48000;

    std::vector<Track>  tracks;
    std::vector<Clip>   clips;
    std::vector<Source> sources;
    std::vector<Score>  scores;

    int64_t loop_start = 0;
    int64_t loop_end   = 0;

    // Ids are handed out here and never reused, so a stale command or a stale
    // graph can always be recognised as stale rather than aliasing a live object.
    TrackId  next_track  = 1;
    ClipId   next_clip   = 1;
    SourceId next_source = 1;
    ScoreId  next_score  = 1;

    Track& add_track(std::string name, uint32_t color) {
        tracks.push_back(Track{next_track++, std::move(name), 1.0f, 0.0f, false, false, color});
        return tracks.back();
    }

    Source& add_source(std::filesystem::path path, std::string label) {
        Source s;
        s.id = next_source++;
        s.local_path = std::move(path);
        s.label = std::move(label);
        sources.push_back(std::move(s));
        return sources.back();
    }

    Clip& add_clip(TrackId track, SourceId source, int64_t start, int64_t length) {
        Clip c;
        c.id = next_clip++;
        c.track_id = track;
        c.source_id = source;
        c.start_frame = start;
        c.length = length;
        clips.push_back(c);
        return clips.back();
    }

    /// Creates an empty score and the clip that shows it.
    Clip& add_score_clip(TrackId track, std::string voice_id, std::string label,
                         int64_t start, int64_t length) {
        Score sc;
        sc.id = next_score++;
        sc.voice_id = std::move(voice_id);
        sc.voice_label = std::move(label);
        scores.push_back(std::move(sc));

        Clip c;
        c.id = next_clip++;
        c.track_id = track;
        c.score_id = scores.back().id;
        c.start_frame = start;
        c.length = length;
        c.name = scores.back().voice_label + "_Clip_1";
        clips.push_back(c);
        return clips.back();
    }

    Score* find_score(ScoreId id) {
        if (id == 0) return nullptr;
        for (auto& s : scores)
            if (s.id == id) return &s;
        return nullptr;
    }
    Source* find_source(SourceId id) {
        for (auto& s : sources)
            if (s.id == id) return &s;
        return nullptr;
    }
    Clip* find_clip(ClipId id) {
        for (auto& c : clips)
            if (c.id == id) return &c;
        return nullptr;
    }
    Track* find_track(TrackId id) {
        for (auto& t : tracks)
            if (t.id == id) return &t;
        return nullptr;
    }

    double seconds_per_bar() const {
        return static_cast<double>(beats_per_bar) * 60.0 / bpm;
    }
    int64_t frames_per_bar() const {
        return static_cast<int64_t>(seconds_per_bar() * rate);
    }
    int64_t length_frames() const {
        int64_t end = 0;
        for (const auto& c : clips) end = std::max(end, c.start_frame + c.length);
        return end;
    }
};

/// Flattens the session into the immutable snapshot the audio thread reads.
///
/// Every published graph gets a fresh generation, which is what lets the mixer
/// free the ones the callback skipped.
inline std::unique_ptr<PlaybackGraph> build_graph(Session& session, uint64_t generation) {
    auto g = std::make_unique<PlaybackGraph>();
    g->generation = generation;

    for (const auto& t : session.tracks) {
        g->tracks.push_back(GraphTrack{t.id, t.gain, t.pan, t.muted, t.soloed});
        if (t.soloed) g->any_solo = true;
    }

    for (const auto& c : session.clips) {
        Source* src = session.find_source(c.source_id);
        if (!src || !src->loaded()) continue;

        GraphClip gc;
        gc.track_id      = c.track_id;
        gc.clip_id       = c.id;
        gc.samples       = src->buffer->data();
        gc.source_frames = src->buffer->frames;
        gc.channels      = src->buffer->channels;
        gc.start_frame   = c.start_frame;
        gc.length        = c.length;
        gc.source_offset = c.source_offset;
        gc.fade_in       = c.fade_in;
        gc.fade_out      = c.fade_out;
        gc.gain          = c.gain;
        g->clips.push_back(gc);

        // The keepalive is what makes those raw pointers safe: the buffer
        // cannot be freed while a graph still refers to it, and the refcount is
        // only ever touched here, on the UI thread.
        g->keepalive.push_back(src->buffer);
    }

    std::sort(g->clips.begin(), g->clips.end(),
              [](const GraphClip& a, const GraphClip& b) {
                  return a.start_frame < b.start_frame;
              });
    return g;
}

}  // namespace mx
