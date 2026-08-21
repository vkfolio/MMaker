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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio/graph.h"
#include "media/peaks.h"

namespace mx {

using SourceId = uint32_t;

/// Where a piece of audio came from, and the decoded result.
///
/// The cache is keyed by content hash rather than by server id, so it stays
/// correct even while the server's id guarantees are weak.
struct Source {
    SourceId    id = 0;
    std::string project_id;
    std::string stem_id;
    std::string version_id;
    std::string url;
    std::string sha256;

    std::filesystem::path local_path;
    std::string           label;

    BufferRef buffer;      // decoded, at device rate
    Peaks     peaks;
    uint32_t  source_rate = 0;

    bool loaded() const { return buffer != nullptr; }
};

struct Clip {
    ClipId      id = 0;
    TrackId     track_id = 0;
    SourceId    source_id = 0;

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

    int64_t loop_start = 0;
    int64_t loop_end   = 0;

    // Ids are handed out here and never reused, so a stale command or a stale
    // graph can always be recognised as stale rather than aliasing a live object.
    TrackId  next_track  = 1;
    ClipId   next_clip   = 1;
    SourceId next_source = 1;

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
