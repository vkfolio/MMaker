// The contract between the UI thread and the audio callback.
//
// Two channels cross this boundary and they have different shapes on purpose:
//
//   content  -- what to play. Published as a whole immutable graph through one
//               atomic pointer, because a clip drag changes many things at once
//               and the callback must never observe half an edit.
//   control  -- gain, mute, solo, transport. A drag produces changes faster
//               than the callback drains them, so these coalesce (last wins)
//               instead of queueing.
//
// Nothing here allocates, locks, or touches a refcount on the audio side.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace mx {

using TrackId = uint32_t;
using ClipId  = uint32_t;

/// Decoded audio, already at the device sample rate.
///
/// Pre-resampling at decode is a deliberate fork: it keeps the mixer free of
/// per-clip resampler state, which is the part that is genuinely hard to keep
/// click-free. The cost is that changing the default device to a different rate
/// invalidates every buffer, so `source_rate` is kept to make a re-decode
/// possible rather than a re-download.
struct AudioBuffer {
    std::vector<float> samples;      // interleaved
    uint32_t           channels = 2;
    uint32_t           rate     = 48000;   // == device rate, post-resample
    uint32_t           source_rate = 48000;
    int64_t            frames   = 0;

    const float* data() const { return samples.data(); }
};

using BufferRef = std::shared_ptr<const AudioBuffer>;

/// One clip as the callback sees it: raw pointers and integers, nothing that
/// can allocate or block. Lifetime is guaranteed by the keepalives on the
/// owning graph, which only the UI thread ever touches.
struct GraphClip {
    TrackId      track_id      = 0;
    ClipId       clip_id       = 0;
    const float* samples       = nullptr;
    int64_t      source_frames = 0;
    uint32_t     channels      = 2;

    int64_t start_frame  = 0;   // where on the timeline it begins
    int64_t length       = 0;   // how long it plays for
    int64_t source_offset = 0;  // where in the source it starts reading

    int64_t fade_in  = 0;       // frames
    int64_t fade_out = 0;
    float   gain     = 1.0f;
};

struct GraphTrack {
    TrackId id     = 0;
    float   gain   = 1.0f;
    float   pan    = 0.0f;      // -1 left, +1 right
    bool    muted  = false;
    bool    soloed = false;
};

/// An immutable snapshot. Once published it is never modified -- the next edit
/// builds a new one.
struct PlaybackGraph {
    uint64_t                generation = 0;
    std::vector<GraphTrack> tracks;
    std::vector<GraphClip>  clips;      // sorted by start_frame
    bool                    any_solo = false;

    /// Keeps decoded audio alive for exactly as long as some published graph
    /// still points at it. The audio thread never reads this member, so the
    /// refcounts it holds are only ever touched on the UI thread.
    std::vector<BufferRef> keepalive;
};

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

/// Per-track control that a drag can outrun.
///
/// A fader at mouse-poll rate produces changes far faster than a 375 Hz
/// callback drains a queue, so these are coalescing cells rather than messages:
/// the callback reads whatever the latest value is and never falls behind.
/// Indexed by *slot*, with the id stored alongside, so a graph swap can never
/// aim a stale command at the wrong track.
struct TrackControl {
    std::atomic<TrackId> id{0};
    std::atomic<float>   gain{1.0f};
    std::atomic<float>   pan{0.0f};
    std::atomic<bool>    muted{false};
    std::atomic<bool>    soloed{false};
    std::atomic<float>   peak{0.0f};     // RT accumulates, UI exchanges to 0
};

constexpr int kMaxTracks = 128;

/// Transport intent. Seeks carry a sequence number so the UI can tell an
/// acknowledged seek from a playhead that merely happens to be near the target.
struct Transport {
    std::atomic<bool>     playing{false};
    std::atomic<bool>     looping{false};
    std::atomic<int64_t>  loop_start{0};
    std::atomic<int64_t>  loop_end{0};

    std::atomic<int64_t>  seek_to{-1};     // -1 = no seek pending
    std::atomic<uint64_t> seek_request{0};
    std::atomic<uint64_t> seek_ack{0};
};

}  // namespace mx
