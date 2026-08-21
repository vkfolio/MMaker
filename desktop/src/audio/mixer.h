// The real-time mixer.
//
// Everything in `process()` runs on the audio thread: no allocation, no locks,
// no logging, no refcount traffic. The rules are not stylistic -- each one
// corresponds to an audible defect.
//
// Three decisions worth stating, because they are the ones that are painful to
// retrofit:
//
//   * A clip's read position is *derived* from the playhead, not stored. That
//     is what makes editing during playback safe: moving a clip cannot reset a
//     cursor that does not exist.
//   * Per-clip state that genuinely must persist -- the gain smoother and
//     whether the clip was audible last block -- lives here in a preallocated
//     table, never in the graph. State in the graph is reset by every edit,
//     which is the classic "works, but clicks whenever you touch anything".
//   * Every discontinuity gets a ramp: transport start/stop, seek, and a clip
//     appearing or vanishing under the playhead because of an edit.

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "graph.h"

namespace mx {

class Mixer {
public:
    void prepare(uint32_t sample_rate, uint32_t channels);

    /// Called from the UI thread. Takes ownership of the graph; it is freed
    /// only once the audio thread can no longer be looking at it.
    void publish(std::unique_ptr<PlaybackGraph> graph);

    /// Called from the UI thread, after `publish`, to free graphs the audio
    /// thread has moved past. Cheap; call it once a frame.
    void collect(bool device_running);

    /// The audio callback. Writes `frames` interleaved frames into `out`.
    void process(float* out, int64_t frames) noexcept;

    // --- published for the UI ------------------------------------------------

    /// Where the listener actually is, in frames -- the render position minus
    /// the device's output latency. An uncompensated playhead visibly leads the
    /// sound, which is the sync defect users notice first.
    int64_t heard_frame() const noexcept;
    int64_t rendered_frame() const noexcept {
        return played_.load(std::memory_order_relaxed);
    }

    void     set_latency_frames(uint32_t f) { latency_frames_.store(f, std::memory_order_relaxed); }
    uint32_t xruns() const { return xruns_.load(std::memory_order_relaxed); }
    uint64_t active_generation() const { return active_gen_.load(std::memory_order_acquire); }

    /// Peak since the last call, then reset. Accumulate on the RT side and
    /// exchange on the UI side, so a meter cannot miss the transient it exists
    /// to show -- a plain store loses every block the UI does not happen to read.
    float take_master_peak() { return master_peak_.exchange(0.0f, std::memory_order_relaxed); }
    float take_track_peak(int slot) {
        return tracks_[slot].peak.exchange(0.0f, std::memory_order_relaxed);
    }

    Transport      transport;
    TrackControl   tracks_[kMaxTracks];

private:
    struct ClipState {
        ClipId  clip_id = 0;
        bool    in_use  = false;
        float   gain    = 0.0f;   // smoothed, so an edit cannot step it
        bool    was_on  = false;
        // Where this clip was reading last block. A change means the content
        // under the playhead jumped -- which a gain ramp cannot repair, because
        // the ramp only learns about it after the jump has been emitted.
        int64_t last_start  = INT64_MIN;
        int64_t last_offset = INT64_MIN;
    };

    ClipState& clip_state(ClipId id) noexcept;
    void       arm_declick() noexcept;

    uint32_t rate_     = 48000;
    uint32_t channels_ = 2;

    const PlaybackGraph* current_ = nullptr;      // audio thread only

    std::atomic<const PlaybackGraph*> pending_{nullptr};
    std::atomic<uint64_t> active_gen_{0};
    std::atomic<int64_t>  played_{0};
    std::atomic<uint32_t> latency_frames_{0};
    std::atomic<uint32_t> xruns_{0};
    std::atomic<float>    master_peak_{0.0f};

    // UI thread only: every graph ever published, oldest first.
    std::vector<std::unique_ptr<PlaybackGraph>> retired_;

    float master_gain_ = 0.0f;    // ramped, never switched
    float track_gain_[kMaxTracks]{};

    // Declicker. A ramp handles a change of *level*; it cannot handle a change
    // of *content*, which is what a seek, a loop wrap, or dragging a clip under
    // the playhead all are. Those leave a step, so the last output value is
    // held and decayed out underneath the new material -- a short crossfade
    // against a frozen sample.
    static constexpr int kDeclickChannels = 2;
    float last_out_[kDeclickChannels]{};
    float declick_[kDeclickChannels]{};
    float declick_gain_ = 0.0f;

    static constexpr int kClipSlots = 2048;   // power of two
    std::vector<ClipState> clip_states_;
};

}  // namespace mx
