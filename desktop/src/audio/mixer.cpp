#include "mixer.h"

#include <algorithm>
#include <cmath>

namespace mx {
namespace {

/// A time constant short enough to be inaudible as a slide and long enough to
/// hide a step. ~5 ms is the usual compromise for gain; below ~2 ms a step
/// still ticks.
inline float ramp_step(uint32_t rate, float seconds) {
    return 1.0f / std::max(1.0f, seconds * static_cast<float>(rate));
}

inline float approach(float current, float target, float step) {
    if (current < target) return std::min(target, current + step);
    if (current > target) return std::max(target, current - step);
    return current;
}

}  // namespace

void Mixer::prepare(uint32_t sample_rate, uint32_t channels) {
    rate_     = sample_rate;
    channels_ = channels;
    clip_states_.assign(kClipSlots, ClipState{});
    std::fill(std::begin(track_gain_), std::end(track_gain_), 0.0f);
    master_gain_ = 0.0f;
    declick_gain_ = 0.0f;
    std::fill(std::begin(last_out_), std::end(last_out_), 0.0f);
    std::fill(std::begin(declick_), std::end(declick_), 0.0f);
}

void Mixer::begin_offline() {
    offline_ = true;
    master_gain_ = 1.0f;
    declick_gain_ = 0.0f;
    played_.store(0, std::memory_order_relaxed);
    for (auto& st : clip_states_) st = ClipState{};
}

Mixer::ClipState& Mixer::clip_state(ClipId id) noexcept {
    // Open addressing over a preallocated table: no allocation on the RT
    // thread, and a bounded probe. A collision degrades to a shared smoother,
    // which is inaudible; running out of slots would not be, so the table is
    // sized well above any plausible number of simultaneously audible clips.
    const uint32_t mask = kClipSlots - 1;
    uint32_t h = (id * 2654435761u) & mask;
    for (int probe = 0; probe < 16; ++probe) {
        ClipState& s = clip_states_[h];
        if (!s.in_use) {
            s = ClipState{id, true, 0.0f, false};
            return s;
        }
        if (s.clip_id == id) return s;
        h = (h + 1) & mask;
    }
    return clip_states_[h];   // pathological: reuse a slot rather than fail
}


void Mixer::arm_declick() noexcept {
    // Continue from wherever the output actually was and decay that value away,
    // so the discontinuity is covered by a short crossfade instead of a step.
    for (int c = 0; c < kDeclickChannels; ++c) declick_[c] = last_out_[c];
    declick_gain_ = 1.0f;
}

void Mixer::publish(std::unique_ptr<PlaybackGraph> graph) {
    const PlaybackGraph* raw = graph.get();
    retired_.push_back(std::move(graph));
    pending_.store(raw, std::memory_order_release);
}

void Mixer::collect(bool device_running) {
    if (retired_.empty()) return;

    if (!device_running) {
        // Nothing is reading. Keep the newest -- it is what will be adopted
        // when the device starts -- and free the rest now. Without this path a
        // stopped or lost device leaks every graph an edit produces.
        if (retired_.size() > 1)
            retired_.erase(retired_.begin(), retired_.end() - 1);
        return;
    }

    // Generation-based retirement, not hand-back. The callback may never
    // observe most published graphs -- a 60 fps clip drag publishes ~60 graphs
    // against a ~375 Hz drain -- so waiting for each to be handed back leaks
    // every one that was skipped.
    const uint64_t live = active_gen_.load(std::memory_order_acquire);
    auto dead = std::remove_if(
        retired_.begin(), retired_.end(),
        [&](const std::unique_ptr<PlaybackGraph>& g) { return g->generation < live; });
    retired_.erase(dead, retired_.end());
}

int64_t Mixer::heard_frame() const noexcept {
    const int64_t rendered = played_.load(std::memory_order_relaxed);
    const int64_t latency  = latency_frames_.load(std::memory_order_relaxed);
    return std::max<int64_t>(0, rendered - latency);
}

void Mixer::process(float* out, int64_t frames) noexcept {
    const int ch = static_cast<int>(channels_);
    std::fill(out, out + frames * ch, 0.0f);

    // Adopt a newly published graph as a whole. Doing this once per block,
    // rather than per clip, is what guarantees the callback never observes half
    // an edit.
    if (const PlaybackGraph* next = pending_.load(std::memory_order_acquire);
        next != current_) {
        current_ = next;
        if (current_) active_gen_.store(current_->generation, std::memory_order_release);
    }

    int64_t pos = played_.load(std::memory_order_relaxed);

    // A seek is a discontinuity: ramp out, jump, ramp back in. Acknowledging by
    // sequence number lets the UI tell "my seek landed" apart from "the playhead
    // happens to be near where I asked".
    if (const int64_t target = transport.seek_to.exchange(-1, std::memory_order_acq_rel);
        target >= 0) {
        pos = target;
        arm_declick();
        transport.seek_ack.store(transport.seek_request.load(std::memory_order_relaxed),
                                 std::memory_order_release);
    }

    const bool    playing   = transport.playing.load(std::memory_order_relaxed);
    const float   gain_step = ramp_step(rate_, 0.005f);
    const bool    looping   = transport.looping.load(std::memory_order_relaxed);
    const int64_t loop_a    = transport.loop_start.load(std::memory_order_relaxed);
    const int64_t loop_b    = transport.loop_end.load(std::memory_order_relaxed);

    if (!playing && master_gain_ <= 0.0f) {
        // Stopped and already silent: still publish the position, so a seek made
        // while stopped is visible, but do no mixing work.
        played_.store(pos, std::memory_order_relaxed);
        return;
    }

    const bool any_solo = current_ && current_->any_solo;
    float block_peak = 0.0f;

    if (current_) {
        for (const GraphClip& clip : current_->clips) {
            if (!clip.samples || clip.length <= 0) continue;

            const int64_t clip_end = clip.start_frame + clip.length;
            if (clip_end <= pos || clip.start_frame >= pos + frames) {
                clip_state(clip.clip_id).was_on = false;
                continue;
            }

            // Look the track up by id, never by index: an index would follow a
            // graph swap straight into the wrong track, or off the end.
            const GraphTrack* track = nullptr;
            int slot = -1;
            for (size_t i = 0; i < current_->tracks.size() && i < kMaxTracks; ++i) {
                if (current_->tracks[i].id == clip.track_id) {
                    track = &current_->tracks[i];
                    slot  = static_cast<int>(i);
                    break;
                }
            }
            if (!track) continue;

            const bool audible = !track->muted && (!any_solo || track->soloed);

            ClipState& st = clip_state(clip.clip_id);
            const float target_gain = audible ? clip.gain * track->gain : 0.0f;

            // Dragging a clip changes which source sample lands at time t, so
            // the waveform steps even though nothing about its *level* changed.
            // A derived read position prevents a cursor being reset; it does not
            // and cannot prevent the content itself jumping.
            //
            // The repair is a crossfade, and it takes both halves: hold and
            // decay the old output value, *and* bring the new material up from
            // silence. Doing only the first just relocates the step -- the new
            // content still arrives at full level -- which is why this needs to
            // be a fade rather than a patch over the seam.
            const bool jumped = st.was_on && (clip.start_frame != st.last_start ||
                                              clip.source_offset != st.last_offset);
            st.last_start  = clip.start_frame;
            st.last_offset = clip.source_offset;

            if (jumped && !offline_) arm_declick();
            if (!offline_ && (jumped || !st.was_on)) st.gain = 0.0f;
            if (offline_) st.gain = target_gain;   // settled: nothing to hide
            st.was_on = true;

            const float pan  = track->pan;
            const float panl = std::sqrt(std::max(0.0f, 1.0f - pan));
            const float panr = std::sqrt(std::max(0.0f, 1.0f + pan));

            for (int64_t i = 0; i < frames; ++i) {
                const int64_t t = pos + i;
                if (t < clip.start_frame || t >= clip_end) continue;

                // Derived from the playhead, not a stored cursor. This is why
                // dragging a clip during playback does not click: there is no
                // cursor for the edit to reset.
                const int64_t into = t - clip.start_frame;
                const int64_t src  = clip.source_offset + into;
                if (src < 0 || src >= clip.source_frames) continue;

                st.gain = approach(st.gain, target_gain, gain_step);

                float env = st.gain;
                if (clip.fade_in > 0 && into < clip.fade_in)
                    env *= static_cast<float>(into) / static_cast<float>(clip.fade_in);
                if (clip.fade_out > 0 && into >= clip.length - clip.fade_out)
                    env *= static_cast<float>(clip.length - into) /
                           static_cast<float>(clip.fade_out);

                const float* s = clip.samples + src * clip.channels;
                const float l = s[0] * env;
                const float r = (clip.channels > 1 ? s[1] : s[0]) * env;

                out[i * ch] += l * panl;
                if (ch > 1) out[i * ch + 1] += r * panr;
            }

            if (slot >= 0) {
                float p = 0.0f;
                for (int64_t i = 0; i < frames; ++i)
                    p = std::max(p, std::abs(out[i * ch]));
                const float prev = tracks_[slot].peak.load(std::memory_order_relaxed);
                tracks_[slot].peak.store(std::max(prev, p), std::memory_order_relaxed);
            }
        }
    }

    // Ramping out on stop only works if the material keeps advancing behind
    // the ramp. Freezing the position replays the same block underneath a
    // falling gain, which steps at the block boundary -- audibly worse than no
    // ramp at all.
    const bool sounding = playing || master_gain_ > 0.0f;

    const float master_target = playing ? 1.0f : 0.0f;
    const float declick_step = ramp_step(rate_, 0.005f);
    for (int64_t i = 0; i < frames; ++i) {
        master_gain_ = approach(master_gain_, master_target, gain_step);
        for (int c = 0; c < ch; ++c) {
            float v = out[i * ch + c] * master_gain_;
            if (declick_gain_ > 0.0f && c < kDeclickChannels)
                v += declick_[c] * declick_gain_;
            // Clamp for the device, never for a bounce. A sound card cannot
            // accept more than full scale, so clipping there is the least-bad
            // option -- but a float file can hold it, and clamping an export
            // throws away headroom the user may still want to recover. The
            // null test found this by clipping on a three-stem sum.
            if (!offline_) v = std::clamp(v, -1.0f, 1.0f);
            out[i * ch + c] = v;
            if (c < kDeclickChannels) last_out_[c] = v;
            block_peak = std::max(block_peak, std::abs(v));
        }
        if (declick_gain_ > 0.0f)
            declick_gain_ = std::max(0.0f, declick_gain_ - declick_step);
    }

    if (sounding) {
        pos += frames;
        if (looping && loop_b > loop_a && pos >= loop_b) {
            pos = loop_a + (pos - loop_b);
            arm_declick();
        }
    }
    played_.store(pos, std::memory_order_relaxed);

    const float prev = master_peak_.load(std::memory_order_relaxed);
    master_peak_.store(std::max(prev, block_peak), std::memory_order_relaxed);
}

}  // namespace mx
