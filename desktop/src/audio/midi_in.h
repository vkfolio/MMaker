// MidiIn -- note events from a MIDI keyboard.
//
// Windows' own midiIn rather than a library: winmm is already linked for the
// audio backend, and what is needed here is note on, note off and a timestamp.
// A cross-platform MIDI dependency would be a build-system change bought for
// three message types.
//
// The driver callback runs on a high-priority system thread and is documented
// as forbidden from calling back into the MIDI API or blocking, so it does the
// same thing the audio callback does: writes into a preallocated ring and
// returns. The UI drains it on the tick it already runs.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace mx {

/// One note event. Control changes, pitch bend and clock are ignored -- what a
/// piano roll can show is notes, and inventing a representation for the rest
/// before anything reads it would be guessing.
struct MidiNote {
    uint8_t  pitch = 60;
    uint8_t  velocity = 0;     // 0 means note off
    bool     on = false;
    uint32_t ms = 0;           // driver timestamp, milliseconds since open
};

/// Where a played note belongs on the timeline.
///
/// The driver stamps each event as it arrives; the UI drains them up to a
/// frame later, so a whole block would otherwise land on one instant. Anchor
/// the newest event to the current position and offset the rest by their own
/// stamps, and a chord stays a chord while an arpeggio stays an arpeggio -- at
/// 60 Hz that is the difference between sixteen milliseconds of slop and none.
///
/// Never before zero: a stamp older than the anchor means the transport was
/// started mid-block, and a negative start is not a place.
inline int64_t midi_frame_of(uint32_t event_ms, uint32_t newest_ms,
                             int64_t anchor_frame, uint32_t rate) {
    const int64_t behind =
        static_cast<int64_t>(newest_ms - event_ms) * static_cast<int64_t>(rate) / 1000;
    const int64_t at = anchor_frame - behind;
    return at > 0 ? at : 0;
}

class MidiIn {
public:
    ~MidiIn();

    /// Names of the input devices, in the order `open` indexes them.
    static std::vector<std::string> devices();

    /// Opens a device. -1 opens the first available. Returns false with a
    /// reason in error() when there is none.
    bool open(int index = -1);
    void close();

    bool listening() const { return open_.load(std::memory_order_acquire); }
    const std::string& name() const { return name_; }
    const std::string& error() const { return error_; }

    /// UI thread. Takes whatever has arrived since the last call.
    std::vector<MidiNote> drain();

    /// Driver thread. Never blocks, never allocates.
    void push(const MidiNote& note);

private:
    static constexpr size_t kCapacity = 512;

    std::array<MidiNote, kCapacity> ring_{};
    std::atomic<uint64_t> head_{0};
    std::atomic<uint64_t> tail_{0};
    std::atomic<bool>     open_{false};
    std::atomic<uint64_t> dropped_{0};

    void*       handle_ = nullptr;   // HMIDIIN
    std::string name_;
    std::string error_;
};

}  // namespace mx
