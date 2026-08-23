// The output device.
//
// Thin on purpose: miniaudio owns the backend, the mixer owns the sound, and
// this only connects them and reports what the device turned out to be. The
// device's real latency is queried rather than assumed, because it is what the
// playhead is compensated by -- guessing it would produce a playhead that is
// confidently wrong.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mixer.h"
#include "recorder.h"

struct ma_device;

namespace mx {

class Device {
public:
    /// What the audio callback needs, defined in the .cpp. Public only so the
    /// callback -- a free function, as miniaudio requires -- can name it.
    struct Sides;

    ~Device();

    /// Opens the default output device and starts pulling from `mixer`.
    ///
    /// `recorder`, when given, makes this a duplex device: the same callback
    /// receives capture as well as playback. One device rather than two,
    /// because two run on independent clocks and a take recorded against a
    /// drifting one lands progressively further from the beat -- audible over
    /// a minute and impossible to fix afterwards.
    bool start(Mixer& mixer, uint32_t preferred_rate = 48000,
               Recorder* recorder = nullptr, int capture_index = -1);

    /// Names of the capture devices, in the order `start` indexes them.
    ///
    /// A DAW that cannot choose its input is not usable on a laptop, where the
    /// default is routinely an array microphone the OS has muted and the one
    /// you want is third in the list.
    static std::vector<std::string> capture_devices();

    /// Whether the device opened with an input side.
    bool capturing() const { return capture_channels_ > 0; }
    uint32_t capture_channels() const { return capture_channels_; }
    const std::string& capture_name() const { return capture_name_; }
    void stop();

    bool        running() const { return running_; }
    uint32_t    rate() const { return rate_; }
    uint32_t    channels() const { return channels_; }
    uint32_t    latency_frames() const { return latency_frames_; }
    const std::string& backend() const { return backend_; }
    const std::string& error() const { return error_; }

private:
    ma_device*  device_ = nullptr;
    Sides*      sides_ = nullptr;   // owned; freed only after the device stops
    Mixer*      mixer_  = nullptr;
    bool        running_ = false;
    uint32_t    rate_ = 0;
    uint32_t    channels_ = 2;
    uint32_t    latency_frames_ = 0;
    uint32_t    capture_channels_ = 0;
    std::string capture_name_;
    std::string backend_;
    std::string error_;
};

}  // namespace mx
