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

#include "mixer.h"

struct ma_device;

namespace mx {

class Device {
public:
    ~Device();

    /// Opens the default output device and starts pulling from `mixer`.
    bool start(Mixer& mixer, uint32_t preferred_rate = 48000);
    void stop();

    bool        running() const { return running_; }
    uint32_t    rate() const { return rate_; }
    uint32_t    channels() const { return channels_; }
    uint32_t    latency_frames() const { return latency_frames_; }
    const std::string& backend() const { return backend_; }
    const std::string& error() const { return error_; }

private:
    ma_device*  device_ = nullptr;
    Mixer*      mixer_  = nullptr;
    bool        running_ = false;
    uint32_t    rate_ = 0;
    uint32_t    channels_ = 2;
    uint32_t    latency_frames_ = 0;
    std::string backend_;
    std::string error_;
};

}  // namespace mx
