#include "device.h"

#include "miniaudio.h"

namespace mx {
namespace {

void on_data(ma_device* device, void* out, const void*, ma_uint32 frames) {
    auto* mixer = static_cast<Mixer*>(device->pUserData);
    if (mixer) mixer->process(static_cast<float*>(out), static_cast<int64_t>(frames));
}

}  // namespace

Device::~Device() { stop(); }

bool Device::start(Mixer& mixer, uint32_t preferred_rate) {
    stop();

    ma_device_config cfg  = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = channels_;
    cfg.sampleRate        = preferred_rate;
    cfg.dataCallback      = &on_data;
    cfg.pUserData         = &mixer;

    device_ = new ma_device{};
    if (ma_device_init(nullptr, &cfg, device_) != MA_SUCCESS) {
        error_ = "no output device";
        delete device_;
        device_ = nullptr;
        return false;
    }

    // What the device actually gave us, which need not be what was asked for.
    rate_     = device_->sampleRate;
    channels_ = device_->playback.channels;
    backend_  = ma_get_backend_name(device_->pContext->backend);

    // Measured, not assumed. This is subtracted from the render position to
    // give the playhead the user sees; a wrong figure here shows up as a
    // playhead that leads or lags the sound, which is the sync error people
    // notice before any other.
    const ma_uint32 period  = device_->playback.internalPeriodSizeInFrames;
    const ma_uint32 periods = device_->playback.internalPeriods;
    latency_frames_ = period * (periods > 0 ? periods : 1);

    mixer.prepare(rate_, channels_);
    mixer.set_latency_frames(latency_frames_);
    mixer_ = &mixer;

    if (ma_device_start(device_) != MA_SUCCESS) {
        error_ = "device would not start";
        ma_device_uninit(device_);
        delete device_;
        device_ = nullptr;
        return false;
    }

    running_ = true;
    error_.clear();
    return true;
}

void Device::stop() {
    if (!device_) return;
    ma_device_uninit(device_);      // stops the callback before returning
    delete device_;
    device_ = nullptr;
    running_ = false;
    mixer_ = nullptr;
}

}  // namespace mx
