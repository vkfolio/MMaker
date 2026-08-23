#include "device.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <mmdeviceapi.h>
#include <endpointvolume.h>

#include "miniaudio.h"

namespace mx {

/// What the audio callback needs, in one allocation that outlives it.
struct Device::Sides {
    Mixer*    mixer = nullptr;
    Recorder* recorder = nullptr;
    uint32_t  capture_channels = 0;
};

namespace {

void on_data(ma_device* device, void* out, const void* in, ma_uint32 frames) {
    auto* sides = static_cast<Device::Sides*>(device->pUserData);
    if (!sides) return;
    // Capture first: the input for this block is already here, and pushing it
    // before the mix keeps the two in step even if the mix is the slow part.
    if (sides->recorder && in)
        sides->recorder->push(static_cast<const float*>(in),
                              static_cast<int64_t>(frames), sides->capture_channels);
    if (sides->mixer)
        sides->mixer->process(static_cast<float*>(out), static_cast<int64_t>(frames));
}

}  // namespace

Device::~Device() { stop(); }

bool Device::input_muted() {
    // RPC_E_CHANGED_MODE only means COM was already initialised on this thread
    // with the other apartment model; the interfaces below work regardless.
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED |
                                                   COINIT_DISABLE_OLE1DDE);
    if (FAILED(co) && co != RPC_E_CHANGED_MODE) return false;
    const bool owns_com = SUCCEEDED(co);

    bool muted = false;
    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDevice* endpoint = nullptr;
    IAudioEndpointVolume* volume = nullptr;
    if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                   __uuidof(IMMDeviceEnumerator),
                                   reinterpret_cast<void**>(&enumerator))) &&
        SUCCEEDED(enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &endpoint)) &&
        SUCCEEDED(endpoint->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL,
                                     nullptr, reinterpret_cast<void**>(&volume)))) {
        BOOL is_muted = FALSE;
        if (SUCCEEDED(volume->GetMute(&is_muted))) muted = is_muted != FALSE;
    }
    if (volume) volume->Release();
    if (endpoint) endpoint->Release();
    if (enumerator) enumerator->Release();
    if (owns_com) CoUninitialize();
    return muted;
}

std::vector<std::string> Device::capture_devices() {
    std::vector<std::string> out;
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) return out;
    ma_device_info* playback = nullptr;
    ma_device_info* capture = nullptr;
    ma_uint32 pn = 0, cn = 0;
    if (ma_context_get_devices(&ctx, &playback, &pn, &capture, &cn) == MA_SUCCESS)
        for (ma_uint32 i = 0; i < cn; ++i)
            out.emplace_back(capture[i].name);
    ma_context_uninit(&ctx);
    return out;
}

bool Device::start(Mixer& mixer, uint32_t preferred_rate, Recorder* recorder,
                   int capture_index) {
    stop();

    const bool want_capture = recorder != nullptr;
    ma_device_config cfg = ma_device_config_init(
        want_capture ? ma_device_type_duplex : ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = channels_;
    // Held for as long as the config points into it.
    ma_context ctx;
    bool have_ctx = false;
    ma_device_info* dev_capture = nullptr;
    ma_uint32 capture_count = 0;
    if (want_capture && capture_index >= 0 &&
        ma_context_init(nullptr, 0, nullptr, &ctx) == MA_SUCCESS) {
        have_ctx = true;
        ma_device_info* playback = nullptr;
        ma_uint32 pn = 0;
        ma_context_get_devices(&ctx, &playback, &pn, &dev_capture, &capture_count);
    }

    if (want_capture) {
        if (dev_capture && capture_index >= 0 &&
            capture_index < static_cast<int>(capture_count))
            cfg.capture.pDeviceID = &dev_capture[capture_index].id;
        cfg.capture.format = ma_format_f32;
        // 0 means "whatever the device has". Asking for two on a mono
        // interface is how an input device fails to open at all.
        cfg.capture.channels = 0;
        cfg.capture.shareMode = ma_share_mode_shared;
    }
    cfg.sampleRate        = preferred_rate;
    cfg.dataCallback      = &on_data;

    sides_ = new Device::Sides{&mixer, recorder, 0};
    cfg.pUserData = sides_;

    device_ = new ma_device{};
    if (ma_device_init(nullptr, &cfg, device_) != MA_SUCCESS) {
        // A machine with no microphone still deserves playback. Falling back
        // rather than failing is the difference between "recording is
        // unavailable" and "the app has no sound".
        if (want_capture) {
            delete device_;
            device_ = nullptr;
            delete sides_;
            sides_ = nullptr;
            capture_channels_ = 0;
            if (have_ctx) ma_context_uninit(&ctx);
            const bool ok = start(mixer, preferred_rate, nullptr);
            if (ok) error_ = "no input device -- recording is unavailable";
            return ok;
        }
        error_ = "no output device";
        delete device_;
        device_ = nullptr;
        delete sides_;
        sides_ = nullptr;
        return false;
    }
    capture_channels_ = want_capture ? device_->capture.channels : 0;
    capture_name_ = want_capture ? device_->capture.name : std::string{};
    sides_->capture_channels = capture_channels_;
    if (have_ctx) ma_context_uninit(&ctx);

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
    // Only after uninit: the callback dereferences this every block, and
    // freeing it while the device still runs is a use-after-free on the audio
    // thread, which crashes somewhere else entirely.
    delete sides_;
    sides_ = nullptr;
    running_ = false;
    capture_channels_ = 0;
    capture_name_.clear();
    mixer_ = nullptr;
}

}  // namespace mx
