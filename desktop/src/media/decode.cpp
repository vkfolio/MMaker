#include "decode.h"

#include <vector>

#include "miniaudio.h"

namespace mx {

Media decode_file(const std::filesystem::path& path, uint32_t device_rate,
                  uint32_t channels) {
    Media out;

    const std::string utf8 = path.string();

    // Open once at the file's own rate purely to learn what that rate is.
    // Recording it matters: it is what makes a device-rate change recoverable
    // by re-decoding rather than re-downloading. Reading it back off a decoder
    // that has already been told to convert would just report the target rate
    // again, which would be a lie stored in a field named `source_rate`.
    {
        ma_decoder_config native = ma_decoder_config_init(ma_format_f32, channels, 0);
        ma_decoder probe;
        if (ma_decoder_init_file(utf8.c_str(), &native, &probe) == MA_SUCCESS) {
            out.source_rate = probe.outputSampleRate;
            ma_decoder_uninit(&probe);
        }
    }

    // Now ask for exactly what the mixer wants. miniaudio resamples on the way
    // out, so there is one conversion here rather than one per playback.
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, channels, device_rate);

    ma_decoder decoder;
    if (ma_decoder_init_file(utf8.c_str(), &cfg, &decoder) != MA_SUCCESS) {
        out.error = "cannot open or decode: " + utf8;
        return out;
    }

    auto buf = std::make_shared<AudioBuffer>();
    buf->channels    = channels;
    buf->rate        = device_rate;
    buf->source_rate = out.source_rate ? out.source_rate : device_rate;

    constexpr ma_uint64 kChunk = 1 << 16;
    std::vector<float> chunk(static_cast<size_t>(kChunk) * channels);
    for (;;) {
        ma_uint64 read = 0;
        const ma_result r = ma_decoder_read_pcm_frames(&decoder, chunk.data(), kChunk, &read);
        if (read > 0) {
            buf->samples.insert(buf->samples.end(), chunk.begin(),
                                chunk.begin() + static_cast<size_t>(read * channels));
        }
        if (r != MA_SUCCESS || read < kChunk) break;
    }
    ma_decoder_uninit(&decoder);

    buf->frames = static_cast<int64_t>(buf->samples.size() / channels);
    if (buf->frames <= 0) {
        out.error = "decoded to zero frames: " + utf8;
        return out;
    }

    out.seconds = static_cast<double>(buf->frames) / static_cast<double>(device_rate);
    out.peaks   = build_peaks(buf->data(), buf->frames, channels);
    out.buffer  = buf;
    return out;
}

}  // namespace mx
