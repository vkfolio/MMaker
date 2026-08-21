#include "bounce.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "miniaudio.h"

namespace mx {

BounceResult bounce(Session& session, const std::filesystem::path& path,
                    double tail_seconds) {
    BounceResult out;

    const int64_t rate = session.rate ? session.rate : 48000;
    const int64_t length = session.length_frames();
    if (length <= 0) {
        out.error = "nothing to bounce -- the session has no audio";
        return out;
    }
    const int64_t total = length + static_cast<int64_t>(tail_seconds * rate);

    // Same mixer, same graph builder, same generation discipline as playback.
    Mixer mixer;
    mixer.prepare(static_cast<uint32_t>(rate), 2);
    mixer.publish(build_graph(session, 1));
    mixer.begin_offline();
    mixer.transport.playing.store(true);
    mixer.transport.looping.store(false);

    ma_encoder_config cfg =
        ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 2,
                               static_cast<ma_uint32>(rate));
    ma_encoder encoder;
    const std::string utf8 = path.string();
    if (ma_encoder_init_file(utf8.c_str(), &cfg, &encoder) != MA_SUCCESS) {
        out.error = "cannot open " + utf8 + " for writing";
        return out;
    }

    // A larger block than the device uses. Block size must not change the
    // result -- if it does, something is carrying state across calls that
    // should not be, and this is where that would show up.
    constexpr int64_t kBlock = 4096;
    std::vector<float> buffer(static_cast<size_t>(kBlock) * 2);

    const auto started = std::chrono::steady_clock::now();
    int64_t written = 0;
    while (written < total) {
        const int64_t want = std::min(kBlock, total - written);
        mixer.process(buffer.data(), want);

        for (int64_t i = 0; i < want * 2; ++i) {
            const float v = std::abs(buffer[i]);
            if (v > out.peak) out.peak = v;
            if (v >= 0.999f) out.clipped = true;
        }

        ma_uint64 done = 0;
        ma_encoder_write_pcm_frames(&encoder, buffer.data(),
                                    static_cast<ma_uint64>(want), &done);
        if (done == 0) {
            ma_encoder_uninit(&encoder);
            out.error = "write failed partway through the bounce";
            return out;
        }
        written += static_cast<int64_t>(done);
    }
    const auto finished = std::chrono::steady_clock::now();
    ma_encoder_uninit(&encoder);

    out.ok = true;
    out.frames = written;
    out.seconds = static_cast<double>(written) / static_cast<double>(rate);
    out.render_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    out.realtime_ratio = out.render_ms > 0.0
                             ? (out.seconds * 1000.0) / out.render_ms
                             : 0.0;
    return out;
}

}  // namespace mx
