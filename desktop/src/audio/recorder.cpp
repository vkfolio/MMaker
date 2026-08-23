#include "recorder.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "miniaudio.h"

namespace mx {
namespace {

/// About a second of stereo at 48 kHz, rounded to a power of two so the index
/// arithmetic is a mask rather than a division.
constexpr size_t kRingFrames = 131072;

}  // namespace

Recorder::~Recorder() { stop(); }

bool Recorder::start(const std::filesystem::path& path, uint32_t rate,
                     uint32_t channels) {
    stop();
    error_.clear();
    channels_ = channels ? channels : 1;
    path_ = path;

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    ma_encoder_config cfg = ma_encoder_config_init(ma_encoding_format_wav,
                                                   ma_format_f32, channels_, rate);
    encoder_ = new ma_encoder{};
    if (ma_encoder_init_file(path.string().c_str(), &cfg, encoder_) != MA_SUCCESS) {
        error_ = "could not open " + path.string() + " for recording";
        delete encoder_;
        encoder_ = nullptr;
        return false;
    }

    // Preallocated once. Everything the audio thread touches must already
    // exist before it is allowed to run.
    ring_.assign(kRingFrames * channels_, 0.0f);
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
    written_.store(0, std::memory_order_relaxed);
    overruns_.store(0, std::memory_order_relaxed);
    peak_.store(0.0f, std::memory_order_relaxed);

    recording_.store(true, std::memory_order_release);
    writer_ = std::thread([this] { drain_loop(); });
    return true;
}

void Recorder::stop() {
    if (!recording_.load(std::memory_order_acquire)) {
        // Still tidy up a half-open encoder from a failed start.
        if (encoder_) {
            ma_encoder_uninit(encoder_);
            delete encoder_;
            encoder_ = nullptr;
        }
        return;
    }
    recording_.store(false, std::memory_order_release);
    if (writer_.joinable()) writer_.join();   // drains what is left before closing
    if (encoder_) {
        ma_encoder_uninit(encoder_);          // patches the RIFF sizes
        delete encoder_;
        encoder_ = nullptr;
    }
}

void Recorder::push(const float* input, int64_t frames, uint32_t channels) {
    if (!input || frames <= 0) return;
    if (!recording_.load(std::memory_order_acquire)) return;

    const uint32_t ch = channels_;
    const size_t capacity = ring_.size();
    if (capacity == 0) return;

    const uint64_t head = head_.load(std::memory_order_relaxed);
    const uint64_t tail = tail_.load(std::memory_order_acquire);
    const size_t used = static_cast<size_t>(head - tail);
    const size_t free_samples = capacity > used ? capacity - used : 0;
    const size_t wanted = static_cast<size_t>(frames) * ch;

    if (wanted > free_samples) {
        // The writer fell behind. Counted, not hidden: a take with a hole in it
        // and no explanation is worse than one that says it dropped audio.
        overruns_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    float peak = 0.0f;
    for (int64_t f = 0; f < frames; ++f) {
        for (uint32_t c = 0; c < ch; ++c) {
            // The device may hand us more or fewer channels than we write; take
            // what exists and duplicate the last one rather than reading past
            // the end of the block.
            const uint32_t src_c = c < channels ? c : channels - 1;
            const float sample = input[f * channels + src_c];
            peak = std::max(peak, std::fabs(sample));
            ring_[static_cast<size_t>((head + static_cast<uint64_t>(f) * ch + c) %
                                      capacity)] = sample;
        }
    }
    head_.store(head + wanted, std::memory_order_release);

    // Accumulate rather than store: at 375 blocks a second against a 60 Hz
    // reader, a plain store loses five of every six blocks -- exactly the
    // transients a peak meter exists to catch.
    float seen = peak_.load(std::memory_order_relaxed);
    while (peak > seen &&
           !peak_.compare_exchange_weak(seen, peak, std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) {
    }
}

void Recorder::drain_loop() {
    std::vector<float> block(8192);
    for (;;) {
        const uint64_t head = head_.load(std::memory_order_acquire);
        const uint64_t tail = tail_.load(std::memory_order_relaxed);
        const size_t available = static_cast<size_t>(head - tail);

        if (available == 0) {
            if (!recording_.load(std::memory_order_acquire)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }

        const size_t capacity = ring_.size();
        const size_t take = std::min(available, block.size());
        for (size_t i = 0; i < take; ++i)
            block[i] = ring_[static_cast<size_t>((tail + i) % capacity)];

        const ma_uint64 frames = take / channels_;
        if (frames > 0 && encoder_) {
            ma_uint64 done = 0;
            ma_encoder_write_pcm_frames(encoder_, block.data(), frames, &done);
            written_.fetch_add(static_cast<int64_t>(done), std::memory_order_relaxed);
        }
        tail_.store(tail + frames * channels_, std::memory_order_release);
    }
}

}  // namespace mx
