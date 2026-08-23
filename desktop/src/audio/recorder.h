// Recorder -- capture from the input device to a file, without ever blocking
// the audio thread.
//
// The contract is the same one the mixer holds: the callback may not allocate,
// lock, or touch the filesystem. So capture pushes into a preallocated
// single-producer ring and returns; a writer thread drains it to disk. The ring
// is sized for roughly a second, which is far more than the few milliseconds a
// scheduling hiccup can cost, and an overrun is counted rather than papered
// over -- silently dropping input would show up as a take with a hole in it
// and no explanation.
//
// The file is written as it arrives rather than buffered and saved at the end,
// so a crash mid-take costs the tail of the take rather than the take. That is
// the whole reason recording is a milestone and not a feature: the failure mode
// of getting it wrong is losing a performance nobody can repeat.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

struct ma_encoder;

namespace mx {

class Recorder {
public:
    ~Recorder();

    /// Opens `path` and starts the writer thread. Channel count is what the
    /// capture device gives; the caller passes what the device reported.
    bool start(const std::filesystem::path& path, uint32_t rate, uint32_t channels);

    /// Closes the file and joins the writer. Safe to call twice.
    void stop();

    /// Audio-thread side. Never blocks, never allocates.
    void push(const float* input, int64_t frames, uint32_t channels);

    bool     recording() const { return recording_.load(std::memory_order_acquire); }
    int64_t  frames_written() const { return written_.load(std::memory_order_relaxed); }
    uint64_t overruns() const { return overruns_.load(std::memory_order_relaxed); }
    const std::string& error() const { return error_; }
    const std::filesystem::path& path() const { return path_; }

    /// Peak of the last block pushed, for an input meter. Accumulated on the
    /// audio thread and cleared by the reader, so a quiet block cannot erase a
    /// transient the meter existed to show.
    float take_peak() { return peak_.exchange(0.0f, std::memory_order_acq_rel); }

    /// The loudest sample of the whole take, never cleared.
    ///
    /// A take that came back at denormal level is a muted or blocked input,
    /// not a quiet performance, and saying so beats handing back a clip that
    /// looks like a recording and plays as nothing.
    float session_peak() const { return loudest_.load(std::memory_order_relaxed); }

private:
    void drain_loop();

    std::vector<float>    ring_;
    std::atomic<uint64_t> head_{0};      // written by the audio thread
    std::atomic<uint64_t> tail_{0};      // written by the writer thread
    std::atomic<bool>     recording_{false};
    std::atomic<int64_t>  written_{0};
    std::atomic<uint64_t> overruns_{0};
    std::atomic<float>    peak_{0.0f};
    std::atomic<float>    loudest_{0.0f};

    ma_encoder*  encoder_ = nullptr;
    std::thread  writer_;
    uint32_t     channels_ = 2;
    std::filesystem::path path_;
    std::string  error_;
};

}  // namespace mx
