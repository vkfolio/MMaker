// Phase 0 spike: prove the three seams a DAW needs from this framework.
//
// Not a product. Three panels, each answering one question that would be
// expensive to discover later:
//
//   1. AUDIO   -- can miniaudio drive WASAPI while a real-time callback
//                 publishes a playhead the UI can read without locking?
//   2. ASYNC   -- can a worker thread hand results to a strictly
//                 single-threaded UI, and wake it from an idle sleep?
//   3. RENDER  -- can paint_skia draw a waveform-scale point cloud at 60 fps
//                 while being panned and zoomed?
//
// The rules being tested are the ones that bite later: the audio callback
// allocates nothing and locks nothing, workers never touch an entity, and the
// draw surface is one canvas rather than thousands of elements.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <format>
#include <mutex>
#include <numbers>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "miniaudio.h"
#include "vikui/vikui.h"
#include "vikui/elements/canvas.h"

#include <include/core/SkCanvas.h>
#include <include/core/SkPaint.h>
#include <include/core/SkPoint.h>

namespace {

constexpr int kSampleRate = 48000;

// ---------------------------------------------------------------------------
// 1. AUDIO
// ---------------------------------------------------------------------------

/// Everything the audio callback may touch. No allocation, no locks, no
/// shared_ptr traffic -- the callback runs on a real-time thread and anything
/// that can block is a click waiting to happen.
struct AudioState {
    std::atomic<bool>     playing{false};
    std::atomic<uint64_t> frame{0};      // frames rendered, published for the UI
    std::atomic<float>    peak{0.0f};    // accumulate-and-clear, never a plain store
    std::atomic<uint32_t> xruns{0};
    std::atomic<uint32_t> latency_frames{0};

    // A ramp, not a switch: flipping gain instantly is the classic edit click.
    float gain = 0.0f;
    double phase = 0.0;
};

AudioState g_audio;

void audio_callback(ma_device* device, void* out, const void*, ma_uint32 count) {
    auto* dst = static_cast<float*>(out);
    const bool want = g_audio.playing.load(std::memory_order_relaxed);
    const float target = want ? 0.25f : 0.0f;
    // ~5 ms ramp so start/stop does not click.
    const float step = 1.0f / (0.005f * kSampleRate);

    const double omega = 2.0 * std::numbers::pi * 220.0 / kSampleRate;
    float block_peak = 0.0f;

    for (ma_uint32 i = 0; i < count; ++i) {
        g_audio.gain += std::clamp(target - g_audio.gain, -step, step);
        const auto s = static_cast<float>(std::sin(g_audio.phase)) * g_audio.gain;
        g_audio.phase += omega;
        if (g_audio.phase > 2.0 * std::numbers::pi) g_audio.phase -= 2.0 * std::numbers::pi;

        dst[i * 2 + 0] = s;
        dst[i * 2 + 1] = s;
        block_peak = std::max(block_peak, std::abs(s));
    }

    if (want) g_audio.frame.fetch_add(count, std::memory_order_relaxed);

    // A plain store loses ~5 of every 6 blocks at 60 Hz UI polling, which is
    // exactly the transients a peak meter exists to catch.
    float prev = g_audio.peak.load(std::memory_order_relaxed);
    while (block_peak > prev &&
           !g_audio.peak.compare_exchange_weak(prev, block_peak,
                                               std::memory_order_relaxed)) {
    }
    (void)device;
}

struct AudioEngine {
    ma_device device{};
    bool ok = false;
    std::string backend = "none";

    bool start() {
        ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
        cfg.playback.format = ma_format_f32;
        cfg.playback.channels = 2;
        cfg.sampleRate = kSampleRate;
        cfg.dataCallback = audio_callback;
        if (ma_device_init(nullptr, &cfg, &device) != MA_SUCCESS) return false;
        if (ma_device_start(&device) != MA_SUCCESS) {
            ma_device_uninit(&device);
            return false;
        }
        backend = ma_get_backend_name(device.pContext->backend);

        // The playhead must be what the user HEARS, not what we have rendered.
        ma_uint32 latency = device.playback.internalPeriodSizeInFrames *
                            device.playback.internalPeriods;
        g_audio.latency_frames.store(latency, std::memory_order_relaxed);
        ok = true;
        return true;
    }

    void stop() {
        if (ok) { ma_device_uninit(&device); ok = false; }
    }
};

AudioEngine g_engine;

/// What the user hears, in seconds: rendered position minus device latency.
double heard_seconds() {
    const auto rendered = g_audio.frame.load(std::memory_order_relaxed);
    const auto latency = g_audio.latency_frames.load(std::memory_order_relaxed);
    const auto heard = rendered > latency ? rendered - latency : 0;
    return static_cast<double>(heard) / kSampleRate;
}

// ---------------------------------------------------------------------------
// 2. ASYNC
// ---------------------------------------------------------------------------

/// A closed message type, deliberately: a std::function here would let a worker
/// capture and refcount a UI handle, which is the bug class this whole
/// arrangement exists to prevent.
struct WorkerMessage {
    int  id = 0;
    int  points = 0;
    double took_ms = 0.0;
};

struct ResultQueue {
    std::mutex mutex;
    std::deque<WorkerMessage> items;
    std::atomic<int> posted{0};

    void push(WorkerMessage m) {
        {
            std::lock_guard lock(mutex);
            items.push_back(m);
        }
        posted.fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<WorkerMessage> drain() {
        std::lock_guard lock(mutex);
        std::vector<WorkerMessage> out(items.begin(), items.end());
        items.clear();
        return out;
    }
};

ResultQueue g_results;

// ---------------------------------------------------------------------------
// 3. RENDER
// ---------------------------------------------------------------------------

/// Stand-in for a waveform peak pyramid: min/max per column, drawn as one
/// batched call rather than one element per column.
std::vector<SkPoint> build_points(int columns, double zoom, double scroll) {
    std::vector<SkPoint> pts;
    pts.reserve(static_cast<size_t>(columns) * 2);
    for (int i = 0; i < columns; ++i) {
        const double t = scroll + i * zoom;
        const float a = static_cast<float>(std::sin(t * 0.07) * std::sin(t * 0.011));
        const float b = static_cast<float>(std::sin(t * 0.13 + 1.7));
        pts.push_back(SkPoint::Make(static_cast<float>(i), a));
        pts.push_back(SkPoint::Make(static_cast<float>(i), b));
    }
    return pts;
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

struct Spike {
    // audio
    bool playing = false;

    // async
    int  jobs_started = 0;
    int  jobs_landed = 0;
    double last_job_ms = 0.0;
    bool woke_from_idle = false;

    // render
    int    columns = 10000;
    double zoom = 1.0;
    double scroll = 0.0;
    double last_paint_ms = 0.0;
    double worst_paint_ms = 0.0;
    int    frames = 0;
    int    ticks  = 0;

    void report_line() const {
        std::printf(
            "SEAM audio_ok=%d backend=%s heard=%.2fs latency_ms=%.1f peak=%.3f "
            "| worker_started=%d worker_landed=%d woke=%d "
            "| columns=%d paint_ms=%.2f worst_ms=%.2f frames=%d ticks=%d\n",
            g_engine.ok ? 1 : 0, g_engine.backend.c_str(), heard_seconds(),
            1000.0 * g_audio.latency_frames.load(std::memory_order_relaxed) / kSampleRate,
            g_audio.peak.load(std::memory_order_relaxed),
            jobs_started, jobs_landed, woke_from_idle ? 1 : 0,
            columns, last_paint_ms, worst_paint_ms, frames, ticks);
        std::fflush(stdout);
    }

    void pump(vik::App& app) {
        for (const auto& m : g_results.drain()) {
            ++jobs_landed;
            last_job_ms = m.took_ms;
            woke_from_idle = true;
            (void)m;
        }
        (void)app;
    }

    void spawn_worker(vik::App& app) {
        ++jobs_started;
        const int id = jobs_started;
        auto* platform = &app.platform();
        // The worker gets ids and plain data -- never a Handle, never the App.
        std::thread([id, platform] {
            const auto t0 = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            volatile double sink = 0;
            for (int i = 0; i < 2'000'000; ++i) sink += std::sin(i * 0.001);
            const auto t1 = std::chrono::steady_clock::now();
            g_results.push(WorkerMessage{
                id, 2'000'000,
                std::chrono::duration<double, std::milli>(t1 - t0).count()});
            // Without this the UI can sit in SDL_WaitEventTimeout until the
            // user happens to move the mouse.
            platform->wake();
        }).detach();
    }

    vik::AnyElement render(vik::Window&, vik::Context<Spike>& cx) {
        const double heard = heard_seconds();
        const float peak = g_audio.peak.exchange(0.0f, std::memory_order_relaxed);
        const auto latency_ms =
            1000.0 * g_audio.latency_frames.load(std::memory_order_relaxed) / kSampleRate;

        auto stat = [](const std::string& label, const std::string& value) {
            return vik::div().flex_row().justify_between().gap_4()
                .child(vik::text(label).text_color(vik::rgb(0x8d94a3)))
                .child(vik::text(value).text_color(vik::rgb(0xe6e8ec)));
        };

        auto panel = [](const std::string& title) {
            return vik::div().flex_col().gap_2().p_4().rounded_md()
                .bg(vik::rgb(0x1b1e27)).border_1().border_color(vik::rgb(0x2c313d))
                .child(vik::text(title).text_color(vik::rgb(0xd9903c)));
        };

        // --- 3. RENDER: one canvas, one batched draw ------------------------
        auto* self = this;
        auto surface =
            vik::canvas([self](vik::Bounds bounds, vik::Window& w, vik::App&) {
                const auto t0 = std::chrono::steady_clock::now();
                const int columns = self->columns;
                const double zoom = self->zoom;
                const double scroll = self->scroll;
                auto pts = build_points(columns, zoom, scroll);

                w.paint_skia([bounds, pts = std::move(pts), columns](SkCanvas* c) {
                    const float x0 = bounds.origin.x;
                    const float y0 = bounds.origin.y;
                    const float wpx = bounds.size.width;
                    const float hpx = bounds.size.height;
                    const float mid = y0 + hpx * 0.5f;
                    const float sx = columns > 1 ? wpx / (columns - 1) : 1.0f;

                    SkPaint bg;
                    bg.setColor(SkColorSetRGB(0x14, 0x16, 0x1d));
                    c->drawRect(SkRect::MakeXYWH(x0, y0, wpx, hpx), bg);

                    // Pixel rects computed analytically, not via a scaled CTM:
                    // a matrix would blur 1px lines and re-raster glyphs per zoom.
                    std::vector<SkPoint> screen;
                    screen.reserve(pts.size());
                    for (size_t i = 0; i < pts.size(); ++i) {
                        screen.push_back(SkPoint::Make(
                            x0 + pts[i].fX * sx,
                            mid + pts[i].fY * hpx * 0.45f));
                    }

                    SkPaint wave;
                    wave.setColor(SkColorSetRGB(0x73, 0x7d, 0x92));
                    wave.setStrokeWidth(1.0f);
                    wave.setAntiAlias(false);
                    c->drawPoints(SkCanvas::kLines_PointMode,
                                  SkSpan<const SkPoint>(screen.data(), screen.size()),
                                  wave);
                });

                const auto t1 = std::chrono::steady_clock::now();
                self->last_paint_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                self->worst_paint_ms =
                    std::max(self->worst_paint_ms, self->last_paint_ms);
                ++self->frames;
            }).size_full();

        auto button = [&cx](const char* id, const std::string& label, auto handler) {
            return vik::div().id(id).px_3().py_1().rounded_md()
                .bg(vik::rgb(0x21252f)).border_1().border_color(vik::rgb(0x3b4250))
                .cursor_pointer()
                .hover([](vik::StyleRefinement& s) { s.bg(vik::rgb(0x2c313d)); })
                .on_click(cx.listener(handler))
                .child(vik::text(label).text_color(vik::rgb(0xe6e8ec)));
        };

        return vik::div().size_full().flex_col().gap_3().p_4()
            .bg(vik::rgb(0x14161d))
            .child(vik::text("musicX Studio - Phase 0 system check")
                       .text_xl().text_color(vik::white()))
            .child(
                vik::div().flex_row().gap_3()
                    .child(panel("1. AUDIO (miniaudio -> device)")
                        .w_px(330.0f)
                        .child(stat("backend", g_engine.backend))
                        .child(stat("device ok", g_engine.ok ? "yes" : "NO"))
                        .child(stat("heard position", std::format("{:.2f}s", heard)))
                        .child(stat("output latency", std::format("{:.1f} ms", latency_ms)))
                        .child(stat("peak", std::format("{:.3f}", peak)))
                        .child(stat("xruns", std::to_string(
                            g_audio.xruns.load(std::memory_order_relaxed))))
                        .child(button("play", playing ? "Stop" : "Play",
                            [](Spike& s, const vik::ClickEvent&, vik::Window&,
                               vik::Context<Spike>& c) {
                                s.playing = !s.playing;
                                g_audio.playing.store(s.playing,
                                                      std::memory_order_relaxed);
                                c.notify();
                            })))
                    .child(panel("2. ASYNC (worker -> wake -> UI)")
                        .w_px(330.0f)
                        .child(stat("started", std::to_string(jobs_started)))
                        .child(stat("landed", std::to_string(jobs_landed)))
                        .child(stat("last job", std::format("{:.0f} ms", last_job_ms)))
                        .child(stat("woke UI", woke_from_idle ? "yes" : "not yet"))
                        .child(button("work", "Run worker",
                            [](Spike& s, const vik::ClickEvent&, vik::Window&,
                               vik::Context<Spike>& c) {
                                s.spawn_worker(c.app());
                                c.notify();
                            })))
                    .child(panel("3. RENDER (paint_skia)")
                        .w_px(330.0f)
                        .child(stat("columns", std::to_string(columns)))
                        .child(stat("points", std::to_string(columns * 2)))
                        .child(stat("paint", std::format("{:.2f} ms", last_paint_ms)))
                        .child(stat("worst", std::format("{:.2f} ms", worst_paint_ms)))
                        .child(stat("frames", std::to_string(frames)))
                        .child(button("more", "10k columns",
                            [](Spike& s, const vik::ClickEvent&, vik::Window&,
                               vik::Context<Spike>& c) {
                                s.columns = s.columns >= 10000 ? 5000 : 10000;
                                s.worst_paint_ms = 0;
                                c.notify();
                            }))))
            .child(vik::div().flex_1().rounded_md().overflow_hidden()
                       .border_1().border_color(vik::rgb(0x2c313d))
                       .child(std::move(surface)))
            .into_any();
    }
};

}  // namespace

int main(int, char**) {
    if (!g_engine.start()) {
        // Not fatal: the render and async panels still answer their questions.
        g_engine.backend = "FAILED TO OPEN";
    }

    vik::App::run([](vik::App& app) {
        app.open_window(
            vik::WindowOptions{.title = "musicX Studio - Phase 0 system check",
                               .size = {1180.0f, 760.0f}},
            [](vik::Window&, vik::App& app) {
                auto handle = app.add_entity<Spike>();

                // Drives the playhead while rolling, and drains worker results.
                // A real app cancels this on stop; the spike keeps it simple.
                // Self-drive: start audio, fire a worker, and report, so a
                // plain run is evidence rather than something to click through.
                app.set_timer(std::chrono::milliseconds(300), [handle](vik::App& a) {
                    a.update_entity(handle, [&a](Spike& s, vik::Context<Spike>& c) {
                        s.playing = true;
                        g_audio.playing.store(true, std::memory_order_relaxed);
                        s.spawn_worker(a);
                        c.notify();
                    });
                });
                app.set_interval(std::chrono::milliseconds(1000), [handle](vik::App& a) {
                    a.read_entity(handle).report_line();
                    return true;
                });
                app.set_interval(std::chrono::milliseconds(16),
                                 [handle](vik::App& a) {
                                     a.update_entity(handle, [&a](Spike& s,
                                                                  vik::Context<Spike>& c) {
                                         ++s.ticks;
                                         s.zoom = 1.0 + 0.5 * std::sin(s.ticks * 0.05);
                                         s.pump(a);
                                         c.notify();
                                     });
                                     return true;   // keep repeating
                                 });
                return handle;
            });
    });

    g_engine.stop();
    return 0;
}
