// musicX Studio -- Phase 1 slice: transport, timeline, playback.
//
// What this is: local WAVs on a real timeline, with real waveforms, a smooth
// latency-compensated playhead, and editing that stays click-free while the
// transport rolls. That last property is the one the plan calls the gate, and
// it is held to by tests/mixer_test.cpp rather than by listening.
//
// What this is not, yet: no server, no undo, no trimming, no recording. Those
// are later phases and are deliberately absent rather than half-present.
//
// Usage:
//   musicx                    -- synthesised demo content, so it always runs
//   musicx <folder>           -- loads every .wav/.mp3/.flac in the folder
//   musicx --selftest <fldr>  -- loads, plays, reports, exits (no clicking)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <format>
#include <mutex>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include "audio/device.h"
#include "audio/mixer.h"
#include "media/decode.h"
#include "session.h"
#include "ui/arrangement.h"

#include "vikui/vikui.h"
#include "vikui/elements/canvas.h"
#include "components/phosphor.h"

#include <include/core/SkStream.h>
#include <include/core/SkSurface.h>
#include <include/encode/SkPngEncoder.h>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Worker -> UI
// ---------------------------------------------------------------------------

/// What a worker is allowed to send back: a closed set of plain data.
///
/// Deliberately not std::function<void(App&)>. A callable is exactly the type
/// that lets a worker capture and refcount a Handle<T>, which would defeat the
/// rule that workers never touch an entity. A variant cannot smuggle one.
struct DecodedMessage {
    mx::SourceId source_id = 0;
    mx::Media    media;
    double       decode_ms = 0.0;
};

class Inbox {
public:
    void push(DecodedMessage m) {
        std::lock_guard lock(mutex_);
        queue_.push_back(std::move(m));
    }
    std::vector<DecodedMessage> drain() {
        std::lock_guard lock(mutex_);
        std::vector<DecodedMessage> out(std::make_move_iterator(queue_.begin()),
                                        std::make_move_iterator(queue_.end()));
        queue_.clear();
        return out;
    }

private:
    std::mutex                 mutex_;
    std::deque<DecodedMessage> queue_;
};

Inbox      g_inbox;
mx::Mixer  g_mixer;
mx::Device g_device;

std::atomic<int> g_pending{0};

// ---------------------------------------------------------------------------
// Demo content, so the app runs with nothing to point it at
// ---------------------------------------------------------------------------

mx::BufferRef synth(double seconds, double hz, float amp, uint32_t rate,
                    bool pluck) {
    auto buf = std::make_shared<mx::AudioBuffer>();
    buf->channels = 2;
    buf->rate = rate;
    buf->source_rate = rate;
    buf->frames = static_cast<int64_t>(seconds * rate);
    buf->samples.resize(static_cast<size_t>(buf->frames) * 2);

    const double omega = 2.0 * std::numbers::pi * hz / rate;
    const double beat  = rate * 0.5;
    for (int64_t i = 0; i < buf->frames; ++i) {
        double env = 1.0;
        if (pluck) {
            const double into = std::fmod(static_cast<double>(i), beat);
            env = std::exp(-into / (rate * 0.08));
        }
        const float v = static_cast<float>(amp * env * std::sin(omega * i));
        buf->samples[static_cast<size_t>(i) * 2]     = v;
        buf->samples[static_cast<size_t>(i) * 2 + 1] = v;
    }
    return buf;
}

// ---------------------------------------------------------------------------

struct Studio {
    mx::Session session;
    mx::View    view;

    uint64_t   generation = 1;
    mx::ClipId selected   = 0;

    // Drag state. Held here rather than in the graph, so a drag in progress is
    // a property of the UI and cannot reach the audio thread half-applied.
    mx::ClipId dragging     = 0;
    int64_t    drag_grab    = 0;     // offset from clip start to grab point
    bool       scrubbing    = false;

    double last_paint_ms  = 0.0;
    double worst_paint_ms = 0.0;
    int    frames         = 0;
    int64_t columns       = 0;
    double  build_ms      = 0.0;
    double  submit_ms     = 0.0;
    double  last_decode_ms = 0.0;
    int     loaded_sources = 0;
    float   meter          = 0.0f;

    bool  fitted      = false;
    float timeline_px = 1100.0f;   // last measured canvas width

    bool selftest = false;
    int  reports  = 0;

    // -- content ------------------------------------------------------------

    void load_folder(const fs::path& folder, vik::App& app) {
        std::vector<fs::path> files;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(folder, ec)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".mp3" || ext == ".flac")
                files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());

        // Nothing usable in there -- a bad path, or a folder with no
        // audio. Bail before anything touches files.front().
        if (files.empty()) { build_demo(); return; }

        // Exported stem sets share a render id, so every filename starts with
        // the same 37 characters. Left alone, every track header reads
        // "283a6411-cef6-...-vocals" and the only part that identifies the
        // track is off the end of the label. Strip whatever prefix they all
        // share -- but only up to a separator, so files that merely happen to
        // start with the same letters are not truncated mid-word.
        std::string shared = files.front().stem().string();
        for (const auto& f : files) {
            const std::string name = f.stem().string();
            size_t i = 0;
            while (i < shared.size() && i < name.size() && shared[i] == name[i]) ++i;
            shared.resize(i);
        }
        while (!shared.empty() && shared.back() != '_' && shared.back() != '-' &&
               shared.back() != ' ')
            shared.pop_back();
        if (files.size() < 2) shared.clear();

        auto label_for = [&shared](const fs::path& f) {
            std::string name = f.stem().string();
            if (!shared.empty() && name.rfind(shared, 0) == 0)
                name = name.substr(shared.size());
            return name.empty() ? f.stem().string() : name;
        };

        static const uint32_t colors[] = {0xff4f8ef7, 0xffe0813c, 0xff56c08a,
                                          0xffc169d6, 0xffe05c72, 0xff40b8c4,
                                          0xffd9b23c, 0xff7f8cf0};
        int n = 0;
        for (const auto& file : files) {
            auto& track = session.add_track(label_for(file),
                                            colors[n % std::size(colors)]);
            auto& source = session.add_source(file, label_for(file));
            // Length is unknown until the decode lands; the clip is created
            // then, so a clip never exists pointing at audio that is not there.
            spawn_decode(source.id, file, app);
            (void)track;
            ++n;
        }
    }

    void build_demo() {
        const uint32_t rate = session.rate;
        struct Part { const char* name; double hz; float amp; bool pluck; uint32_t color; };
        const Part parts[] = {
            {"bass",  55.0,  0.35f, true,  0xff4f8ef7},
            {"keys",  220.0, 0.22f, true,  0xffe0813c},
            {"pad",   330.0, 0.14f, false, 0xff56c08a},
            {"lead",  440.0, 0.16f, true,  0xffc169d6},
        };
        int n = 0;
        for (const Part& p : parts) {
            auto& track = session.add_track(p.name, p.color);
            auto& source = session.add_source({}, p.name);
            source.buffer = synth(8.0, p.hz, p.amp, rate, p.pluck);
            source.peaks = mx::build_peaks(source.buffer->data(), source.buffer->frames,
                                           source.buffer->channels);
            source.source_rate = rate;
            auto& clip = session.add_clip(track.id, source.id,
                                          static_cast<int64_t>(n) * rate,
                                          source.buffer->frames);
            clip.name = p.name;
            clip.fade_in = clip.fade_out = rate / 100;   // 10 ms, so edges are clean
            ++n;
            ++loaded_sources;
        }
        session.loop_end = session.length_frames();
    }

    void spawn_decode(mx::SourceId id, fs::path file, vik::App& app) {
        ++g_pending;
        auto* platform = &app.platform();
        const uint32_t rate = session.rate;
        // Plain data only: an id and a path. No handle, no App, no entity.
        std::thread([id, file = std::move(file), rate, platform] {
            const auto t0 = std::chrono::steady_clock::now();
            mx::Media media = mx::decode_file(file, rate);
            const auto t1 = std::chrono::steady_clock::now();
            g_inbox.push(DecodedMessage{
                id, std::move(media),
                std::chrono::duration<double, std::milli>(t1 - t0).count()});
            // Without this the UI can sit in SDL_WaitEventTimeout until the
            // user happens to move the mouse.
            platform->wake();
        }).detach();
    }

    /// Everything a worker produced is applied here, on the UI thread.
    bool pump() {
        bool changed = false;
        for (auto& msg : g_inbox.drain()) {
            --g_pending;
            last_decode_ms = msg.decode_ms;
            mx::Source* src = session.find_source(msg.source_id);
            if (!src) continue;
            if (!msg.media.ok()) {
                std::printf("decode failed: %s\n", msg.media.error.c_str());
                continue;
            }
            src->buffer      = msg.media.buffer;
            src->peaks       = std::move(msg.media.peaks);
            src->source_rate = msg.media.source_rate;

            // Stems from one render are aligned, so they all start at zero.
            mx::TrackId track = 0;
            for (size_t i = 0; i < session.sources.size(); ++i)
                if (session.sources[i].id == msg.source_id && i < session.tracks.size())
                    track = session.tracks[i].id;
            if (track) {
                auto& clip = session.add_clip(track, src->id, 0, src->buffer->frames);
                clip.name = src->label;
                clip.fade_in = clip.fade_out = session.rate / 200;
            }
            ++loaded_sources;
            changed = true;
        }
        if (changed) {
            session.loop_end = session.length_frames();
            // Fit on the first content to arrive. Re-fitting on every later
            // decode would yank the view out from under someone who had
            // already started navigating.
            if (!fitted && session.length_frames() > 0) {
                view.frames_per_pixel = mx::View::fit(session.length_frames(), timeline_px);
                fitted = true;
            }
            publish();
        }
        return changed;
    }

    void publish() { g_mixer.publish(mx::build_graph(session, ++generation)); }

    // -- transport ----------------------------------------------------------

    int64_t playhead() const { return g_mixer.heard_frame(); }

    void toggle_play() {
        const bool now = !g_mixer.transport.playing.load();
        g_mixer.transport.playing.store(now);
    }

    void seek(int64_t frame) {
        g_mixer.transport.seek_request.fetch_add(1);
        g_mixer.transport.seek_to.store(std::max<int64_t>(0, frame));
    }

    // -- hit-testing --------------------------------------------------------

    /// The inverse of the draw mapping, not a test against retained shapes.
    mx::ClipId clip_at(float x, float y) const {
        const int index = mx::track_at(y, static_cast<int>(session.tracks.size()));
        if (index < 0) return 0;
        const mx::TrackId track = session.tracks[index].id;
        const int64_t frame = view.frame_at(x);
        for (const auto& c : session.clips) {
            if (c.track_id != track) continue;
            if (frame >= c.start_frame && frame < c.start_frame + c.length) return c.id;
        }
        return 0;
    }

    void report() {
        std::printf(
            "STUDIO device=%s rate=%u latency_ms=%.1f | sources=%d pending=%d "
            "decode_ms=%.0f | tracks=%zu clips=%zu | playhead=%.2fs xruns=%u "
            "| paint_ms=%.2f (build %.2f skia %.2f) worst=%.2f frames=%d columns=%lld\n",
            g_device.running() ? g_device.backend().c_str() : "NONE",
            g_device.rate(), 1000.0 * g_device.latency_frames() /
                std::max(1u, g_device.rate()),
            loaded_sources, g_pending.load(), last_decode_ms,
            session.tracks.size(), session.clips.size(),
            static_cast<double>(playhead()) / std::max(1u, session.rate),
            g_mixer.xruns(), last_paint_ms, build_ms, submit_ms, worst_paint_ms, frames,
            static_cast<long long>(columns));
        std::fflush(stdout);
    }

    vik::AnyElement render(vik::Window&, vik::Context<Studio>& cx);
};

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

vik::AnyElement Studio::render(vik::Window&, vik::Context<Studio>& cx) {
    auto* self = this;

    const bool playing = g_mixer.transport.playing.load();
    const double pos_s = static_cast<double>(playhead()) / std::max(1u, session.rate);
    meter = std::max(meter * 0.85f, g_mixer.take_master_peak());

    // Transport controls read as icons, the way every DAW's do -- a row of
    // words is slower to scan than a row of shapes you already know.
    auto icon_button = [&cx](const char* id, const char* icon, bool active, auto fn) {
        return vik::div().id(id).px_3().py_2().rounded_md()
            .flex_row().items_center().justify_center()
            .bg(active ? vik::rgb(0x3a4a68) : vik::rgb(0x21252f))
            .border_1().border_color(vik::rgb(0x3b4250))
            .cursor_pointer()
            .hover([](vik::StyleRefinement& s) { s.bg(vik::rgb(0x2c313d)); })
            .on_click(cx.listener(fn))
            .child(vik::ui::phosphor(icon, vik::ui::PhWeight::Fill)
                       .size(16.0f)
                       .color(active ? vik::rgb(0xffffff) : vik::rgb(0xc9cedb)));
    };

    auto readout = [](const std::string& label, const std::string& value) {
        return vik::div().flex_col()
            .child(vik::text(label).text_xs().text_color(vik::rgb(0x6c7383)))
            .child(vik::text(value).text_color(vik::rgb(0xe6e8ec)));
    };

    // --- transport ----------------------------------------------------------
    const int64_t bar = session.frames_per_bar() > 0
                            ? playhead() / session.frames_per_bar() : 0;
    const int64_t beat_frames = std::max<int64_t>(
        1, session.frames_per_bar() / std::max(1, session.beats_per_bar));
    const int64_t beat = session.frames_per_bar() > 0
                             ? (playhead() % session.frames_per_bar()) / beat_frames : 0;

    auto transport =
        vik::div().flex_row().items_center().gap_4().px_4().py_2()
            .bg(vik::rgb(0x1b1e27)).border_1().border_color(vik::rgb(0x2c313d))
            .child(icon_button("rtz", "skip-back", false,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.seek(0); c.notify(); }))
            .child(icon_button("play", playing ? "pause" : "play", playing,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.toggle_play(); c.notify(); }))
            .child(icon_button("loop", "repeat",
                               g_mixer.transport.looping.load(),
                [](Studio&, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    auto& t = g_mixer.transport;
                    t.looping.store(!t.looping.load());
                    c.notify();
                }))
            .child(readout("position", std::format("{:.2f}s", pos_s)))
            .child(readout("bar.beat", std::format("{}.{}", bar + 1, beat + 1)))
            .child(readout("tempo", std::format("{:.0f} BPM", session.bpm)))
            .child(readout("device", g_device.running()
                                         ? std::format("{} {} Hz", g_device.backend(),
                                                       g_device.rate())
                                         : std::string("no output")))
            .child(readout("latency", std::format("{:.1f} ms",
                1000.0 * g_device.latency_frames() / std::max(1u, g_device.rate()))))
            // An xrun counter that is always visible, per the plan. A number you
            // have to go looking for is a number nobody looks at.
            .child(readout("xruns", std::to_string(g_mixer.xruns())))
            .child(readout("peak", std::format("{:.2f}", meter)))
            .child(readout("paint", std::format("{:.2f} ms", last_paint_ms)));

    // --- track headers: real elements, because they are chrome --------------
    auto headers = vik::div().flex_col().w_px(160.0f)
                       .bg(vik::rgb(0x1b1e27))
                       .border_1().border_color(vik::rgb(0x2c313d))
                       .child(vik::div().h_px(mx::kRulerHeight));

    for (size_t i = 0; i < session.tracks.size(); ++i) {
        const auto& t = session.tracks[i];
        const auto tid = t.id;
        headers = std::move(headers).child(
            vik::div().h_px(mx::kTrackHeight).mb(mx::kTrackGap)
                .flex_col().justify_center().gap_1().px_3()
                .bg(vik::rgb(0x1e222c))
                .child(vik::text(t.name).text_color(vik::rgb(0xe6e8ec)))
                .child(vik::div().flex_row().gap_2()
                    .child(vik::div().id("m").px_2().rounded_sm().cursor_pointer()
                        .bg(t.muted ? vik::rgb(0xc0503c) : vik::rgb(0x2c313d))
                        .on_click(cx.listener([tid](Studio& s, const vik::ClickEvent&,
                                                    vik::Window&,
                                                    vik::Context<Studio>& c) {
                            if (auto* tr = s.session.find_track(tid)) tr->muted = !tr->muted;
                            s.publish();
                            c.notify();
                        }))
                        .child(vik::ui::phosphor("speaker-slash", vik::ui::PhWeight::Regular)
                                   .size(13.0f)
                                   .color(t.muted ? vik::rgb(0xffffff)
                                                  : vik::rgb(0x8d94a3))))
                    .child(vik::div().id("s").px_2().rounded_sm().cursor_pointer()
                        .bg(t.soloed ? vik::rgb(0xc7a13c) : vik::rgb(0x2c313d))
                        .on_click(cx.listener([tid](Studio& s, const vik::ClickEvent&,
                                                    vik::Window&,
                                                    vik::Context<Studio>& c) {
                            if (auto* tr = s.session.find_track(tid)) tr->soloed = !tr->soloed;
                            s.publish();
                            c.notify();
                        }))
                        .child(vik::ui::phosphor("headphones", vik::ui::PhWeight::Regular)
                                   .size(13.0f)
                                   .color(t.soloed ? vik::rgb(0xffffff)
                                                   : vik::rgb(0x8d94a3))))));
    }

    // --- the arrangement: one canvas ---------------------------------------
    auto surface = vik::canvas([self](vik::Bounds b, vik::Window& w, vik::App&) {
        const int64_t head = self->playhead();
        // Timed *inside* the closure. paint_skia only records it -- timing
        // around the call measures how long it takes to store a lambda, which
        // is why the first version of this reported a flat 0.00 ms.
        w.paint_skia([self, b, head](SkCanvas* c) {
            const auto t0 = std::chrono::steady_clock::now();
            const SkRect rect = SkRect::MakeXYWH(b.origin.x, b.origin.y,
                                                 b.size.width, b.size.height);
            c->save();
            c->clipRect(rect);
            const auto stats = mx::draw_arrangement(c, rect, self->session, self->view,
                                                    head, self->selected);
            c->restore();
            const auto t1 = std::chrono::steady_clock::now();
            self->columns = stats.columns_drawn;
            self->timeline_px = b.size.width;
            self->build_ms = stats.build_ms;
            self->submit_ms = stats.submit_ms;
            self->last_paint_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            self->worst_paint_ms = std::max(self->worst_paint_ms, self->last_paint_ms);
            ++self->frames;
        });
    }).size_full();

    auto timeline =
        vik::div().flex_1().relative().overflow_hidden()
            .child(std::move(surface))
            .on_mouse_down(vik::MouseButton::Left, cx.listener(
                [](Studio& s, const vik::MouseDownEvent& e, vik::Window&,
                   vik::Context<Studio>& c) {
                    // The canvas fills this div, so the div's origin is the
                    // surface origin; event positions are already local.
                    const float x = e.position.x;
                    const float y = e.position.y;
                    if (y < mx::kRulerHeight) {
                        s.scrubbing = true;
                        s.seek(s.view.frame_at(x));
                    } else if (const mx::ClipId id = s.clip_at(x, y)) {
                        s.selected = id;
                        s.dragging = id;
                        if (const auto* clip = s.session.find_clip(id))
                            s.drag_grab = s.view.frame_at(x) - clip->start_frame;
                    } else {
                        s.selected = 0;
                    }
                    c.notify();
                }))
            .capture_mouse_move(cx.listener(
                [](Studio& s, const vik::MouseMoveEvent& e, vik::Window&,
                   vik::Context<Studio>& c) {
                    if (s.scrubbing) {
                        s.seek(s.view.frame_at(e.position.x));
                        c.notify();
                        return;
                    }
                    if (!s.dragging) return;
                    auto* clip = s.session.find_clip(s.dragging);
                    if (!clip) return;

                    int64_t start = s.view.frame_at(e.position.x) - s.drag_grab;
                    start = std::max<int64_t>(0, start);
                    if (start == clip->start_frame) return;
                    clip->start_frame = start;

                    // A new graph per drag frame. The mixer's generation-based
                    // retirement is what makes that affordable, and its
                    // crossfade on a moved clip is what makes it silent.
                    s.publish();
                    c.notify();
                }))
            .capture_mouse_up(vik::MouseButton::Left, cx.listener(
                [](Studio& s, const vik::MouseUpEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    if (s.dragging || s.scrubbing) {
                        s.dragging = 0;
                        s.scrubbing = false;
                        s.session.loop_end = s.session.length_frames();
                        c.notify();
                    }
                }))
            .on_scroll_wheel(cx.listener(
                [](Studio& s, const vik::ScrollWheelEvent& e, vik::Window&,
                   vik::Context<Studio>& c) {
                    if (e.modifiers.control || e.modifiers.alt) {
                        const double factor = e.delta.y > 0 ? 1.15 : 1.0 / 1.15;
                        s.view.zoom_about(e.position.x, factor, 8.0, 65536.0);
                    } else {
                        s.view.scroll_frames = std::max<int64_t>(
                            0, s.view.scroll_frames -
                                   static_cast<int64_t>(e.delta.x * s.view.frames_per_pixel));
                    }
                    c.notify();
                }));

    return vik::div().size_full().flex_col().bg(vik::rgb(0x14161d))
        .on_key_down(cx.listener(
            [](Studio& s, const vik::KeyDownEvent& e, vik::Window&,
               vik::Context<Studio>& c) {
                if (e.key == "space") s.toggle_play();
                else if (e.key == "home") s.seek(0);
                else if (e.key == "l") {
                    auto& t = g_mixer.transport;
                    t.looping.store(!t.looping.load());
                } else return;
                c.notify();
            }))
        .child(std::move(transport))
        .child(vik::div().flex_1().flex_row()
                   .child(std::move(headers))
                   .child(std::move(timeline)))
        .into_any();
}

}  // namespace

/// Draws the arrangement to a PNG with no window involved.
///
/// Capturing a real window turns out to be unreliable to automate -- focus does
/// not always follow the request, and a screenshot of the wrong window is worse
/// than no screenshot, because it still looks like evidence. Rendering offscreen
/// is deterministic, works over a terminal, and doubles as a visual regression
/// check that can be diffed.
int render_png(const fs::path& folder, const fs::path& out, int w, int h) {
    Studio studio;
    studio.session.rate = 48000;

    if (folder.empty()) {
        studio.build_demo();
    } else {
        // Synchronous: there is no UI here to keep responsive.
        std::vector<fs::path> files;
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(folder, ec)) {
            if (!e.is_regular_file()) continue;
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".mp3" || ext == ".flac") files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) studio.build_demo();

        static const uint32_t colors[] = {0xff4f8ef7, 0xffe0813c, 0xff56c08a,
                                          0xffc169d6, 0xffe05c72, 0xff40b8c4,
                                          0xffd9b23c, 0xff7f8cf0};
        int n = 0;
        for (const auto& f : files) {
            mx::Media media = mx::decode_file(f, studio.session.rate);
            if (!media.ok()) {
                std::printf("  skip: %s\n", media.error.c_str());
                continue;
            }
            auto& track = studio.session.add_track(f.stem().string(),
                                                   colors[n % std::size(colors)]);
            auto& src = studio.session.add_source(f, f.stem().string());
            src.buffer = media.buffer;
            src.peaks  = std::move(media.peaks);
            auto& clip = studio.session.add_clip(track.id, src.id, 0, src.buffer->frames);
            clip.name = f.stem().string();
            ++n;
        }
    }

    const float timeline_w = static_cast<float>(w) - 160.0f;
    studio.view.frames_per_pixel =
        mx::View::fit(studio.session.length_frames(), timeline_w);

    auto surface = SkSurfaces::Raster(SkImageInfo::MakeN32Premul(w, h));
    if (!surface) {
        std::printf("cannot create a raster surface\n");
        return 1;
    }
    SkCanvas* c = surface->getCanvas();
    c->clear(SkColorSetRGB(0x14, 0x16, 0x1d));

    const SkRect rect =
        SkRect::MakeXYWH(160.0f, 0.0f, timeline_w, static_cast<float>(h));
    const auto stats = mx::draw_arrangement(c, rect, studio.session, studio.view,
                                            studio.session.rate * 20, 0);

    SkPixmap pm;
    if (!surface->peekPixels(&pm)) {
        std::printf("cannot read pixels back\n");
        return 1;
    }
    SkFILEWStream stream(out.string().c_str());
    if (!SkPngEncoder::Encode(&stream, pm, {})) {
        std::printf("cannot encode png\n");
        return 1;
    }
    std::printf("rendered %s  tracks=%zu clips=%d columns=%lld "
                "build=%.2fms skia=%.2fms\n",
                out.string().c_str(), studio.session.tracks.size(),
                stats.clips_drawn, static_cast<long long>(stats.columns_drawn),
                stats.build_ms, stats.submit_ms);
    return 0;
}

int main(int argc, char** argv) {
    fs::path folder;
    fs::path render_to;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--selftest") selftest = true;
        else if (arg == "--render" && i + 1 < argc) render_to = argv[++i];
        else folder = arg;
    }

    if (!render_to.empty()) return render_png(folder, render_to, 1600, 900);

    if (!g_device.start(g_mixer, 48000))
        std::printf("audio: %s\n", g_device.error().c_str());

    vik::App::run([&](vik::App& app) {
        app.open_window(
            vik::WindowOptions{.title = "musicX Studio", .size = {1280.0f, 800.0f}},
            [&](vik::Window&, vik::App& app) {
                auto handle = app.add_entity<Studio>();
                app.update_entity(handle, [&](Studio& s, vik::Context<Studio>& c) {
                    s.selftest = selftest;
                    // The device decides the rate; the session follows it,
                    // because everything decoded is resampled to it once.
                    if (g_device.running()) s.session.rate = g_device.rate();
                    s.session.loop_start = 0;
                    if (folder.empty()) s.build_demo();
                    else s.load_folder(folder, app);
                    s.view.frames_per_pixel = 512.0;
                    s.publish();
                    c.notify();
                });

                // One tick drives everything that changes on its own: worker
                // results, and the playhead while the transport rolls. When
                // nothing is moving it does not notify, so the app goes back to
                // sleep rather than spinning at 60 Hz forever.
                app.set_interval(std::chrono::milliseconds(16), [handle](vik::App& a) {
                    a.update_entity(handle, [](Studio& s, vik::Context<Studio>& c) {
                        const bool landed = s.pump();
                        const bool rolling = g_mixer.transport.playing.load();
                        g_mixer.collect(g_device.running());
                        if (landed || rolling || s.meter > 0.001f) c.notify();
                    });
                    return true;
                });

                // The heartbeat runs in every mode. A GUI that can only be
                // diagnosed by looking at it is a GUI that cannot be diagnosed
                // over a terminal, which is where this gets debugged.
                app.set_interval(std::chrono::milliseconds(1000), [handle](vik::App& a) {
                    bool done = false;
                    a.update_entity(handle, [&](Studio& s, vik::Context<Studio>&) {
                        s.report();
                        done = s.selftest && ++s.reports >= 8;
                    });
                    if (done) a.quit();
                    return !done;
                });

                if (selftest) {
                    app.set_timer(std::chrono::milliseconds(1200), [handle](vik::App& a) {
                        a.update_entity(handle, [](Studio& s, vik::Context<Studio>& c) {
                            g_mixer.transport.playing.store(true);
                            c.notify();
                        });
                    });
                }
                return handle;
            });
    });

    g_device.stop();
    return 0;
}
