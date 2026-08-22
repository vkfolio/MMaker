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
#include <variant>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include "audio/device.h"
#include "audio/mixer.h"
#include "media/bounce.h"
#include "net/api.h"
#include "net/cache.h"
#include "app/crash.h"
#include "app/document.h"
#include "app/settings.h"
#include "media/decode.h"
#include "session.h"
#include "ui/arrangement.h"

#include "vikui/vikui.h"
#include "vikui/theme.h"
#include "vikui/elements/canvas.h"
#include "components/menu.h"
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
struct MsgDecoded {
    mx::SourceId source_id = 0;
    mx::Media    media;
    double       decode_ms = 0.0;
};

struct MsgProjects {
    std::vector<mx::net::ProjectSummary> list;
};

struct MsgProjectOpened {
    std::string id;
    std::string title;
    double      bpm = 120.0;
    int         bars = 32;
    int         stems = 0;
};

/// One stem, downloaded and decoded, ready for the UI to place.
///
/// It carries a label and an ordinal rather than ids: ids are handed out on the
/// UI thread, and a worker inventing one would be inventing an identity the
/// session has not agreed to.
struct MsgStemArrived {
    std::string label;
    std::string stem_id;
    std::string version_id;
    int         ordinal = 0;
    mx::Media   media;
    double      fetch_ms = 0.0;
    bool        from_cache = false;
};

/// A stem whose audio was replaced by a server-side render.
struct MsgStemUpdated {
    mx::TrackId  track_id = 0;
    mx::SourceId source_id = 0;
    std::string  version_id;
    std::string  op;                 // which tool produced it
    int64_t      affected_from = -1; // where to put the playhead, or -1
    mx::Media    media;
};

/// Progress for one server-side job. Named separately from MsgStatus because
/// a job has identity and a lifetime -- it can be cancelled, and it has to
/// survive several updates without being confused with a passing message.
struct MsgJob {
    std::string id;
    std::string kind;
    std::string status;
    std::string message;
    int         queue_position = 0;
    bool        finished = false;
};

/// What the connected engine can actually render.
struct MsgCapabilities {
    mx::net::Capabilities caps;
};

struct MsgStatus {
    std::string text;
    bool        error = false;
};

using UiMessage = std::variant<MsgDecoded, MsgProjects, MsgProjectOpened,
                               MsgStemArrived, MsgStemUpdated, MsgJob,
                               MsgCapabilities, MsgStatus>;

class Inbox {
public:
    void push(UiMessage m) {
        std::lock_guard lock(mutex_);
        queue_.push_back(std::move(m));
    }
    std::vector<UiMessage> drain() {
        std::lock_guard lock(mutex_);
        std::vector<UiMessage> out(std::make_move_iterator(queue_.begin()),
                                   std::make_move_iterator(queue_.end()));
        queue_.clear();
        return out;
    }

private:
    std::mutex             mutex_;
    std::deque<UiMessage>  queue_;
};

Inbox      g_inbox;
mx::Mixer  g_mixer;
mx::Device g_device;

std::atomic<int> g_pending{0};

/// Worker lifetime, because the framework has no hook for it.
///
/// App::run destroys the platform and calls SDL_Quit on its way out, and
/// nothing joins detached workers first -- so a render that finishes during
/// teardown calls wake() on a torn-down SDL and takes the process with it.
/// This was not theoretical: cancelling a job crashed on exit every time.
///
/// Two halves. `g_ui_alive` is cleared before quitting so no worker wakes a
/// platform that is going away, and `g_workers` is waited on after run()
/// returns so no thread is still touching the inbox during static destruction.
std::atomic<bool> g_ui_alive{true};
std::atomic<int>  g_workers{0};

/// Wake the UI, unless it is on its way out.
void wake_ui(vik::Platform* platform) {
    if (platform && g_ui_alive.load(std::memory_order_acquire)) platform->wake();
}

/// Scoped worker registration -- correct even when a worker returns early,
/// which several of them do on the error paths.
struct WorkerScope {
    WorkerScope() { g_workers.fetch_add(1, std::memory_order_acq_rel); }
    ~WorkerScope() { g_workers.fetch_sub(1, std::memory_order_acq_rel); }
};

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
    // The selection every AI tool acts on, and the drag that builds it.
    mx::Selection selection;
    bool          selecting = false;
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

    mx::Settings settings;
    // Keys are dispatched along the focused node's path. With nothing focused
    // the path is empty and on_key_down never fires, which is why every
    // shortcut silently did nothing.
    /// A handle to this entity, for callbacks that receive only (Window&, App&)
    /// -- menu items, chiefly. cx.listener() cannot wrap those because it wraps
    /// an event-carrying signature. This lives on the UI thread only; workers
    /// still never hold a handle.
    vik::Handle<Studio> self;

    vik::FocusHandle keys;
    bool             keys_focused = false;
    bool  fitted      = false;
    float timeline_px = 1100.0f;   // last measured canvas width
    // Where the canvas sits in the window, captured at paint.
    //
    // Mouse events arrive in window coordinates, while the view maps frames to
    // canvas-relative pixels. Using the raw event position put every click off
    // by the width of everything to the left of the timeline -- the sidebar
    // plus the track headers -- so the playhead landed nowhere near the pointer.
    float canvas_x = 0.0f;
    float canvas_y = 0.0f;

    /// Event position -> canvas position. Every mouse handler goes through
    /// these, so the mapping cannot drift between them again.
    float local_x(float window_x) const { return window_x - canvas_x; }
    float local_y(float window_y) const { return window_y - canvas_y; }

    // --- network ------------------------------------------------------------
    // The document is local and is the app's own. The pod is a render service
    // configured in settings, not the place work lives.
    std::filesystem::path document_path;
    bool                  dirty = false;
    bool                  show_settings = false;
    bool                  show_layers = false;
    bool                  show_generate = false;
    /// Which AI tool's modal is open. One field rather than a bool each: they
    /// are mutually exclusive, and separate bools drift out of sync.
    enum class Tool { None, Layer, Splitter, Inspire, Voice, Enhance, Extend };
    Tool                  tool = Tool::None;
    bool                  show_tool_picker = false;
    std::string           split_tier = "basic";
    bool                  split_remove_reverb = false;
    int                   extend_bars = 8;
    std::string           gen_prompt = "warm indie soul, brushed drums, rhodes";
    std::string           gen_lyrics;
    std::string           gen_quality = "ultra";
    int                   gen_bars = 16;
    bool                  gen_instrumental = false;
    std::string           doc_status;

    std::string pod_url;
    std::string pod_token;
    std::string net_status = "not connected";

    /// What the pod is doing for us right now.
    ///
    /// The plan asks for real queue position, elapsed time and cancel -- and
    /// specifically not a percentage, because the engine reports no progress
    /// signal and a bar would be inventing one. Elapsed seconds are true.
    struct ActiveJob {
        std::string id;
        std::string kind;
        std::string message;
        int         queue_position = 0;
        std::chrono::steady_clock::time_point started;
        bool        cancelling = false;
    };
    std::vector<ActiveJob> jobs;
    bool        net_busy = false;
    bool        net_error = false;
    std::vector<mx::net::ProjectSummary> projects;
    mx::net::Capabilities caps;
    std::string open_project_id;
    int         stems_expected = 0;
    int         stems_arrived = 0;
    int         cache_hits = 0;

    bool selftest = false;
    int  reports  = 0;
    int  stress_tick = 0;
    // Drives the select-and-regenerate loop without a mouse, so the milestone
    // is checkable from a terminal rather than only by hand.
    double regen_from = -1.0, regen_to = -1.0;
    bool   regen_fired = false;
    bool   regen_done = false;
    double cancel_after = -1.0;   // scripted cancel, for testing the path
    std::string scripted_layer;   // scripted add-a-layer, likewise
    std::string scripted_prompt;  // scripted generate, likewise
    bool        stress = false;   // cycle panels hard, to shake out frame limits
    bool        theme_ok = false;

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
            WorkerScope scope;
            const auto t0 = std::chrono::steady_clock::now();
            mx::Media media = mx::decode_file(file, rate);
            const auto t1 = std::chrono::steady_clock::now();
            g_inbox.push(MsgDecoded{
                id, std::move(media),
                std::chrono::duration<double, std::milli>(t1 - t0).count()});
            // Without this the UI can sit in SDL_WaitEventTimeout until the
            // user happens to move the mouse.
            wake_ui(platform);
        }).detach();
    }

    /// Switch where renders happen. Nothing else changes: the client speaks to
    /// a URL, so a mode is which URL, not a second implementation.
    void set_mode(mx::Mode mode, vik::App& app) {
        settings.mode = mode;
        settings.save();
        pod_url = settings.active_url();
        pod_token = settings.active_token();
        projects.clear();
        net_error = false;
        net_status = pod_url.empty() ? "no address set" : "not connected";
        if (!pod_url.empty()) connect(app);
    }

    void mark_dirty() { dirty = true; }

    void new_document() {
        session = mx::Session{};
        session.rate = g_device.running() ? g_device.rate() : 48000;
        document_path.clear();
        open_project_id.clear();
        loaded_sources = stems_arrived = stems_expected = 0;
        fitted = false;
        dirty = false;
        doc_status = "new project";
        selection.clear();
        publish();
    }

    /// Save, defaulting to Documents/MusicMaker when the project has no path.
    void save_document_now() {
        if (document_path.empty()) {
            std::string name = session.title.empty() ? "Untitled" : session.title;
            for (char& c : name)
                if (std::strchr("\/:*?\"<>|", c)) c = '_';
            document_path = mx::documents_root() / (name + ".mmproj");
        }
        std::string error;
        if (mx::save_document(session, document_path, &error)) {
            dirty = false;
            doc_status = "saved " + document_path.filename().string();
            settings.last_document = document_path.string();
            settings.save();
        } else {
            doc_status = "save failed: " + error;
        }
        std::printf("%s\n", doc_status.c_str());
        std::fflush(stdout);
    }

    /// Open a .mmproj and start decoding whatever audio it names.
    void open_document(const std::filesystem::path& path, vik::App& app) {
        const auto result = mx::load_document(session, path);
        if (!result.ok) {
            doc_status = result.error;
            return;
        }
        document_path = path;
        dirty = false;
        fitted = false;
        loaded_sources = 0;
        // Provenance travels with the document, so a project opened here knows
        // which pod project it came from without the server being consulted.
        open_project_id.clear();
        for (const auto& src : session.sources)
            if (!src.project_id.empty()) { open_project_id = src.project_id; break; }

        doc_status = result.missing_audio
            ? std::to_string(result.missing_audio) + " missing from cache"
            : "opened " + path.filename().string();

        // Decode on workers, one per source, as the network path does.
        for (auto& src : session.sources) {
            if (src.local_path.empty() ||
                !std::filesystem::exists(src.local_path)) continue;
            spawn_decode(src.id, src.local_path, app);
        }
        settings.last_document = path.string();
        settings.save();
        publish();
    }

    std::vector<std::filesystem::path> local_documents() const {
        std::vector<std::filesystem::path> found;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(
                 mx::documents_root(), ec))
            if (e.is_regular_file() && e.path().extension() == ".mmproj")
                found.push_back(e.path());
        std::sort(found.begin(), found.end());
        return found;
    }

    /// Connect and list what is on the pod. Runs on a worker.
    void connect(vik::App& app) {
        if (net_busy || pod_url.empty()) return;
        net_busy = true;
        net_status = "connecting…";
        net_error = false;
        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);
            const auto health = api.health();
            if (!health.reachable) {
                g_inbox.push(MsgStatus{"cannot reach the pod: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            // Reachability, not an inventory. Listing the pod's projects would
            // be asking a generation service what work exists, which is the
            // question the local document answers.
            std::string note = health.ready ? "ready" : health.status;
            if (!health.gpu.empty()) note += " · " + health.gpu;
            g_inbox.push(MsgStatus{note, false});

            // Ask what it can render rather than assuming. A tier whose weights
            // are absent renders as something else without saying so, and that
            // is now routine: a laptop carries the small tier, a pod may carry
            // all three.
            g_inbox.push(MsgCapabilities{api.capabilities()});
            wake_ui(platform);
        }).detach();
    }

    /// Open a project: fetch its stems, decode them, stream them in.
    ///
    /// Each stem is reported the moment it is ready rather than waiting for the
    /// set, so a slow one does not hold up the rest -- and crucially the whole
    /// unit of work (download, decode, build the pyramid) finishes on the
    /// worker before anything is reported done. Reporting earlier would move
    /// the decode onto the UI thread, which is where dropped frames come from.
    void open_project(const std::string& id, vik::App& app) {
        if (net_busy) return;
        net_busy = true;
        net_error = false;
        net_status = "opening…";
        stems_expected = stems_arrived = cache_hits = 0;
        auto* platform = &app.platform();
        const uint32_t rate = session.rate;

        std::thread([url = pod_url, token = pod_token, id, rate, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            auto detail = api.project(id);
            if (!detail) {
                g_inbox.push(MsgStatus{"cannot open project: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgProjectOpened{detail->summary.id, detail->summary.title,
                                          detail->summary.bpm, detail->summary.bars,
                                          static_cast<int>(detail->stems.size())});
            wake_ui(platform);

            int ordinal = 0;
            for (const auto& stem : detail->stems) {
                const auto* version = stem.version();
                if (!version || version->audio.empty()) continue;

                const auto t0 = std::chrono::steady_clock::now();
                const uint64_t key = mx::net::cache_key(detail->summary.id,
                                                        version->id, version->audio);
                const auto cached = mx::net::cache_path(key);
                bool hit = std::filesystem::exists(cached);

                if (!hit &&
                    !api.fetch_audio(detail->summary.id, version->audio,
                                     cached.string())) {
                    g_inbox.push(MsgStatus{stem.track_class + ": " + api.last_error(), true});
                    wake_ui(platform);
                    continue;
                }

                mx::Media media = mx::decode_file(cached, rate);
                if (!media.ok()) {
                    // A cached file that will not decode is a poisoned cache
                    // entry, not a permanent failure. Drop it so the next
                    // attempt refetches instead of failing forever.
                    std::error_code ec;
                    std::filesystem::remove(cached, ec);
                    g_inbox.push(MsgStatus{stem.track_class + ": " + media.error, true});
                    wake_ui(platform);
                    continue;
                }

                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                g_inbox.push(MsgStemArrived{
                    stem.label.empty() ? stem.track_class : stem.label,
                    stem.id, version->id, ordinal++, std::move(media), ms, hit});
                wake_ui(platform);
            }
            g_inbox.push(MsgStatus{"", false});
            wake_ui(platform);
        }).detach();
    }

    /// Regenerate the selected range of the selected track, on the pod.
    ///
    /// This is the loop the whole product rests on: drag a range, hit a tool,
    /// hear the result. Everything it needs already exists -- the server takes
    /// second-precision ranges, versions are append-only, and the client can
    /// follow a job with cancel -- so this is the wiring, not new machinery.
    void regenerate(vik::App& app) {
        if (net_busy || !selection.active() || open_project_id.empty()) return;

        // Which stem is this? The session keeps provenance for exactly this
        // question; without a stem id there is nothing to ask the server about.
        const mx::Clip* clip = nullptr;
        for (const auto& candidate : session.clips)
            if (candidate.track_id == selection.track) { clip = &candidate; break; }
        if (!clip) return;
        mx::Source* source = session.find_source(clip->source_id);
        if (!source || source->stem_id.empty()) {
            net_status = "that track did not come from the pod";
            net_error = true;
            return;
        }

        const double rate = static_cast<double>(session.rate);
        const double start_s = static_cast<double>(selection.begin()) / rate;
        const double end_s = static_cast<double>(selection.end()) / rate;

        net_busy = true;
        net_error = false;
        net_status = "regenerating…";

        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, project = open_project_id,
                     stem = source->stem_id, source_id = source->id,
                     track_id = selection.track, start_s, end_s,
                     rate = session.rate, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            // A key derived from the request itself: if the response is lost
            // and this is retried, the server recognises it as the same work
            // rather than starting a second GPU job.
            const std::string key = project + ":" + stem + ":" +
                                    std::to_string(static_cast<int64_t>(start_s * 1000)) +
                                    ":" + std::to_string(static_cast<int64_t>(end_s * 1000));

            auto job = api.repaint(project, stem, start_s, end_s, {}, key);
            if (!job) {
                g_inbox.push(MsgStatus{"repaint refused: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }

            g_inbox.push(MsgJob{job->id, "regenerate", job->status, job->message,
                                job->queue_position, false});
            wake_ui(platform);

            const auto finished = api.follow(job->id, [&](const mx::net::JobRef& j) {
                g_inbox.push(MsgJob{j.id, "regenerate", j.status, j.message,
                                    j.queue_position, false});
                wake_ui(platform);
                return true;
            });
            g_inbox.push(MsgJob{finished.id, "regenerate", finished.status,
                                finished.message, 0, true});
            wake_ui(platform);

            if (finished.status == "cancelled") {
                // Terminal, like success: a scripted run must not sit waiting
                // for audio that is never coming.
                g_inbox.push(MsgStatus{"cancelled", false});
                wake_ui(platform);
                return;
            }
            if (finished.status != "done") {
                g_inbox.push(MsgStatus{
                    "repaint " + finished.status +
                        (finished.error.empty() ? "" : ": " + finished.error), true});
                wake_ui(platform);
                return;
            }

            // The result is a NEW version, not a rewritten file -- so re-read
            // the project and take whatever is current now. Guessing the path
            // would assume the server mutates audio in place, which is exactly
            // the thing it was fixed not to do.
            auto detail = api.project(project);
            if (!detail) {
                g_inbox.push(MsgStatus{"rendered, but cannot re-read the project: " +
                                       api.last_error(), true});
                wake_ui(platform);
                return;
            }
            const mx::net::VersionRef* version = nullptr;
            for (const auto& st : detail->stems)
                if (st.id == stem) version = st.version();
            if (!version || version->audio.empty()) {
                g_inbox.push(MsgStatus{"rendered, but the stem has no current version", true});
                wake_ui(platform);
                return;
            }

            const uint64_t ckey = mx::net::cache_key(project, version->id, version->audio);
            const auto cached = mx::net::cache_path(ckey);
            if (!std::filesystem::exists(cached) &&
                !api.fetch_audio(project, version->audio, cached.string())) {
                g_inbox.push(MsgStatus{"cannot fetch the new take: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            mx::Media media = mx::decode_file(cached, rate);
            if (!media.ok()) {
                std::error_code ec;
                std::filesystem::remove(cached, ec);
                g_inbox.push(MsgStatus{"new take will not decode: " + media.error, true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgStemUpdated{
                track_id, source_id, version->id, "repaint",
                static_cast<int64_t>(start_s * rate), std::move(media)});
            wake_ui(platform);
        }).detach();
    }

    /// The set of instruments the server will actually condition on.
    ///
    /// Fixed, not free text: the model takes a track class, so anything else is
    /// a rejected request rather than a creative prompt. Chips, as in the
    /// reference, because a list you pick from cannot be typed wrong.
    static constexpr const char* kTrackClasses[] = {
        "drums", "bass", "guitar", "keyboard", "strings", "synth",
        "brass", "woodwinds", "percussion", "fx", "backing_vocals", "vocals",
    };

    /// Ad-hoc generation: describe something, get stems on the timeline.
    ///
    /// The pod is a generation service here, not a place work lives. It needs a
    /// workspace to hang audio and stem ids off, so one is created per request
    /// and kept only as provenance -- the local document is the project, and
    /// nothing in the UI ever offers the pod's projects as things to open.
    void generate(const std::string& prompt, vik::App& app) {
        if (net_busy || pod_url.empty() || prompt.empty()) return;

        net_busy = true;
        net_error = false;
        net_status = "generating…";
        show_generate = false;

        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, prompt,
                     lyrics = gen_lyrics, bars = gen_bars, quality = gen_quality,
                     instrumental = gen_instrumental, bpm = session.bpm,
                     rate = session.rate, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            auto made = api.generate(prompt, lyrics, bars, quality, instrumental, bpm);
            if (!made.ok) {
                g_inbox.push(MsgStatus{"generate refused: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgJob{made.job.id, "generate", made.job.status,
                                made.job.message, made.job.queue_position, false});
            wake_ui(platform);

            auto finished = api.follow(made.job.id, [&](const mx::net::JobRef& j) {
                g_inbox.push(MsgJob{j.id, "generate", j.status, j.message,
                                    j.queue_position, false});
                wake_ui(platform);
                return true;
            });
            g_inbox.push(MsgJob{finished.id, "generate", finished.status,
                                finished.message, 0, true});
            wake_ui(platform);
            if (finished.status != "done") {
                g_inbox.push(MsgStatus{"generate " + finished.status +
                    (finished.error.empty() ? "" : ": " + finished.error),
                    finished.status != "cancelled"});
                wake_ui(platform);
                return;
            }

            auto detail = api.project(made.project_id);
            if (!detail || detail->variations.empty()) {
                g_inbox.push(MsgStatus{"rendered, but no take came back", true});
                wake_ui(platform);
                return;
            }

            // No split. Generate produces a take; separating it into parts is
            // the Stem Splitter tool, invoked deliberately. Doing it here
            // doubled time-to-first-sound and hid the take that was rendered.

            g_inbox.push(MsgProjectOpened{made.project_id, "Untitled",
                                          detail ? detail->summary.bpm : 120.0,
                                          detail ? detail->summary.bars : 32,
                                          detail ? static_cast<int>(detail->stems.size())
                                                 : 0});
            wake_ui(platform);

            // Stems if the split produced any, otherwise the take itself, so a
            // failed split still leaves something to listen to.
            int ordinal = 0;
            if (detail && !detail->stems.empty()) {
                for (const auto& stem : detail->stems) {
                    const auto* version = stem.version();
                    if (!version || version->audio.empty()) continue;
                    const uint64_t key = mx::net::cache_key(made.project_id,
                                                            version->id, version->audio);
                    const auto cached = mx::net::cache_path(key);
                    const bool hit = std::filesystem::exists(cached);
                    if (!hit && !api.fetch_audio(made.project_id, version->audio,
                                                 cached.string()))
                        continue;
                    mx::Media media = mx::decode_file(cached, rate);
                    if (!media.ok()) continue;
                    g_inbox.push(MsgStemArrived{
                        stem.label.empty() ? stem.track_class : stem.label,
                        stem.id, version->id, ordinal++, std::move(media), 0.0, hit});
                    wake_ui(platform);
                }
            } else if (detail) {
                const auto& take = detail->variations.front();
                const uint64_t key = mx::net::cache_key(made.project_id, take.id,
                                                        take.audio);
                const auto cached = mx::net::cache_path(key);
                if (std::filesystem::exists(cached) ||
                    api.fetch_audio(made.project_id, take.audio, cached.string())) {
                    mx::Media media = mx::decode_file(cached, rate);
                    if (media.ok())
                        g_inbox.push(MsgStemArrived{"take", "", take.id, 0,
                                                    std::move(media), 0.0, false});
                }
            }
            g_inbox.push(MsgStatus{"", false});
            wake_ui(platform);
        }).detach();
    }

    /// Generate a new part against what the project already has.
    ///
    /// The selection is optional here, unlike regenerate: with a range the
    /// layer covers that span, without one it covers the arrangement. Both are
    /// useful, so neither is forced.
    void add_layer(const std::string& track_class, vik::App& app) {
        if (net_busy || open_project_id.empty()) return;

        double start_s = -1.0, end_s = -1.0;
        if (selection.active()) {
            start_s = static_cast<double>(selection.begin()) / session.rate;
            end_s = static_cast<double>(selection.end()) / session.rate;
        }

        // Which stems exist already, so the new one can be told apart. The
        // server returns a job, not a stem id, and a layer is simply the stem
        // that was not there before.
        std::vector<std::string> known;
        for (const auto& src : session.sources)
            if (!src.stem_id.empty()) known.push_back(src.stem_id);

        net_busy = true;
        net_error = false;
        net_status = "adding " + track_class + "…";
        show_layers = false;

        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, project = open_project_id,
                     track_class, start_s, end_s, known = std::move(known),
                     ordinal = static_cast<int>(session.tracks.size()),
                     rate = session.rate, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            const std::string key = project + ":layer:" + track_class + ":" +
                                    std::to_string(static_cast<int64_t>(start_s * 1000));
            auto job = api.add_layer(project, track_class, {}, start_s, end_s, key);
            if (!job) {
                g_inbox.push(MsgStatus{"layer refused: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgJob{job->id, "layer", job->status, job->message,
                                job->queue_position, false});
            wake_ui(platform);

            const auto finished = api.follow(job->id, [&](const mx::net::JobRef& j) {
                g_inbox.push(MsgJob{j.id, "layer", j.status, j.message,
                                    j.queue_position, false});
                wake_ui(platform);
                return true;
            });
            g_inbox.push(MsgJob{finished.id, "layer", finished.status,
                                finished.message, 0, true});
            wake_ui(platform);

            if (finished.status != "done") {
                g_inbox.push(MsgStatus{
                    "layer " + finished.status +
                        (finished.error.empty() ? "" : ": " + finished.error),
                    finished.status != "cancelled"});
                wake_ui(platform);
                return;
            }

            auto detail = api.project(project);
            if (!detail) {
                g_inbox.push(MsgStatus{"layered, but cannot re-read the project: " +
                                       api.last_error(), true});
                wake_ui(platform);
                return;
            }

            const mx::net::StemRef* fresh = nullptr;
            for (const auto& stem : detail->stems)
                if (std::find(known.begin(), known.end(), stem.id) == known.end())
                    fresh = &stem;
            if (!fresh || !fresh->version()) {
                g_inbox.push(MsgStatus{"layered, but no new stem appeared", true});
                wake_ui(platform);
                return;
            }

            const auto* version = fresh->version();
            const uint64_t ckey = mx::net::cache_key(project, version->id, version->audio);
            const auto cached = mx::net::cache_path(ckey);
            if (!std::filesystem::exists(cached) &&
                !api.fetch_audio(project, version->audio, cached.string())) {
                g_inbox.push(MsgStatus{"cannot fetch the new layer: " + api.last_error(),
                                       true});
                wake_ui(platform);
                return;
            }
            mx::Media media = mx::decode_file(cached, rate);
            if (!media.ok()) {
                std::error_code ec;
                std::filesystem::remove(cached, ec);
                g_inbox.push(MsgStatus{"new layer will not decode: " + media.error, true});
                wake_ui(platform);
                return;
            }

            g_inbox.push(MsgStemArrived{
                fresh->label.empty() ? fresh->track_class : fresh->label,
                fresh->id, version->id, ordinal, std::move(media), 0.0, false});
            g_inbox.push(MsgStatus{"added " + track_class, false});
            wake_ui(platform);
        }).detach();
    }

    /// The stem this selection refers to, or nullptr with a reason set.
    ///
    /// Every per-stem tool needs the same three things, and each one deriving
    /// them separately is how they drift apart.
    mx::Source* selected_source(std::string* why = nullptr) {
        if (open_project_id.empty()) {
            if (why) *why = "nothing here came from the pod";
            return nullptr;
        }
        const mx::TrackId track =
            selection.active() ? selection.track
                               : (session.tracks.empty() ? 0 : session.tracks.front().id);
        for (const auto& clip : session.clips) {
            if (clip.track_id != track) continue;
            mx::Source* src = session.find_source(clip.source_id);
            if (src && !src->stem_id.empty()) return src;
        }
        if (why) *why = "that track did not come from the pod";
        return nullptr;
    }

    /// Shared tail for every per-stem tool: follow the job, re-read the project,
    /// fetch whatever version is current now, decode it, hand it back.
    ///
    /// Written once because the five tools differ only in which request starts
    /// them -- and because each one re-deriving "which audio is the result"
    /// would be five chances to assume the server names files predictably.
    void run_tool(const char* op, mx::SourceId source_id, std::string stem_id,
                  int64_t affected_from, vik::App& app,
                  std::function<std::optional<mx::net::JobRef>(
                      mx::net::ApiClient&, const std::string&, const std::string&)>
                      start) {
        if (net_busy || open_project_id.empty()) return;
        net_busy = true;
        net_error = false;
        net_status = std::string(op) + "…";

        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, project = open_project_id,
                     op = std::string(op), stem_id, source_id, affected_from,
                     rate = session.rate, start = std::move(start), platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            auto job = start(api, project, stem_id);
            if (!job) {
                g_inbox.push(MsgStatus{op + " refused: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgJob{job->id, op, job->status, job->message,
                                job->queue_position, false});
            wake_ui(platform);

            const auto finished = api.follow(job->id, [&](const mx::net::JobRef& j) {
                g_inbox.push(MsgJob{j.id, op, j.status, j.message,
                                    j.queue_position, false});
                wake_ui(platform);
                return true;
            });
            g_inbox.push(MsgJob{finished.id, op, finished.status, finished.message,
                                0, true});
            wake_ui(platform);

            if (finished.status != "done") {
                g_inbox.push(MsgStatus{
                    op + " " + finished.status +
                        (finished.error.empty() ? "" : ": " + finished.error),
                    finished.status != "cancelled"});
                wake_ui(platform);
                return;
            }

            auto detail = api.project(project);
            if (!detail) {
                g_inbox.push(MsgStatus{"rendered, but cannot re-read the project: " +
                                       api.last_error(), true});
                wake_ui(platform);
                return;
            }

            // A split makes new stems rather than a new version of this one, so
            // reload the whole set. Anything else replaces one stem in place.
            if (op == "split") {
                g_inbox.push(MsgProjectOpened{project, "", detail->summary.bpm,
                                              detail->summary.bars,
                                              static_cast<int>(detail->stems.size())});
                wake_ui(platform);
                int ordinal = 0;
                for (const auto& stem : detail->stems) {
                    const auto* version = stem.version();
                    if (!version || version->audio.empty()) continue;
                    const uint64_t key = mx::net::cache_key(project, version->id,
                                                            version->audio);
                    const auto cached = mx::net::cache_path(key);
                    const bool hit = std::filesystem::exists(cached);
                    if (!hit && !api.fetch_audio(project, version->audio,
                                                 cached.string()))
                        continue;
                    mx::Media media = mx::decode_file(cached, rate);
                    if (!media.ok()) continue;
                    g_inbox.push(MsgStemArrived{
                        stem.label.empty() ? stem.track_class : stem.label,
                        stem.id, version->id, ordinal++, std::move(media), 0.0, hit});
                    wake_ui(platform);
                }
                g_inbox.push(MsgStatus{"", false});
                wake_ui(platform);
                return;
            }

            const mx::net::VersionRef* version = nullptr;
            for (const auto& stem : detail->stems)
                if (stem.id == stem_id) version = stem.version();
            if (!version || version->audio.empty()) {
                g_inbox.push(MsgStatus{"rendered, but the stem has no current version",
                                       true});
                wake_ui(platform);
                return;
            }

            const uint64_t key = mx::net::cache_key(project, version->id, version->audio);
            const auto cached = mx::net::cache_path(key);
            if (!std::filesystem::exists(cached) &&
                !api.fetch_audio(project, version->audio, cached.string())) {
                g_inbox.push(MsgStatus{"cannot fetch the result: " + api.last_error(),
                                       true});
                wake_ui(platform);
                return;
            }
            mx::Media media = mx::decode_file(cached, rate);
            if (!media.ok()) {
                std::error_code ec;
                std::filesystem::remove(cached, ec);
                g_inbox.push(MsgStatus{"the result will not decode: " + media.error,
                                       true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgStemUpdated{0, source_id, version->id, op,
                                        affected_from, std::move(media)});
            wake_ui(platform);
        }).detach();
    }

    void run_split(vik::App& app) {
        if (net_busy || open_project_id.empty()) return;
        // Split acts on the take, not on a stem, so it needs a variation id --
        // which means asking the pod what this project actually has.
        auto* platform = &app.platform();
        net_busy = true;
        net_error = false;
        net_status = "splitting…";
        std::thread([url = pod_url, token = pod_token, project = open_project_id,
                     tier = split_tier, dereverb = split_remove_reverb,
                     rate = session.rate, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            auto detail = api.project(project);
            if (!detail || detail->variations.empty()) {
                g_inbox.push(MsgStatus{"nothing to split -- generate a take first",
                                       true});
                wake_ui(platform);
                return;
            }
            auto job = api.split(project, detail->variations.front().id, tier,
                                 dereverb);
            if (!job) {
                g_inbox.push(MsgStatus{"split refused: " + api.last_error(), true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgJob{job->id, "split", job->status, job->message,
                                job->queue_position, false});
            wake_ui(platform);
            const auto finished = api.follow(job->id, [&](const mx::net::JobRef& j) {
                g_inbox.push(MsgJob{j.id, "split", j.status, j.message,
                                    j.queue_position, false});
                wake_ui(platform);
                return true;
            });
            g_inbox.push(MsgJob{finished.id, "split", finished.status,
                                finished.message, 0, true});
            wake_ui(platform);
            if (finished.status != "done") {
                g_inbox.push(MsgStatus{"split " + finished.status,
                                       finished.status != "cancelled"});
                wake_ui(platform);
                return;
            }

            detail = api.project(project);
            if (!detail) {
                g_inbox.push(MsgStatus{"split, but cannot re-read the project", true});
                wake_ui(platform);
                return;
            }
            g_inbox.push(MsgProjectOpened{project, "", detail->summary.bpm,
                                          detail->summary.bars,
                                          static_cast<int>(detail->stems.size())});
            wake_ui(platform);
            int ordinal = 0;
            for (const auto& stem : detail->stems) {
                const auto* version = stem.version();
                if (!version || version->audio.empty()) continue;
                const uint64_t key = mx::net::cache_key(project, version->id,
                                                        version->audio);
                const auto cached = mx::net::cache_path(key);
                const bool hit = std::filesystem::exists(cached);
                if (!hit && !api.fetch_audio(project, version->audio, cached.string()))
                    continue;
                mx::Media media = mx::decode_file(cached, rate);
                if (!media.ok()) continue;
                g_inbox.push(MsgStemArrived{
                    stem.label.empty() ? stem.track_class : stem.label,
                    stem.id, version->id, ordinal++, std::move(media), 0.0, hit});
                wake_ui(platform);
            }
            g_inbox.push(MsgStatus{"", false});
            wake_ui(platform);
        }).detach();
    }

    void run_isolate(vik::App& app) {
        std::string why;
        mx::Source* src = selected_source(&why);
        if (!src) { net_status = why; net_error = true; return; }
        run_tool("enhance", src->id, src->stem_id, selection.begin(), app,
                 [](mx::net::ApiClient& api, const std::string& project,
                    const std::string& stem) {
                     return api.isolate(project, stem, project + ":isolate:" + stem);
                 });
    }

    void run_extend(vik::App& app) {
        std::string why;
        mx::Source* src = selected_source(&why);
        if (!src) { net_status = why; net_error = true; return; }
        run_tool("extend", src->id, src->stem_id, -1, app,
                 [bars = extend_bars](mx::net::ApiClient& api,
                                      const std::string& project,
                                      const std::string& stem) {
                     return api.extend(project, stem, bars,
                                       project + ":extend:" + stem + ":" +
                                           std::to_string(bars));
                 });
    }

    void cancel_job(const std::string& id, vik::App& app) {
        for (auto& job : jobs)
            if (job.id == id) job.cancelling = true;
        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, id, platform] {
            WorkerScope scope;
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);
            // Best effort. The follow loop is what actually notices the job
            // ending, so nothing here needs to report success -- and a cancel
            // that races a job finishing is not an error worth surfacing.
            api.cancel(id);
            wake_ui(platform);
        }).detach();
    }

    /// Mix the session to a file next to the cache, and say where it went.
    ///
    /// No file dialog: vikui has no native picker, and writing one is its own
    /// job. A known location plus the path on screen beats a half-built dialog.
    void export_mix() {
        if (session.clips.empty()) {
            net_status = "nothing to export";
            net_error = true;
            return;
        }
        auto dir = mx::net::cache_root().parent_path() / "exports";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        std::string name = session.title.empty() ? "mix" : session.title;
        for (char& c : name)
            if (std::strchr("\/:*?\"<>|", c)) c = '_';
        const auto out = dir / (name + ".wav");

        const auto result = mx::bounce(session, out);
        net_error = !result.ok;
        net_status = result.ok
            ? ("exported " + out.filename().string() + "  " +
               std::to_string(static_cast<int>(result.seconds)) + "s")
            : ("export failed: " + result.error);
        if (result.ok)
            std::printf("exported %s (%.1fs, %.0fx realtime)\n",
                        out.string().c_str(), result.seconds, result.realtime_ratio);
    }

    /// Places a stem that a worker finished. Ids are minted here, on the UI
    /// thread, which is the only place they may be.
    void place_stem(MsgStemArrived& msg) {
        static const uint32_t colors[] = {0xff4f8ef7, 0xffe0813c, 0xff56c08a,
                                          0xffc169d6, 0xffe05c72, 0xff40b8c4,
                                          0xffd9b23c, 0xff7f8cf0};
        auto& track = session.add_track(msg.label,
                                        colors[msg.ordinal % std::size(colors)]);
        auto& source = session.add_source({}, msg.label);
        source.source_rate = msg.media.source_rate;
        source.adopt(mx::SourceVersion{msg.version_id, {}, "import", {},
                                       msg.media.buffer,
                                       std::move(msg.media.peaks)});
        // Provenance, not a mirror: enough to fetch this audio again and to say
        // where it came from. No copy of the server's project.
        source.project_id = open_project_id;
        source.stem_id = msg.stem_id;
        source.version_id = msg.version_id;

        auto& clip = session.add_clip(track.id, source.id, 0, source.buffer->frames);
        clip.name = msg.label;
        clip.fade_in = clip.fade_out = session.rate / 200;

        ++stems_arrived;
        if (msg.from_cache) ++cache_hits;
        ++loaded_sources;
    }

    /// Everything a worker produced is applied here, on the UI thread.
    bool pump() {
        bool changed = false;
        for (auto& message : g_inbox.drain()) {
            std::visit([&](auto& m) {
                using T = std::decay_t<decltype(m)>;

                if constexpr (std::is_same_v<T, MsgDecoded>) {
                    --g_pending;
                    last_decode_ms = m.decode_ms;
                    mx::Source* src = session.find_source(m.source_id);
                    if (!src) return;
                    if (!m.media.ok()) {
                        std::printf("decode failed: %s\n", m.media.error.c_str());
                        return;
                    }
                    src->buffer = m.media.buffer;
                    src->peaks = std::move(m.media.peaks);
                    src->source_rate = m.media.source_rate;

                    mx::TrackId track = 0;
                    for (size_t i = 0; i < session.sources.size(); ++i)
                        if (session.sources[i].id == m.source_id &&
                            i < session.tracks.size())
                            track = session.tracks[i].id;
                    if (track) {
                        auto& clip = session.add_clip(track, src->id, 0,
                                                      src->buffer->frames);
                        clip.name = src->label;
                        clip.fade_in = clip.fade_out = session.rate / 200;
                    }
                    ++loaded_sources;
                    changed = true;

                } else if constexpr (std::is_same_v<T, MsgProjects>) {
                    projects = std::move(m.list);
                    net_busy = false;
                    net_error = false;
                    net_status = std::to_string(projects.size()) + " projects";

                } else if constexpr (std::is_same_v<T, MsgProjectOpened>) {
                    // A fresh project replaces the arrangement rather than
                    // adding to it: two projects share no timeline, and merging
                    // them silently would be a worse surprise than clearing.
                    session.tracks.clear();
                    session.clips.clear();
                    session.sources.clear();
                    loaded_sources = 0;
                    fitted = false;
                    open_project_id = m.id;
                    session.title = m.title;
                    session.bpm = m.bpm;
                    stems_expected = m.stems;
                    net_status = m.stems > 0
                        ? ("loading " + std::to_string(m.stems) + " stems…")
                        : "no stems -- split a take on the pod first";
                    changed = true;

                } else if constexpr (std::is_same_v<T, MsgStemArrived>) {
                    place_stem(m);
                    last_decode_ms = m.fetch_ms;
                    if (selftest && regen_fired &&
                        (!scripted_layer.empty() ||
                         (!scripted_prompt.empty() && stems_arrived >= stems_expected &&
                          stems_expected > 0)))
                        regen_done = true;
                    net_status = "loaded " + std::to_string(stems_arrived) + "/" +
                                 std::to_string(std::max(stems_expected, stems_arrived));
                    changed = true;

                } else if constexpr (std::is_same_v<T, MsgStemUpdated>) {
                    // Append, never replace. A result you cannot compare against
                    // the take before it is indistinguishable from no result at
                    // all unless you happen to replay the exact region -- which
                    // is precisely how a working regenerate got reported as
                    // "nothing changed".
                    if (mx::Source* src = session.find_source(m.source_id)) {
                        src->adopt(mx::SourceVersion{m.version_id, {}, m.op, {},
                                                     m.media.buffer,
                                                     std::move(m.media.peaks)});
                        for (auto& clip : session.clips)
                            if (clip.source_id == src->id)
                                clip.length = std::min(clip.length, src->buffer->frames);

                        // Put the playhead at the start of what changed, so the
                        // next thing heard is the thing that was rendered.
                        if (m.affected_from >= 0) seek(m.affected_from);
                    }
                    net_busy = false;
                    net_status = m.op.empty() ? "done" : (m.op + " applied");
                    if (selftest) regen_done = true;
                    changed = true;

                } else if constexpr (std::is_same_v<T, MsgJob>) {
                    auto it = std::find_if(jobs.begin(), jobs.end(),
                                           [&](const ActiveJob& j) { return j.id == m.id; });
                    if (m.finished) {
                        if (it != jobs.end()) jobs.erase(it);
                    } else if (it != jobs.end()) {
                        it->message = m.message.empty() ? m.status : m.message;
                        it->queue_position = m.queue_position;
                    } else {
                        jobs.push_back(ActiveJob{
                            m.id, m.kind,
                            m.message.empty() ? m.status : m.message,
                            m.queue_position, std::chrono::steady_clock::now(), false});
                    }

                } else if constexpr (std::is_same_v<T, MsgCapabilities>) {
                    caps = std::move(m.caps);
                    // Never leave the picker on a tier this engine will not
                    // honour -- offering it is the whole failure being avoided.
                    if (caps.known && !caps.tier_is_real(gen_quality)) {
                        for (const char* fallback : {"fast", "high", "ultra"})
                            if (caps.tier_is_real(fallback)) {
                                gen_quality = fallback;
                                break;
                            }
                    }

                } else if constexpr (std::is_same_v<T, MsgStatus>) {
                    net_busy = false;
                    if (!m.text.empty()) {
                        net_status = m.text;
                        net_error = m.error;
                        if (selftest && m.text == "cancelled") regen_done = true;
                    } else if (stems_arrived > 0) {
                        net_status = std::to_string(stems_arrived) + " stems" +
                                     (cache_hits ? " (" + std::to_string(cache_hits) +
                                                   " cached)" : "");
                    }
                }
            }, message);
        }

        if (changed) {
            session.loop_end = session.length_frames();
            // Fit on the first content to arrive. Re-fitting on every later
            // stem would yank the view out from under someone who had already
            // started navigating.
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
        // Built first, not inlined into the argument list: a temporary string's
        // c_str() in a variadic call is the kind of lifetime question nobody
        // should have to re-derive when reading a log line.
        std::string job_note;
        if (!jobs.empty()) {
            const int secs = static_cast<int>(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - jobs.front().started).count());
            job_note = " [" + jobs.front().kind + " " + std::to_string(secs) + "s]";
        }
        std::printf(
            "STUDIO device=%s rate=%u latency_ms=%.1f | sources=%d pending=%d "
            "decode_ms=%.0f | tracks=%zu clips=%zu | playhead=%.2fs xruns=%u "
            "| theme=%d keys=%d net=%s stems=%d/%d cached=%d jobs=%zu%s | paint_ms=%.2f (build %.2f skia %.2f) worst=%.2f frames=%d columns=%lld\n",
            g_device.running() ? g_device.backend().c_str() : "NONE",
            g_device.rate(), 1000.0 * g_device.latency_frames() /
                std::max(1u, g_device.rate()),
            loaded_sources, g_pending.load(), last_decode_ms,
            session.tracks.size(), session.clips.size(),
            static_cast<double>(playhead()) / std::max(1u, session.rate),
            g_mixer.xruns(), theme_ok ? 1 : 0, keys_focused ? 1 : 0,
            net_status.c_str(), stems_arrived, stems_expected,
            cache_hits, jobs.size(), job_note.c_str(),
            last_paint_ms, build_ms, submit_ms, worst_paint_ms, frames,
            static_cast<long long>(columns));
        std::fflush(stdout);
    }

    vik::AnyElement render(vik::Window&, vik::Context<Studio>& cx);
    vik::AnyElement generate_modal(vik::Context<Studio>& cx);
    vik::AnyElement layer_modal(vik::Context<Studio>& cx);
    vik::AnyElement settings_modal(vik::Context<Studio>& cx);
    vik::ui::MenuBuilder ai_tools_menu(vik::Context<Studio>& cx);
    vik::AnyElement tool_modal(vik::Context<Studio>& cx, const char* title,
                               const char* blurb, vik::AnyElement body,
                               const char* action, bool ready,
                               std::function<void(Studio&, vik::App&)> run);
    vik::AnyElement current_tool_modal(vik::Context<Studio>& cx);
    vik::AnyElement quality_chip(vik::Context<Studio>& cx, const char* tier,
                                 bool on);
};

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

/// The generate modal, as its own function rather than another branch inside
/// render().
///
/// Not a style choice. MSVC reserves frame space for the temporaries in every
/// branch of a function, taken or not, and render() had grown large enough that
/// adding this block overflowed the stack on the first draw -- a crash before
/// anything appeared, in a modal that was not even open. Each panel owning its
/// own frame keeps that from being a lurking limit.
/// The instrument picker, in its own frame. See generate_modal for why.
/// Connection settings, in its own frame. See generate_modal for why.
vik::AnyElement Studio::settings_modal(vik::Context<Studio>& cx) {
    auto icon_button = [&cx](const char* id, const char* icon, bool active, auto fn) {
        return vik::div().id(id).px_3().py_2().rounded_md()
            .flex_row().items_center().justify_center()
            .bg(active ? vik::rgb(0x3a4a68) : vik::rgb(0x21252f))
            .border_1().border_color(vik::rgb(0x3b4250))
            .cursor_pointer()
            .on_click(cx.listener(fn))
            .child(vik::ui::phosphor(icon, vik::ui::PhWeight::Fill)
                       .size(16.0f).color(vik::rgb(0xc9cedb)));
    };

        // Connection only. Nothing here lists the pod's projects: the pod
        // generates audio on request, and the project is the local document.
        // Offering its scratch workspaces as things to open is what made it
        // look like a document store.
        const bool cloud = settings.mode == mx::Mode::Cloud;

    auto mode_chip = [&cx](const char* label, const char* detail, bool on,
                           mx::Mode which) {
        return vik::div().id("mode").flex_1().flex_col().gap_1().px_3().py_2()
            .rounded_md().cursor_pointer()
            .bg(vik::rgb(on ? 0x3a4a68 : 0x2c313d))
            .border_1().border_color(vik::rgb(on ? 0x5b5bd6 : 0x3b4250))
            .on_click(cx.listener([which](Studio& s, const vik::ClickEvent&,
                                          vik::Window&, vik::Context<Studio>& c) {
                s.set_mode(which, c.app());
                c.notify();
            }))
            .child(vik::text(label).text_color(vik::rgb(on ? 0xffffff : 0xc9cedb)))
            .child(vik::text(detail).text_xs().text_color(vik::rgb(0x8d94a3)));
    };

    return vik::div().absolute().top(0.0f).left(0.0f).right(0.0f).bottom(0.0f)
        .flex_row().items_center().justify_center()
        .occlude()
        .on_mouse_down(vik::MouseButton::Left, cx.listener(
            [](Studio& s, const vik::MouseDownEvent&, vik::Window&,
               vik::Context<Studio>& c) {
                s.show_settings = false;
                c.notify();
            }))
        .bg(vik::rgba(0x00000099))
        .child(vik::div().flex_col().gap_3().p_4().w_px(500.0f)
            .occlude()
            .on_mouse_down(vik::MouseButton::Left,
                [](const vik::MouseDownEvent&, vik::Window& w, vik::App&) {
                    w.stop_propagation();
                })
            .rounded_lg().bg(vik::rgb(0x232834))
            .border_1().border_color(vik::rgb(0x3b4250))
            .child(vik::div().flex_row().items_center().justify_between()
                .child(vik::text("Settings").text_xl().text_color(vik::white()))
                .child(icon_button("close", "x", false,
                    [](Studio& s, const vik::ClickEvent&, vik::Window&,
                       vik::Context<Studio>& c) {
                        s.show_settings = false;
                        c.notify();
                    })))

            .child(vik::text("Where renders happen")
                       .text_xs().text_color(vik::rgb(0x8d94a3)))
            // Two addresses, one switch. The client only ever speaks to a URL,
            // so a mode is which URL -- not a second implementation, and not a
            // reason to retype the pod address every time you switch back.
            .child(vik::div().flex_row().gap_2()
                .child(mode_chip("Local", "this laptop, turbo tier", !cloud,
                                 mx::Mode::Local))
                .child(mode_chip("Cloud", "RunPod, all tiers", cloud,
                                 mx::Mode::Cloud)))

            .child(vik::text(pod_url.empty() ? "no address set" : pod_url)
                       .text_color(vik::rgb(0xc9cedb)))
            .child(vik::text(net_status).text_xs()
                       .text_color(vik::rgb(net_error ? 0xe05c72 : 0x8d94a3)))

            .child(vik::div().flex_row().items_center().gap_2()
                .child(icon_button("s-conn", "plugs-connected", false,
                    [](Studio& s, const vik::ClickEvent&, vik::Window&,
                       vik::Context<Studio>& c) { s.connect(c.app()); c.notify(); }))
                .child(vik::text("Test the connection")
                           .text_xs().text_color(vik::rgb(0x8d94a3))))

            // Say what each mode needs, rather than leaving a failed connection
            // to be interpreted.
            .child(vik::text(
                    cloud
                        ? (pod_token.empty()
                               ? "No token stored. Start once with --connect <url> "
                                 "--token <token>."
                               : "Token stored, encrypted for this Windows account.")
                        : "Start the engine first:  pwsh -File tools\\start-local.ps1"
                          "   (install-local.ps1 the first time)")
                .text_xs().text_color(vik::rgb(0x6c7383))))
        .into_any();
}

vik::AnyElement Studio::layer_modal(vik::Context<Studio>& cx) {
    auto icon_button = [&cx](const char* id, const char* icon, bool active, auto fn) {
        return vik::div().id(id).px_3().py_2().rounded_md()
            .flex_row().items_center().justify_center()
            .bg(active ? vik::rgb(0x3a4a68) : vik::rgb(0x21252f))
            .border_1().border_color(vik::rgb(0x3b4250))
            .cursor_pointer()
            .on_click(cx.listener(fn))
            .child(vik::ui::phosphor(icon, vik::ui::PhWeight::Fill)
                       .size(16.0f).color(vik::rgb(0xc9cedb)));
    };

        auto chips = vik::div().flex_row().wrap().gap_2();
        for (const char* name : kTrackClasses) {
            chips = std::move(chips).child(
                vik::div().id("chip").px_3().py_2().rounded_md().cursor_pointer()
                    .bg(vik::rgb(0x2c313d))
                    .border_1().border_color(vik::rgb(0x3b4250))
                    .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x3a4a68)); })
                    .on_click(cx.listener([name](Studio& s, const vik::ClickEvent&,
                                                 vik::Window&,
                                                 vik::Context<Studio>& c) {
                        s.add_layer(name, c.app());
                        c.notify();
                    }))
                    .child(vik::text(name).text_color(vik::rgb(0xe6e8ec))));
        }

        return vik::div().absolute().top(0.0f).left(0.0f).right(0.0f).bottom(0.0f)
            .flex_row().items_center().justify_center()
            // Swallow mouse input. Without this a modal is only visually on
            // top: clicks land on the timeline behind it, so opening Generate
            // moved the playhead. Dimming the background is not the same as
            // blocking it.
            .occlude()
            // A click on the dimmed area dismisses, as modals do. The card
            // below stops propagation so clicking its contents does not.
            .on_mouse_down(vik::MouseButton::Left, cx.listener(
                [](Studio& s, const vik::MouseDownEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    s.show_generate = s.show_layers = s.show_settings = false;
                    c.notify();
                }))
            .bg(vik::rgba(0x00000099))
            .child(vik::div().flex_col().gap_3().p_4().w_px(480.0f)
                .occlude()
                .on_mouse_down(vik::MouseButton::Left,
                    [](const vik::MouseDownEvent&, vik::Window& w, vik::App&) {
                        w.stop_propagation();
                    })
                .rounded_lg().bg(vik::rgb(0x232834))
                .border_1().border_color(vik::rgb(0x3b4250))
                .child(vik::div().flex_row().items_center().justify_between()
                    .child(vik::text("Add a Layer").text_xl().text_color(vik::white()))
                    .child(icon_button("lclose", "x", false,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.show_layers = false;
                            c.notify();
                        })))
                // Say what it will act on. A tool whose scope is invisible is a
                // tool people use once and then stop trusting.
                .child(vik::text(selection.active()
                        ? std::format("Over the selection, {:.2f}-{:.2f}s",
                                      static_cast<double>(selection.begin()) / session.rate,
                                      static_cast<double>(selection.end()) / session.rate)
                        : std::string("Over the whole arrangement "
                                      "-- shift-drag a range first to narrow it"))
                    .text_xs().text_color(vik::rgb(0x8d94a3)))
                .child(std::move(chips)))
            .into_any();
    }

/// The AI Tools menu, built once and used by both the right-click menu and the
/// dock, so the two cannot offer different things.
///
/// Tools that exist but cannot act yet are shown disabled rather than hidden: a
/// menu that changes shape depending on state is harder to learn than one where
/// an item explains why it is unavailable.
/// The shell every AI tool shares: "Selection a - b" in the corner, a title, a
/// sentence about what it does, the tool's own controls, and one primary
/// button. Taken from the reference, where every tool looks like this -- and the
/// selection is restated on every one because a tool whose scope is invisible is
/// a tool people stop trusting.
vik::AnyElement Studio::tool_modal(vik::Context<Studio>& cx, const char* title,
                                   const char* blurb, vik::AnyElement body,
                                   const char* action, bool ready,
                                   std::function<void(Studio&, vik::App&)> run) {
    const std::string range =
        selection.active()
            ? std::format("{:.2f} - {:.2f}s",
                          static_cast<double>(selection.begin()) / session.rate,
                          static_cast<double>(selection.end()) / session.rate)
            : std::string("whole track");

    return vik::div().absolute().top(0.0f).left(0.0f).right(0.0f).bottom(0.0f)
        .flex_row().items_center().justify_center()
        .occlude()
        .bg(vik::rgba(0x000000aa))
        .on_mouse_down(vik::MouseButton::Left, cx.listener(
            [](Studio& s, const vik::MouseDownEvent&, vik::Window&,
               vik::Context<Studio>& c) {
                s.tool = Studio::Tool::None;
                c.notify();
            }))
        .child(vik::div().flex_col().gap_3().p_4().w_px(520.0f)
            .occlude()
            .on_mouse_down(vik::MouseButton::Left,
                [](const vik::MouseDownEvent&, vik::Window& w, vik::App&) {
                    w.stop_propagation();
                })
            .rounded_lg().bg(vik::rgb(0x232834))
            .border_1().border_color(vik::rgb(0x3b4250))
            .child(vik::div().flex_row().items_start().justify_between()
                .child(vik::div().flex_col()
                    .child(vik::text("Selection").text_xs()
                               .text_color(vik::rgb(0x6c7383)))
                    .child(vik::text(range).text_xs()
                               .text_color(vik::rgb(0xc9cedb))))
                .child(vik::text(title).text_xl().text_color(vik::white()))
                .child(vik::div().id("tclose").px_2().py_1().rounded_md()
                    .cursor_pointer()
                    .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x2c313d)); })
                    .on_click(cx.listener([](Studio& s, const vik::ClickEvent&,
                                             vik::Window&,
                                             vik::Context<Studio>& c) {
                        s.tool = Studio::Tool::None;
                        c.notify();
                    }))
                    .child(vik::ui::phosphor("x", vik::ui::PhWeight::Bold)
                               .size(14.0f).color(vik::rgb(0xc9cedb)))))
            .child(vik::text(blurb).text_xs().text_color(vik::rgb(0x8d94a3)))
            .child(std::move(body))
            .child(vik::div().id("trun").px_4().py_2().rounded_md()
                .flex_row().justify_center()
                .bg(vik::rgb(ready ? 0x5b5bd6 : 0x2c313d))
                .cursor_pointer()
                .on_click(cx.listener([run = std::move(run), ready](
                        Studio& s, const vik::ClickEvent&, vik::Window&,
                        vik::Context<Studio>& c) {
                    if (!ready) return;
                    s.tool = Studio::Tool::None;
                    run(s, c.app());
                    c.notify();
                }))
                .child(vik::text(action)
                           .text_color(vik::rgb(ready ? 0xffffff : 0x6c7383)))))
        .into_any();
}

/// Whichever tool is open, or an empty element.
vik::AnyElement Studio::current_tool_modal(vik::Context<Studio>& cx) {
    auto chip = [&cx](const std::string& label, bool on, auto fn) {
        return vik::div().id("tc").px_3().py_2().rounded_md().cursor_pointer()
            .bg(vik::rgb(on ? 0x3a4a68 : 0x2c313d))
            .border_1().border_color(vik::rgb(on ? 0x5b5bd6 : 0x3b4250))
            .on_click(cx.listener(fn))
            .child(vik::text(label).text_color(vik::rgb(on ? 0xffffff : 0xc9cedb)));
    };

    switch (tool) {
    case Tool::Splitter: {
        // The reference's dialog, option for option.
        auto body = vik::div().flex_col().gap_2()
            .child(chip("Basic: Vocal + Instrumental", split_tier == "basic",
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.split_tier = "basic"; c.notify(); }))
            .child(chip("Professional: 6 stems", split_tier == "professional",
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    s.split_tier = "professional";
                    c.notify();
                }))
            .child(chip("Advanced: all detected stems", split_tier == "advanced",
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.split_tier = "advanced"; c.notify(); }))
            .child(chip(split_remove_reverb
                            ? "[x] Remove reverb and backing vocals"
                            : "[ ] Remove reverb and backing vocals",
                        split_remove_reverb,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    s.split_remove_reverb = !s.split_remove_reverb;
                    c.notify();
                }))
            .into_any();
        return tool_modal(cx, "Stem Splitter",
                          "Separate the take into parts you can edit on their own.",
                          std::move(body), "Split", !open_project_id.empty(),
                          [](Studio& s, vik::App& app) { s.run_split(app); });
    }
    case Tool::Layer: {
        auto chips = vik::div().flex_row().wrap().gap_2();
        for (const char* name : kTrackClasses) {
            chips = std::move(chips).child(
                vik::div().id("lc").px_3().py_2().rounded_md().cursor_pointer()
                    .bg(vik::rgb(0x2c313d))
                    .border_1().border_color(vik::rgb(0x3b4250))
                    .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x3a4a68)); })
                    .on_click(cx.listener([name](Studio& s, const vik::ClickEvent&,
                                                 vik::Window&,
                                                 vik::Context<Studio>& c) {
                        s.tool = Studio::Tool::None;
                        s.add_layer(name, c.app());
                        c.notify();
                    }))
                    .child(vik::text(name).text_color(vik::rgb(0xe6e8ec))));
        }
        return tool_modal(cx, "Add a Layer",
                          "Write a new part against what is already here. Pick an "
                          "instrument to start.",
                          std::move(chips).into_any(),
                          "Pick an instrument above", false,
                          [](Studio&, vik::App&) {});
    }
    case Tool::Inspire:
        return tool_modal(cx, "Inspire Me",
                          "Render this part again over the selection, keeping "
                          "everything outside it.",
                          vik::div().into_any(), "Generate",
                          selection.active() && !net_busy,
                          [](Studio& s, vik::App& app) { s.regenerate(app); });
    case Tool::Enhance:
        return tool_modal(cx, "Music Enhancer",
                          "Clean up this part in place.",
                          vik::div().into_any(), "Enhance",
                          selection.active() && !net_busy,
                          [](Studio& s, vik::App& app) { s.run_isolate(app); });
    case Tool::Extend: {
        auto body = vik::div().flex_row().gap_2().items_center()
            .child(vik::text("bars").text_xs().text_color(vik::rgb(0x6c7383)))
            .child(chip("4", extend_bars == 4,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.extend_bars = 4; c.notify(); }))
            .child(chip("8", extend_bars == 8,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.extend_bars = 8; c.notify(); }))
            .child(chip("16", extend_bars == 16,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) { s.extend_bars = 16; c.notify(); }))
            .into_any();
        return tool_modal(cx, "Extend",
                          "Continue this part past where it ends.",
                          std::move(body), "Extend",
                          selection.active() && !net_busy,
                          [](Studio& s, vik::App& app) { s.run_extend(app); });
    }
    case Tool::Voice:
        return tool_modal(cx, "Voice Changer",
                          "No voices are loaded from this pod yet, so there is "
                          "nothing to convert to.",
                          vik::div().into_any(), "Convert", false,
                          [](Studio&, vik::App&) {});
    case Tool::None:
    default:
        return vik::div().into_any();
    }
}

vik::ui::MenuBuilder Studio::ai_tools_menu(vik::Context<Studio>&) {
    vik::ui::MenuBuilder menu;
    const bool connected = !open_project_id.empty() && !net_busy;
    const bool ranged = connected && selection.active();

    auto entry = [&menu, handle = self](const char* label, bool enabled, Tool which) {
        if (!enabled) {
            menu.disabled_item(label);
            return;
        }
        menu.item(label, [handle, which](vik::Window&, vik::App& app) {
            app.update_entity(handle, [which](Studio& s, vik::Context<Studio>& c) {
                s.tool = which;
                c.notify();
            });
        });
    };

    // Ordered as in the reference, and disabled with a reason rather than
    // hidden: a menu that changes shape with state is harder to learn than one
    // where an item explains why it cannot run.
    entry("Inspire Me", ranged, Tool::Inspire);
    entry("Add a Layer", connected, Tool::Layer);
    entry("Music Enhancer", ranged, Tool::Enhance);
    entry("Voice Changer", ranged, Tool::Voice);
    entry("Stem Splitter", connected, Tool::Splitter);
    entry("Extend", ranged, Tool::Extend);
    menu.separator();
    menu.disabled_item("Vocal to MIDI");        // no endpoint for it yet
    return menu;
}

/// One quality chip, greyed and annotated when the engine would substitute.
vik::AnyElement Studio::quality_chip(vik::Context<Studio>& cx, const char* tier,
                                     bool on) {
    const bool real = !caps.known || caps.tier_is_real(tier);

    std::string label = tier;
    if (!real) {
        // Say what it would actually be, not merely that it is unavailable.
        for (const auto& [name, model] : caps.effective) {
            if (name != tier) continue;
            std::string shortened = model;
            const std::string prefix = "acestep-v15-";
            if (shortened.rfind(prefix, 0) == 0) shortened = shortened.substr(prefix.size());
            label += "  (renders as " + shortened + ")";
            break;
        }
    }

    auto chip = vik::div().id("qc").px_3().py_1().rounded_md()
        .bg(vik::rgb(on ? 0x3a4a68 : 0x2c313d))
        .child(vik::text(label).text_color(
            vik::rgb(!real ? 0x565c6b : (on ? 0xffffff : 0x8d94a3))));
    if (real) {
        chip = std::move(chip).cursor_pointer().on_click(
            cx.listener([tier](Studio& s, const vik::ClickEvent&, vik::Window&,
                               vik::Context<Studio>& c) {
                s.gen_quality = tier;
                c.notify();
            }));
    }
    return std::move(chip).into_any();
}

vik::AnyElement Studio::generate_modal(vik::Context<Studio>& cx) {
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

        auto chip = [&cx](const char* label, bool on, auto fn) {
            return vik::div().id("c").px_3().py_1().rounded_md().cursor_pointer()
                .bg(vik::rgb(on ? 0x3a4a68 : 0x2c313d))
                .on_click(cx.listener(fn))
                .child(vik::text(label).text_color(vik::rgb(on ? 0xffffff : 0x8d94a3)));
    };

    return vik::div().absolute().top(0.0f).left(0.0f).right(0.0f).bottom(0.0f)
            .flex_row().items_center().justify_center()
            // Swallow mouse input. Without this a modal is only visually on
            // top: clicks land on the timeline behind it, so opening Generate
            // moved the playhead. Dimming the background is not the same as
            // blocking it.
            .occlude()
            // A click on the dimmed area dismisses, as modals do. The card
            // below stops propagation so clicking its contents does not.
            .on_mouse_down(vik::MouseButton::Left, cx.listener(
                [](Studio& s, const vik::MouseDownEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    s.show_generate = s.show_layers = s.show_settings = false;
                    c.notify();
                }))
            .bg(vik::rgba(0x000000aa))
            .child(vik::div().flex_col().gap_3().p_4().w_px(560.0f)
                .occlude()
                .on_mouse_down(vik::MouseButton::Left,
                    [](const vik::MouseDownEvent&, vik::Window& w, vik::App&) {
                        w.stop_propagation();
                    })
                .rounded_lg().bg(vik::rgb(0x232834))
                .border_1().border_color(vik::rgb(0x3b4250))
                .child(vik::div().flex_row().items_center().justify_between()
                    .child(vik::text("Generate").text_xl().text_color(vik::white()))
                    .child(icon_button("gclose", "x", false,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.show_generate = false;
                            c.notify();
                        })))
                // Tags, because that is what the model reads: prose gets
                // averaged into nothing, and three to seven tags is the
                // documented sweet spot.
                .child(vik::text("Style tags -- genre, instruments, mood, tempo")
                           .text_xs().text_color(vik::rgb(0x8d94a3)))
                .child(vik::div().flex_row().wrap().gap_2()
                    .child(chip("warm indie soul", gen_prompt == "warm indie soul, brushed drums, rhodes",
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.gen_prompt = "warm indie soul, brushed drums, rhodes";
                            c.notify();
                        }))
                    .child(chip("cinematic strings", gen_prompt == "cinematic strings, timpani, epic, slow",
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.gen_prompt = "cinematic strings, timpani, epic, slow";
                            c.notify();
                        }))
                    .child(chip("lofi hip hop", gen_prompt == "lofi hip hop, dusty rhodes, vinyl, 80 bpm",
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.gen_prompt = "lofi hip hop, dusty rhodes, vinyl, 80 bpm";
                            c.notify();
                        }))
                    .child(chip("kids lullaby", gen_prompt == "children's lullaby, music box, celesta, gentle",
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.gen_prompt = "children's lullaby, music box, celesta, gentle";
                            c.notify();
                        })))
                .child(vik::text(gen_prompt.empty() ? "(pick a style)" : gen_prompt)
                           .text_color(vik::rgb(0xc9cedb)))
                // A tier the engine cannot honour is shown greyed with what it
                // would really render, rather than offered and quietly
                // substituted. On a laptop only the small tier is installed;
                // on a pod it depends what was downloaded.
                .child(vik::div().flex_row().gap_2().items_center()
                    .child(vik::text("quality").text_xs().text_color(vik::rgb(0x6c7383)))
                    .child(quality_chip(cx, "fast", gen_quality == "fast"))
                    .child(quality_chip(cx, "high", gen_quality == "high"))
                    .child(quality_chip(cx, "ultra", gen_quality == "ultra")))
                .child(vik::div().flex_row().gap_2().items_center()
                    .child(vik::text("length").text_xs().text_color(vik::rgb(0x6c7383)))
                    .child(chip("8 bars", gen_bars == 8,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) { s.gen_bars = 8; c.notify(); }))
                    .child(chip("16 bars", gen_bars == 16,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) { s.gen_bars = 16; c.notify(); }))
                    .child(chip("32 bars", gen_bars == 32,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) { s.gen_bars = 32; c.notify(); })))
                .child(vik::div().flex_row().items_center().gap_3()
                    .child(vik::div().id("go").px_4().py_2().rounded_md().cursor_pointer()
                        .bg(vik::rgb(0x3a4a68))
                        .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x4a5f86)); })
                        .on_click(cx.listener([](Studio& s, const vik::ClickEvent&,
                                                 vik::Window&,
                                                 vik::Context<Studio>& c) {
                            s.generate(s.gen_prompt, c.app());
                            c.notify();
                        }))
                        .child(vik::text("Generate").text_color(vik::white())))
                    .child(vik::text("Renders on the pod, then splits into stems "
                                     "so each part is editable.")
                        .text_xs().text_color(vik::rgb(0x565c6b)))))
            .into_any();
}

/// Is a Theme global installed? Tooltips read it during paint, so its absence
/// is a crash waiting for a hover rather than an error at startup.
bool a_theme_present(vik::App& app) {
    return app.try_global<vik::Theme>() != nullptr;
}

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

    auto pill = [](std::string text, uint32_t tint = 0xe6e8ec) {
        return vik::div().px_2().py_1().rounded_md().bg(vik::rgb(0x21252f))
            .child(vik::text(std::move(text)).text_color(vik::rgb(tint)));
    };

    // ACE Studio puts the transport in the middle of the title bar with the
    // project on the left and Export on the right, and keeps it to a handful of
    // glyphs. Following that: the numbers that change while you work sit beside
    // the buttons, and diagnostics move out of the way rather than competing
    // with them.
    auto transport =
        vik::div().flex_row().items_center().px_3().py_2()
            .bg(vik::rgb(0x1b1e27)).border_1().border_color(vik::rgb(0x2c313d))
            .child(vik::div().flex_row().items_center().gap_2().w_px(260.0f)
                .child(vik::text(session.title.empty() ? "Untitled Project"
                                                       : session.title)
                           .text_color(vik::rgb(0xe6e8ec)))
                .child(vik::text(open_project_id.empty() ? "local" : "pod")
                           .text_xs().text_color(vik::rgb(0x6c7383))))

            // centre cluster
            .child(vik::div().flex_1().flex_row().items_center().justify_center().gap_2()
                .child(pill(std::format("{:.1f}", session.bpm)))
                .child(pill(std::format("{} / 4", session.beats_per_bar), 0x8d94a3))
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
                .child(vik::div().flex_col().items_center().ml(10.0f)
                    .child(vik::text(std::format("{}:{:04.1f}", static_cast<int>(pos_s) / 60,
                                                 std::fmod(pos_s, 60.0)))
                               .text_color(vik::rgb(0xe6e8ec)))
                    .child(vik::text(std::format("{}.{}", bar + 1, beat + 1))
                               .text_xs().text_color(vik::rgb(0x6c7383)))))

            // right: export, then the diagnostics that must stay visible
            .child(vik::div().flex_row().items_center().justify_end().gap_3().w_px(260.0f)
                .child(vik::text(std::format("xruns {}", g_mixer.xruns()))
                           .text_xs()
                           .text_color(vik::rgb(g_mixer.xruns() ? 0xe05c72 : 0x4a5060)))
                .child(vik::text(selection.active()
                           ? std::format("sel {:.2f}-{:.2f}s",
                                         static_cast<double>(selection.begin()) / session.rate,
                                         static_cast<double>(selection.end()) / session.rate)
                           : std::string("shift-drag to select"))
                           .text_xs()
                           .text_color(vik::rgb(selection.active() ? 0xffd58a : 0x6c7383)))
                .child(icon_button("export", "export", false,
                    [](Studio& s, const vik::ClickEvent&, vik::Window&,
                       vik::Context<Studio>& c) { s.export_mix(); c.notify(); })));

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
                                                    head, self->selected, self->selection);
            c->restore();
            const auto t1 = std::chrono::steady_clock::now();
            self->columns = stats.columns_drawn;
            self->timeline_px = b.size.width;
            self->canvas_x = b.origin.x;
            self->canvas_y = b.origin.y;
            self->build_ms = stats.build_ms;
            self->submit_ms = stats.submit_ms;
            self->last_paint_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            self->worst_paint_ms = std::max(self->worst_paint_ms, self->last_paint_ms);
            ++self->frames;
        });
    }).size_full();

    // --- job strip ---------------------------------------------------------
    // Above the dock, and only when something is running. Elapsed seconds and
    // queue position are facts; a progress bar would be a number we invented,
    // since the engine reports none.
    vik::AnyElement job_strip = vik::div().into_any();
    if (!jobs.empty()) {
        auto strip = vik::div().flex_col().gap_1();
        for (const auto& job : jobs) {
            const int secs = static_cast<int>(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - job.started).count());
            const std::string label = job.queue_position > 1
                ? ("queued #" + std::to_string(job.queue_position))
                : job.message;
            strip = std::move(strip).child(
                vik::div().flex_row().items_center().gap_3().px_3().py_2()
                    .rounded_md().bg(vik::rgb(0x232834))
                    .border_1().border_color(vik::rgb(0x3b4250))
                    .occlude()
                    .on_mouse_down(vik::MouseButton::Left,
                        [](const vik::MouseDownEvent&, vik::Window& w, vik::App&) {
                            w.stop_propagation();
                        })
                    .child(vik::ui::phosphor("circle-notch", vik::ui::PhWeight::Bold)
                               .size(13.0f).color(vik::rgb(0xd9903c)))
                    .child(vik::text(job.kind).text_color(vik::rgb(0xe6e8ec)))
                    .child(vik::text(label).text_xs().text_color(vik::rgb(0x8d94a3)))
                    .child(vik::text(std::format("{}s", secs))
                               .text_xs().text_color(vik::rgb(0xc9cedb)))
                    .child(vik::div().id("cancel").px_2().py_1().rounded_sm()
                        .cursor_pointer().bg(vik::rgb(0x2c313d))
                        .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x8a3a3a)); })
                        .on_click(cx.listener([id = job.id](Studio& s,
                                                            const vik::ClickEvent&,
                                                            vik::Window&,
                                                            vik::Context<Studio>& c) {
                            s.cancel_job(id, c.app());
                            c.notify();
                        }))
                        .child(vik::text(job.cancelling ? "stoppingâ€¦" : "Cancel")
                                   .text_xs().text_color(vik::rgb(0xe6e8ec)))));
        }
        job_strip = vik::div().absolute().bottom(72.0f).left(0.0f).right(0.0f)
            .flex_row().justify_center()
            .child(std::move(strip))
            .into_any();
    }

    // The bottom dock, which is the piece of ACE Studio's layout people
    // recognise: a floating pill of icon groups over the canvas rather than a
    // toolbar bolted to an edge. Grouped as in the reference --
    // [record sources] [AI tools] [global] -- with dividers between groups.
    auto dock_tool = [&cx](const char* id, const char* icon, const char* tip,
                           bool enabled, auto fn) {
        auto button = vik::div().id(id).px_3().py_2().rounded_md()
            .flex_row().items_center().justify_center()
            .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x333a4a)); })
            .child(vik::ui::phosphor(icon, vik::ui::PhWeight::Regular)
                       .size(17.0f)
                       .color(vik::rgb(enabled ? 0xe6e8ec : 0x565c6b)));
        button = std::move(button).tooltip(tip);
        if (enabled) button = std::move(button).cursor_pointer().on_click(cx.listener(fn));
        return button;
    };
    auto divider = [] {
        return vik::div().w_px(1.0f).h_px(20.0f).bg(vik::rgb(0x3b4250)).mx(4.0f);
    };

    const bool armed = selection.active() && !open_project_id.empty() && !net_busy;

    auto dock = vik::div().absolute().bottom(18.0f).left(0.0f).right(0.0f)
        .flex_row().justify_center()
        .child(vik::div().flex_row().items_center().gap_1().px_2().py_1()
            .rounded_lg().bg(vik::rgb(0x232834))
            .border_1().border_color(vik::rgb(0x3b4250))
            // The dock floats inside the timeline, so its clicks bubble to the
            // timeline's handler and moved the playhead behind it. Bubbling
            // runs leaf to root: a child handling a click does not stop an
            // ancestor from also handling it.
            .occlude()
            .on_mouse_down(vik::MouseButton::Left,
                [](const vik::MouseDownEvent&, vik::Window& w, vik::App&) {
                    w.stop_propagation();
                })
            .child(dock_tool("t-gen", "sparkle",
                        pod_url.empty() ? "Set a pod first (gear, bottom left)"
                                        : "Generate  (G)",
                        !pod_url.empty() && !net_busy,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.show_generate = true;
                            c.notify();
                        }))
            .child(dock_tool("t-mic", "microphone-stage", "Record a voice take (not yet)",
                        false, [](Studio&, const vik::ClickEvent&, vik::Window&,
                                  vik::Context<Studio>&) {}))
            .child(dock_tool("t-inst", "guitar", "Instruments (not yet)", false,
                        [](Studio&, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>&) {}))
            .child(divider())
            .child(dock_tool("t-regen", "arrows-clockwise",
                        armed ? "Regenerate the selection  (R)"
                              : "Select a range on a pod track first",
                        armed,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) { s.regenerate(c.app()); c.notify(); }))
            .child(dock_tool("t-layer", "stack-plus",
                        open_project_id.empty()
                            ? "Import a pod project first"
                            : "Add a layer  (L)",
                        !open_project_id.empty() && !net_busy,
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            s.show_layers = true;
                            c.notify();
                        }))
            .child(dock_tool("t-split", "scissors", "Stem splitter (not yet)", false,
                        [](Studio&, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>&) {}))
            .child(dock_tool("t-more", "dots-three",
                        open_project_id.empty() ? "Generate something first"
                                                : "AI tools  (right-click too)",
                        !open_project_id.empty(),
                        [](Studio& s, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>& c) {
                            // Same set as the right-click menu; the reference
                            // offers both routes to the identical list.
                            s.show_tool_picker = !s.show_tool_picker;
                            c.notify();
                        }))
            .child(divider())
            .child(dock_tool("t-pod", "globe",
                        open_project_id.empty() ? "Not connected to a pod"
                                                : "Connected",
                        false,
                        [](Studio&, const vik::ClickEvent&, vik::Window&,
                           vik::Context<Studio>&) {})));

    auto timeline =
        vik::div().flex_1().relative().overflow_hidden()
            .child(std::move(surface))
            .child(std::move(job_strip))
            .child(std::move(dock))
            .on_mouse_down(vik::MouseButton::Left, cx.listener(
                [](Studio& s, const vik::MouseDownEvent& e, vik::Window&,
                   vik::Context<Studio>& c) {
                    // The canvas fills this div, so the div's origin is the
                    // surface origin; event positions are already local.
                    const float x = s.local_x(e.position.x);
                    const float y = s.local_y(e.position.y);
                    // Click anywhere that is not a clip to move the playhead,
                    // the way every editor does it. Restricting that to a 28px
                    // ruler strip made the playhead feel stuck, because almost
                    // nowhere you would click actually moved it.
                    if (y < mx::kRulerHeight) {
                        s.scrubbing = true;
                        s.seek(s.view.frame_at(x));
                    } else if (e.modifiers.shift) {
                        // Shift-drag selects a time range on the track under
                        // the cursor. Plain drag still moves the clip, so the
                        // two never fight over the same gesture.
                        const int lane = mx::track_at(
                            y, static_cast<int>(s.session.tracks.size()));
                        if (lane >= 0) {
                            s.selecting = true;
                            s.selection.track = s.session.tracks[lane].id;
                            s.selection.from = s.selection.to =
                                std::max<int64_t>(0, s.view.frame_at(x));
                        }
                    } else if (const mx::ClipId id = s.clip_at(x, y)) {
                        s.selected = id;
                        s.dragging = id;
                        if (const auto* clip = s.session.find_clip(id))
                            s.drag_grab = s.view.frame_at(x) - clip->start_frame;
                    } else {
                        // Empty lane: move the playhead and start scrubbing.
                        s.selected = 0;
                        s.selection.clear();
                        s.scrubbing = true;
                        s.seek(s.view.frame_at(x));
                    }
                    c.notify();
                }))
            .capture_mouse_move(cx.listener(
                [](Studio& s, const vik::MouseMoveEvent& e, vik::Window&,
                   vik::Context<Studio>& c) {
                    if (s.scrubbing) {
                        s.seek(s.view.frame_at(s.local_x(e.position.x)));
                        c.notify();
                        return;
                    }
                    if (s.selecting) {
                        s.selection.to =
                            std::max<int64_t>(0, s.view.frame_at(s.local_x(e.position.x)));
                        c.notify();
                        return;
                    }
                    if (!s.dragging) return;
                    auto* clip = s.session.find_clip(s.dragging);
                    if (!clip) return;

                    int64_t start = s.view.frame_at(s.local_x(e.position.x)) - s.drag_grab;
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
                    if (s.dragging || s.scrubbing || s.selecting) {
                        s.dragging = 0;
                        s.scrubbing = false;
                        // A selection thinner than a few pixels is a misclick,
                        // not a range; keeping it would arm a tool over nothing.
                        if (s.selecting &&
                            std::llabs(s.selection.to - s.selection.from) <
                                static_cast<int64_t>(4 * s.view.frames_per_pixel))
                            s.selection.clear();
                        s.selecting = false;
                        s.session.loop_end = s.session.length_frames();
                        c.notify();
                    }
                }))
            .on_scroll_wheel(cx.listener(
                [](Studio& s, const vik::ScrollWheelEvent& e, vik::Window&,
                   vik::Context<Studio>& c) {
                    // A plain wheel only reports delta.y. Scrolling the timeline
                    // off delta.x meant an ordinary mouse did nothing at all,
                    // which is most of why navigation felt broken.
                    const double step = e.delta.y != 0.0f ? e.delta.y : e.delta.x;
                    if (step == 0.0) return;

                    if (e.modifiers.control || e.modifiers.alt) {
                        // Ctrl+wheel zooms about the cursor: wheel up zooms in,
                        // and the frame under the pointer stays under it. That
                        // is the behaviour every editor shares, and getting the
                        // direction backwards is instantly wrong to anyone.
                        const double factor = step > 0.0 ? 1.0 / 1.2 : 1.2;
                        s.view.zoom_about(s.local_x(e.position.x), factor, 4.0, 262144.0);
                    } else {
                        // Otherwise scroll the timeline horizontally, since that
                        // is the axis a timeline has.
                        s.view.scroll_frames = std::max<int64_t>(
                            0, s.view.scroll_frames -
                                   static_cast<int64_t>(step * 3.0 *
                                                        s.view.frames_per_pixel));
                    }
                    c.notify();
                }));

    // --- sidebar: local projects ------------------------------------------
    // The list here is the user's own documents, not the pod's. The pod renders
    // audio; it does not hold work, and putting its project list in the primary
    // navigation made it look like it did.
    auto row = [](std::string label, std::string value, uint32_t tint) {
        return vik::div().flex_col().gap_1().px_3().py_2()
            .child(vik::text(std::move(label)).text_xs().text_color(vik::rgb(0x6c7383)))
            .child(vik::text(std::move(value)).text_color(vik::rgb(tint)));
    };

    auto sidebar = vik::div().flex_col().w_px(220.0f)
        .bg(vik::rgb(0x1b1e27)).border_1().border_color(vik::rgb(0x2c313d))
        .child(vik::div().flex_row().items_center().justify_between().px_3().py_2()
            .child(vik::text("Projects").text_color(vik::rgb(0xe6e8ec)))
            .child(vik::div().flex_row().gap_1()
                .child(icon_button("new", "file-plus", false,
                    [](Studio& s, const vik::ClickEvent&, vik::Window&,
                       vik::Context<Studio>& c) { s.new_document(); c.notify(); }))
                .child(icon_button("save", "floppy-disk", dirty,
                    [](Studio& s, const vik::ClickEvent&, vik::Window&,
                       vik::Context<Studio>& c) { s.save_document_now(); c.notify(); }))))
        .child(row("document",
                   document_path.empty()
                       ? std::string(dirty ? "unsaved" : "no file")
                       : document_path.filename().string() + (dirty ? " *" : ""),
                   dirty ? 0xd9903c : 0x8d94a3));

    if (!doc_status.empty())
        sidebar = std::move(sidebar).child(row("", doc_status, 0x6c7383));

    {
        auto list = vik::div().flex_col().flex_1().overflow_hidden();
        const auto documents = local_documents();
        if (documents.empty()) {
            list = std::move(list).child(
                vik::div().px_3().py_2().child(
                    vik::text("No saved projects yet")
                        .text_xs().text_color(vik::rgb(0x565c6b))));
        }
        for (const auto& doc : documents) {
            const bool active = doc == document_path;
            list = std::move(list).child(
                vik::div().id("doc").px_3().py_1().cursor_pointer()
                    .bg(vik::rgb(active ? 0x2c3446 : 0x1b1e27))
                    .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x242936)); })
                    .on_click(cx.listener([path = doc](Studio& s,
                                                       const vik::ClickEvent&,
                                                       vik::Window&,
                                                       vik::Context<Studio>& c) {
                        s.open_document(path, c.app());
                        c.notify();
                    }))
                    .child(vik::text(doc.stem().string().substr(0, 24))
                               .text_color(vik::rgb(active ? 0xffd58a : 0xc9cedb))));
        }
        sidebar = std::move(sidebar).child(std::move(list));
    }

    // The pod lives at the bottom, as a service the app is configured to use.
    sidebar = std::move(sidebar).child(
        vik::div().flex_row().items_center().justify_between().px_3().py_2()
            .border_1().border_color(vik::rgb(0x2c313d))
            .child(vik::div().flex_row().items_center().gap_2()
                .child(vik::ui::phosphor("circle", vik::ui::PhWeight::Fill).size(8.0f)
                           .color(vik::rgb(pod_url.empty() ? 0x565c6b
                                           : (net_error ? 0xe05c72 : 0x56c08a))))
                .child(vik::text(pod_url.empty() ? "No pod" : net_status)
                           .text_xs().text_color(vik::rgb(0x8d94a3))))
            .child(icon_button("settings", "gear", show_settings,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    s.show_settings = !s.show_settings;
                    c.notify();
                })));

    vik::AnyElement overlay = vik::div().into_any();

    if (show_generate) overlay = generate_modal(cx);

    if (show_layers) overlay = layer_modal(cx);

    if (tool != Tool::None)
        overlay = current_tool_modal(cx);
    else if (show_settings && !show_layers && !show_generate)
        overlay = settings_modal(cx);

    // Right-click anywhere on the arrangement opens the tools, which is how the
    // reference reaches every one of them. vikui's context_menu_area already
    // does open-at-cursor, submenu flyouts and dismiss-on-escape, so none of
    // that is ours to write.
    auto timeline_area =
        vik::ui::context_menu_area("timeline-tools", ai_tools_menu(cx),
                                   std::move(timeline).into_any());

    return vik::div().size_full().flex_col().relative().bg(vik::rgb(0x14161d))
        .track_focus(keys)
        .key_context("Studio")
        // Claim focus on any click in the window. Keys are dispatched along the
        // focused node's path, so without this the shortcuts fire into nothing
        // -- and a listener that is simply never called looks exactly like a
        // listener that does not work.
        .capture_mouse_down(vik::MouseButton::Left, cx.listener(
            [](Studio& s, const vik::MouseDownEvent&, vik::Window& w,
               vik::Context<Studio>&) {
                if (s.keys && !s.keys_focused) {
                    w.focus(s.keys);
                    s.keys_focused = true;
                }
            }))
        // Also on the first mouse move: requiring a click before the keyboard
        // works is the kind of rule nobody is told and everybody trips over.
        // Once only, so it never steals focus back from a text field.
        .capture_mouse_move(cx.listener(
            [](Studio& s, const vik::MouseMoveEvent&, vik::Window& w,
               vik::Context<Studio>&) {
                if (s.keys && !s.keys_focused) {
                    w.focus(s.keys);
                    s.keys_focused = true;
                }
            }))
        .on_key_down(cx.listener(
            [](Studio& s, const vik::KeyDownEvent& e, vik::Window&,
               vik::Context<Studio>& c) {
                // A modal owns the keyboard while it is open. Firing
                // timeline shortcuts underneath one is the keyboard version of
                // the click falling through.
                if (s.tool != Tool::None || s.show_generate ||
                    s.show_layers || s.show_settings) {
                    if (e.key == "escape") {
                        s.tool = Studio::Tool::None;
                        s.show_generate = s.show_layers = s.show_settings = false;
                        c.notify();
                    }
                    return;
                }

                if (e.key == "space") s.toggle_play();
                else if (e.key == "r") s.regenerate(c.app());
                else if (e.key == "e") s.export_mix();
                else if (e.key == "g" && !s.pod_url.empty())
                    s.show_generate = !s.show_generate;
                else if (e.key == "l" && !s.open_project_id.empty())
                    s.show_layers = !s.show_layers;
                else if (e.key == "escape") s.selection.clear();
                else if (e.key == "home") s.seek(0);
                else if (e.key == "l") {
                    auto& t = g_mixer.transport;
                    t.looping.store(!t.looping.load());
                } else return;
                c.notify();
            }))
        .child(std::move(transport))
        .child(vik::div().flex_1().flex_row()
                   .child(std::move(sidebar))
                   .child(std::move(headers))
                   .child(std::move(timeline_area)))
        .child(std::move(overlay))
        .into_any();
}

}  // namespace

/// Loads a folder into a session on this thread.
///
/// Shared by --render and --bounce. Both are headless, so there is no UI to
/// keep responsive and no reason for the worker dance the window path needs.
bool load_synchronously(Studio& studio, const fs::path& folder) {
    if (folder.empty()) { studio.build_demo(); return true; }

    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(folder, ec)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".wav" || ext == ".mp3" || ext == ".flac") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { studio.build_demo(); return true; }

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
    return n > 0;
}

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
    if (!load_synchronously(studio, folder)) return 1;

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

/// Load a folder and mix it straight to a file. No window, no device.
int bounce_offline(const fs::path& folder, const fs::path& out) {
    Studio studio;
    studio.session.rate = 48000;
    if (!load_synchronously(studio, folder)) return 1;

    const auto r = mx::bounce(studio.session, out);
    if (!r.ok) {
        std::printf("bounce failed: %s\n", r.error.c_str());
        return 1;
    }
    std::printf("bounced %s\n", out.string().c_str());
    std::printf("  %.2fs of audio from %zu clips on %zu tracks\n",
                r.seconds, studio.session.clips.size(), studio.session.tracks.size());
    std::printf("  %.0f ms to render -- %.1fx faster than real time\n",
                r.render_ms, r.realtime_ratio);
    std::printf("  peak %.3f (%.1f dBFS)%s\n", r.peak,
                20.0 * std::log10(std::max(1e-9f, r.peak)),
                r.clipped ? "  CLIPPED" : "");
    return 0;
}

/// Connect to a pod and report what is there. No window.
///
/// The point is to make the whole network path checkable from a terminal before
/// any UI depends on it: TLS backend, auth, project listing, and a real audio
/// download with its throughput.
int connect_probe(const std::string& url, const std::string& token,
                  const std::string& project_id) {
    mx::net::ApiClient api;
    api.set_base(url);
    api.set_token(token);

    std::printf("tls backend : %s\n", mx::net::tls_backend().c_str());
    std::printf("pod         : %s\n", api.base().c_str());

    const auto health = api.health();
    if (!health.reachable) {
        std::printf("unreachable : %s\n", api.last_error().c_str());
        return 1;
    }
    std::printf("status      : %s%s  gpu=%s\n", health.status.c_str(),
                health.auth_required ? "  (token required)" : "",
                health.gpu.empty() ? "none" : health.gpu.c_str());

    const auto list = api.projects();
    std::printf("projects    : %zu\n", list.size());
    if (list.empty()) {
        std::printf("  %s\n", api.last_error().c_str());
        return 1;
    }

    std::string want = project_id;
    if (want.empty()) {
        for (const auto& p : list) {
            auto detail = api.project(p.id);
            if (detail && !detail->stems.empty()) { want = p.id; break; }
        }
        if (want.empty()) want = list.front().id;
    }

    auto detail = api.project(want);
    if (!detail) {
        std::printf("cannot open %s: %s\n", want.c_str(), api.last_error().c_str());
        return 1;
    }
    std::printf("opened      : %s  (%s)\n", detail->summary.title.c_str(),
                detail->summary.id.c_str());
    std::printf("  %.0f BPM, %d bars, %zu takes, %zu stems\n",
                detail->summary.bpm, detail->summary.bars,
                detail->variations.size(), detail->stems.size());

    for (const auto& stem : detail->stems) {
        const auto* v = stem.version();
        std::printf("  %-14s %-10s %s\n", stem.track_class.c_str(),
                    stem.id.c_str(), v ? v->audio.c_str() : "(no version)");
    }

    // Download one stem for real: a listing proves parsing, not transfer.
    for (const auto& stem : detail->stems) {
        const auto* v = stem.version();
        if (!v || v->audio.empty()) continue;
        const auto dest = fs::temp_directory_path() / ("probe_" + stem.id + ".wav");
        const auto t0 = std::chrono::steady_clock::now();
        if (!api.fetch_audio(detail->summary.id, v->audio, dest.string())) {
            std::printf("download    : FAILED %s\n", api.last_error().c_str());
            return 1;
        }
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const auto bytes = fs::file_size(dest);
        std::printf("download    : %s  %.1f MB in %.1fs (%.1f MB/s)\n",
                    stem.track_class.c_str(), bytes / 1e6, secs,
                    bytes / 1e6 / std::max(1e-6, secs));

        mx::Media media = mx::decode_file(dest, 48000);
        if (!media.ok()) {
            std::printf("decode      : FAILED %s\n", media.error.c_str());
            return 1;
        }
        std::printf("decode      : %.1fs of audio, source %u Hz, %zu pyramid levels\n",
                    media.seconds, media.source_rate, media.peaks.levels.size());
        fs::remove(dest);
        break;
    }
    std::printf("\nconnected end to end.\n");
    return 0;
}

/// Save a session and read it back, so the document format is checkable
/// without a window. A format that only round-trips by hand is one nobody
/// notices breaking.
int document_roundtrip(const fs::path& folder) {
    Studio studio;
    studio.session.rate = 48000;
    if (!load_synchronously(studio, folder)) return 1;
    studio.session.title = "roundtrip";

    const auto path = fs::temp_directory_path() / "roundtrip.mmproj";
    std::string error;
    if (!mx::save_document(studio.session, path, &error)) {
        std::printf("save failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("saved   %s (%ju bytes)\n", path.string().c_str(),
                static_cast<uintmax_t>(fs::file_size(path)));
    std::printf("  before: %zu tracks, %zu clips, %zu sources, %.2fs\n",
                studio.session.tracks.size(), studio.session.clips.size(),
                studio.session.sources.size(),
                studio.session.length_frames() / 48000.0);

    mx::Session reloaded;
    const auto result = mx::load_document(reloaded, path);
    if (!result.ok) {
        std::printf("load failed: %s\n", result.error.c_str());
        return 1;
    }
    std::printf("  after : %zu tracks, %zu clips, %zu sources, %.2fs\n",
                reloaded.tracks.size(), reloaded.clips.size(),
                reloaded.sources.size(), reloaded.length_frames() / 48000.0);

    const bool same = reloaded.tracks.size() == studio.session.tracks.size() &&
                      reloaded.clips.size() == studio.session.clips.size() &&
                      reloaded.sources.size() == studio.session.sources.size() &&
                      reloaded.title == studio.session.title &&
                      reloaded.next_clip == studio.session.next_clip;
    std::printf("  ids   : next_clip %u -> %u\n",
                studio.session.next_clip, reloaded.next_clip);
    std::printf("%s\n", same ? "PASS  document round-trips"
                                : "FAIL  document changed across a round trip");
    return same ? 0 : 1;
}

int main(int argc, char** argv) {
    mx::install_crash_handler();

    fs::path folder;
    fs::path render_to;
    fs::path bounce_to;
    std::string pod_url, pod_token, pod_project;
    bool probe_only = false;
    bool doctest = false;
    double cancel_after = -1.0;
    std::string layer_class;
    std::string gen_prompt;
    std::string open_panel;
    bool stress = false;
    double regen_from = -1.0, regen_to = -1.0;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--selftest") selftest = true;
        else if (arg == "--render" && i + 1 < argc) render_to = argv[++i];
        else if (arg == "--bounce" && i + 1 < argc) bounce_to = argv[++i];
        else if (arg == "--connect" && i + 1 < argc) pod_url = argv[++i];
        else if (arg == "--probe") probe_only = true;
        else if (arg == "--doctest") doctest = true;
        else if (arg == "--cancel-after" && i + 1 < argc)
            cancel_after = std::stod(argv[++i]);
        else if (arg == "--layer" && i + 1 < argc) layer_class = argv[++i];
        else if (arg == "--generate" && i + 1 < argc) gen_prompt = argv[++i];
        else if (arg == "--open" && i + 1 < argc) open_panel = argv[++i];
        else if (arg == "--stress") stress = true;
        else if (arg == "--regen" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t colon = spec.find(':');
            if (colon != std::string::npos) {
                regen_from = std::stod(spec.substr(0, colon));
                regen_to = std::stod(spec.substr(colon + 1));
            }
        }
        else if (arg == "--token" && i + 1 < argc) pod_token = argv[++i];
        else if (arg == "--project" && i + 1 < argc) pod_project = argv[++i];
        else folder = arg;
    }

    if (!render_to.empty()) return render_png(folder, render_to, 1600, 900);
    if (doctest) return document_roundtrip(folder);
    if (!bounce_to.empty()) return bounce_offline(folder, bounce_to);
    if (!pod_url.empty() && probe_only)
        return connect_probe(pod_url, pod_token, pod_project);

    if (!g_device.start(g_mixer, 48000))
        std::printf("audio: %s\n", g_device.error().c_str());

    vik::App::run([&](vik::App& app) {
        // Install the theme globals before any window exists.
        //
        // Tooltips are painted by vikui itself and read active_theme(app),
        // which is app.global<Theme>() -- with no theme registered that is a
        // null dereference inside Div::paint. Nothing in this app reads the
        // theme directly, so the omission was invisible until a tooltip was
        // shown, and then it crashed on hovering a dock button rather than on
        // startup.
        vik::init_themes(app);

        app.open_window(
            vik::WindowOptions{.title = "musicX Studio", .size = {1200.0f, 720.0f}},
            [&](vik::Window&, vik::App& app) {
                // Created before the entity that stores it, not inside its
                // update: making an entity while another is being updated
                // re-enters entity storage, and the Studio& held across that
                // call does not survive it. The symptom was a crash at startup,
                // before anything drew.
                // Created before the entity that stores it: making an entity
                // while another is being updated re-enters entity storage, and
                // the Studio& held across that call does not survive it.
                auto key_focus = app.focus_handle();

                auto handle = app.add_entity<Studio>();
                app.update_entity(handle, [&](Studio& s, vik::Context<Studio>& c) {
                    s.selftest = selftest;
                    s.keys = key_focus;
                    s.self = handle;
                    s.settings = mx::Settings::load();
                    // Flags win over stored settings: a scripted run must be
                    // reproducible regardless of what this machine remembers.
                    if (!pod_url.empty()) {
                        // An explicit --connect is a cloud address unless it is
                        // plainly this machine.
                        const bool loopback =
                            pod_url.find("127.0.0.1") != std::string::npos ||
                            pod_url.find("localhost") != std::string::npos;
                        s.settings.mode = loopback ? mx::Mode::Local : mx::Mode::Cloud;
                        (loopback ? s.settings.local_url : s.settings.pod_url) = pod_url;
                        if (!loopback) s.settings.pod_token = pod_token;
                        s.settings.save();
                    }
                    s.pod_url = s.settings.active_url();
                    s.pod_token = s.settings.active_token();
                    s.regen_from = regen_from;
                    s.regen_to = regen_to;
                    s.cancel_after = cancel_after;
                    s.scripted_layer = layer_class;
                    s.scripted_prompt = gen_prompt;
                    // Open a panel at startup so every one of them is drawn at
                    // least once in a scripted run. A modal that only renders
                    // when a human clicks it is a modal nothing tests.
                    s.show_generate = (open_panel == "generate");
                    s.show_layers   = (open_panel == "layers");
                    s.show_settings = (open_panel == "settings");
                    if (open_panel == "splitter") s.tool = Studio::Tool::Splitter;
                    else if (open_panel == "inspire") s.tool = Studio::Tool::Inspire;
                    else if (open_panel == "enhance") s.tool = Studio::Tool::Enhance;
                    else if (open_panel == "extend") s.tool = Studio::Tool::Extend;
                    else if (open_panel == "voice") s.tool = Studio::Tool::Voice;
                    else if (open_panel == "tool-layer") s.tool = Studio::Tool::Layer;
                    s.stress = stress;
                    // Read the theme once at startup. It is what tooltips
                    // dereference, and a missing one used to surface only when
                    // somebody hovered a button.
                    s.theme_ok = a_theme_present(c.app());
                    // The device decides the rate; the session follows it,
                    // because everything decoded is resampled to it once.
                    if (g_device.running()) s.session.rate = g_device.rate();
                    s.session.loop_start = 0;
                    if (!pod_url.empty()) {
                        if (!pod_project.empty()) s.open_project(pod_project, app);
                        else s.connect(app);
                    } else if (folder.empty()) {
                        s.build_demo();
                    } else {
                        s.load_folder(folder, app);
                    }
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

                        if (!s.scripted_prompt.empty() && !s.regen_fired &&
                            !s.net_busy) {
                            s.regen_fired = true;
                            std::printf("GENERATE %s\n", s.scripted_prompt.c_str());
                            std::fflush(stdout);
                            s.generate(s.scripted_prompt, c.app());
                        }

                        if (!s.scripted_layer.empty() && !s.regen_fired &&
                            !s.net_busy && s.stems_arrived > 0) {
                            s.regen_fired = true;
                            std::printf("LAYER adding %s\n",
                                        s.scripted_layer.c_str());
                            std::fflush(stdout);
                            s.add_layer(s.scripted_layer, c.app());
                        }

                        // Fire the scripted regenerate once, once content is
                        // actually present -- doing it earlier would test the
                        // empty-session path instead.
                        if (s.regen_from >= 0.0 && !s.regen_fired && !s.net_busy &&
                            s.stems_arrived > 0 && !s.session.tracks.empty()) {
                            s.regen_fired = true;
                            s.selection.track = s.session.tracks.front().id;
                            s.selection.from =
                                static_cast<int64_t>(s.regen_from * s.session.rate);
                            s.selection.to =
                                static_cast<int64_t>(s.regen_to * s.session.rate);
                            std::printf("REGEN selecting %.2f-%.2fs on track %u\n",
                                        s.regen_from, s.regen_to, s.selection.track);
                            std::fflush(stdout);
                            s.regenerate(c.app());
                        }

                        // Scripted cancel: prove the path exists rather than
                        // trusting that a button wired to an endpoint works.
                        if (s.cancel_after >= 0.0 && !s.jobs.empty() &&
                            !s.jobs.front().cancelling) {
                            const double age = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                s.jobs.front().started).count();
                            if (age >= s.cancel_after) {
                                std::printf("CANCEL requesting stop of %s after %.1fs\n",
                                            s.jobs.front().id.c_str(), age);
                                std::fflush(stdout);
                                s.cancel_job(s.jobs.front().id, c.app());
                            }
                        }

                        // Cycle every panel and both selection states as fast
                        // as the app will draw them. The crash that prompted
                        // this appeared only after minutes of clicking, so the
                        // test has to click.
                        if (s.stress) {
                            ++s.stress_tick;
                            const int phase = (s.stress_tick / 7) % 5;
                            s.show_generate = (phase == 1);
                            s.show_layers   = (phase == 2);
                            s.show_settings = (phase == 3);
                            if (phase == 4 && !s.session.tracks.empty()) {
                                s.selection.track = s.session.tracks.front().id;
                                s.selection.from = 0;
                                s.selection.to = s.session.rate * 2;
                            } else if (phase == 0) {
                                s.selection.clear();
                            }
                            c.notify();
                        }

                        const bool rolling = g_mixer.transport.playing.load();
                        g_mixer.collect(g_device.running());
                        // A job strip shows elapsed seconds, so it has to repaint
                        // while a render runs even though nothing else changed.
                        if (landed || rolling || s.meter > 0.001f ||
                            !s.jobs.empty())
                            c.notify();
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
                        const int budget = s.regen_from >= 0.0 ? 180 : 8;
                        ++s.reports;
                        done = s.selftest &&
                               (s.regen_done || (s.reports >= budget && !s.net_busy));
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

    // The UI is gone; nothing may wake it now.
    g_ui_alive.store(false, std::memory_order_release);

    // Wait for detached workers to unwind before static destruction takes the
    // inbox out from under them. Bounded, because a wedged network call must
    // not hold the process open forever -- but generous enough that a normal
    // download finishes.
    for (int waited = 0; g_workers.load(std::memory_order_acquire) > 0 && waited < 100;
         ++waited)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (const int stuck = g_workers.load(std::memory_order_acquire); stuck > 0)
        std::printf("note: %d worker(s) still running at exit\n", stuck);

    g_device.stop();
    return 0;
}
