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
    mx::Media    media;
};

struct MsgStatus {
    std::string text;
    bool        error = false;
};

using UiMessage = std::variant<MsgDecoded, MsgProjects, MsgProjectOpened,
                               MsgStemArrived, MsgStemUpdated, MsgStatus>;

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

    bool  fitted      = false;
    float timeline_px = 1100.0f;   // last measured canvas width

    // --- network ------------------------------------------------------------
    std::string pod_url;
    std::string pod_token;
    std::string net_status = "not connected";
    bool        net_busy = false;
    bool        net_error = false;
    std::vector<mx::net::ProjectSummary> projects;
    std::string open_project_id;
    int         stems_expected = 0;
    int         stems_arrived = 0;
    int         cache_hits = 0;

    bool selftest = false;
    int  reports  = 0;
    // Drives the select-and-regenerate loop without a mouse, so the milestone
    // is checkable from a terminal rather than only by hand.
    double regen_from = -1.0, regen_to = -1.0;
    bool   regen_fired = false;
    bool   regen_done = false;

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
            g_inbox.push(MsgDecoded{
                id, std::move(media),
                std::chrono::duration<double, std::milli>(t1 - t0).count()});
            // Without this the UI can sit in SDL_WaitEventTimeout until the
            // user happens to move the mouse.
            platform->wake();
        }).detach();
    }

    /// Connect and list what is on the pod. Runs on a worker.
    void connect(vik::App& app) {
        if (net_busy || pod_url.empty()) return;
        net_busy = true;
        net_status = "connecting…";
        net_error = false;
        auto* platform = &app.platform();
        std::thread([url = pod_url, token = pod_token, platform] {
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);
            const auto health = api.health();
            if (!health.reachable) {
                g_inbox.push(MsgStatus{"cannot reach the pod: " + api.last_error(), true});
                platform->wake();
                return;
            }
            auto list = api.projects();
            if (list.empty() && !api.last_error().empty()) {
                g_inbox.push(MsgStatus{api.last_error(), true});
                platform->wake();
                return;
            }
            g_inbox.push(MsgProjects{std::move(list)});
            platform->wake();
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
            mx::net::ApiClient api;
            api.set_base(url);
            api.set_token(token);

            auto detail = api.project(id);
            if (!detail) {
                g_inbox.push(MsgStatus{"cannot open project: " + api.last_error(), true});
                platform->wake();
                return;
            }
            g_inbox.push(MsgProjectOpened{detail->summary.id, detail->summary.title,
                                          detail->summary.bpm, detail->summary.bars,
                                          static_cast<int>(detail->stems.size())});
            platform->wake();

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
                    platform->wake();
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
                    platform->wake();
                    continue;
                }

                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                g_inbox.push(MsgStemArrived{
                    stem.label.empty() ? stem.track_class : stem.label,
                    stem.id, version->id, ordinal++, std::move(media), ms, hit});
                platform->wake();
            }
            g_inbox.push(MsgStatus{"", false});
            platform->wake();
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
                platform->wake();
                return;
            }

            const auto finished = api.follow(job->id, [&](const mx::net::JobRef& j) {
                std::string note = j.queue_position > 1
                    ? ("queued #" + std::to_string(j.queue_position))
                    : (j.message.empty() ? j.status : j.message);
                g_inbox.push(MsgStatus{note, false});
                platform->wake();
                return true;
            });
            if (finished.status != "done") {
                g_inbox.push(MsgStatus{
                    "repaint " + finished.status +
                        (finished.error.empty() ? "" : ": " + finished.error), true});
                platform->wake();
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
                platform->wake();
                return;
            }
            const mx::net::VersionRef* version = nullptr;
            for (const auto& st : detail->stems)
                if (st.id == stem) version = st.version();
            if (!version || version->audio.empty()) {
                g_inbox.push(MsgStatus{"rendered, but the stem has no current version", true});
                platform->wake();
                return;
            }

            const uint64_t ckey = mx::net::cache_key(project, version->id, version->audio);
            const auto cached = mx::net::cache_path(ckey);
            if (!std::filesystem::exists(cached) &&
                !api.fetch_audio(project, version->audio, cached.string())) {
                g_inbox.push(MsgStatus{"cannot fetch the new take: " + api.last_error(), true});
                platform->wake();
                return;
            }
            mx::Media media = mx::decode_file(cached, rate);
            if (!media.ok()) {
                std::error_code ec;
                std::filesystem::remove(cached, ec);
                g_inbox.push(MsgStatus{"new take will not decode: " + media.error, true});
                platform->wake();
                return;
            }
            g_inbox.push(MsgStemUpdated{track_id, source_id, version->id, std::move(media)});
            platform->wake();
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
        source.buffer = msg.media.buffer;
        source.peaks = std::move(msg.media.peaks);
        source.source_rate = msg.media.source_rate;
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
                    net_status = "loaded " + std::to_string(stems_arrived) + "/" +
                                 std::to_string(std::max(stems_expected, stems_arrived));
                    changed = true;

                } else if constexpr (std::is_same_v<T, MsgStemUpdated>) {
                    // Swap the audio under the existing clip. The clip, its
                    // track and its position are untouched -- only the source
                    // buffer changes, so a regenerate cannot quietly move
                    // anything on the timeline.
                    if (mx::Source* src = session.find_source(m.source_id)) {
                        src->buffer = m.media.buffer;
                        src->peaks = std::move(m.media.peaks);
                        src->version_id = m.version_id;
                        for (auto& clip : session.clips)
                            if (clip.source_id == src->id)
                                clip.length = std::min(clip.length, src->buffer->frames);
                    }
                    net_busy = false;
                    net_status = "regenerated";
                    if (selftest) regen_done = true;
                    changed = true;

                } else if constexpr (std::is_same_v<T, MsgStatus>) {
                    net_busy = false;
                    if (!m.text.empty()) {
                        net_status = m.text;
                        net_error = m.error;
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
        std::printf(
            "STUDIO device=%s rate=%u latency_ms=%.1f | sources=%d pending=%d "
            "decode_ms=%.0f | tracks=%zu clips=%zu | playhead=%.2fs xruns=%u "
            "| net=%s stems=%d/%d cached=%d | paint_ms=%.2f (build %.2f skia %.2f) worst=%.2f frames=%d columns=%lld\n",
            g_device.running() ? g_device.backend().c_str() : "NONE",
            g_device.rate(), 1000.0 * g_device.latency_frames() /
                std::max(1u, g_device.rate()),
            loaded_sources, g_pending.load(), last_decode_ms,
            session.tracks.size(), session.clips.size(),
            static_cast<double>(playhead()) / std::max(1u, session.rate),
            g_mixer.xruns(), net_status.c_str(), stems_arrived, stems_expected,
            cache_hits, last_paint_ms, build_ms, submit_ms, worst_paint_ms, frames,
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
            .child(readout("selection", selection.active()
                ? std::format("{:.2f}-{:.2f}s",
                              static_cast<double>(selection.begin()) / session.rate,
                              static_cast<double>(selection.end()) / session.rate)
                : std::string("shift-drag")))
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
            .child(readout("paint", std::format("{:.2f} ms", last_paint_ms)))
            .child(vik::div().flex_row().gap_2()
                .child(icon_button("regen", "sparkle", false,
                    [](Studio& s, const vik::ClickEvent&, vik::Window&,
                       vik::Context<Studio>& c) { s.regenerate(c.app()); c.notify(); }))
                .child(icon_button("export", "download-simple", false,
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
                        s.selected = 0;
                        s.selection.clear();
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
                    if (s.selecting) {
                        s.selection.to = std::max<int64_t>(0, s.view.frame_at(e.position.x));
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

    // --- pod sidebar: real elements, because it is chrome ------------------
    auto row = [](std::string label, std::string value, uint32_t tint) {
        return vik::div().flex_col().gap_1().px_3().py_2()
            .child(vik::text(std::move(label)).text_xs().text_color(vik::rgb(0x6c7383)))
            .child(vik::text(std::move(value)).text_color(vik::rgb(tint)));
    };

    auto sidebar = vik::div().flex_col().w_px(230.0f)
        .bg(vik::rgb(0x1b1e27)).border_1().border_color(vik::rgb(0x2c313d))
        .child(vik::div().flex_row().items_center().gap_2().px_3().py_2()
            .child(vik::ui::phosphor("cloud", vik::ui::PhWeight::Fill)
                       .size(14.0f).color(vik::rgb(pod_url.empty() ? 0x6c7383
                                                                   : 0x56c08a)))
            .child(vik::text("Pod").text_color(vik::rgb(0xe6e8ec))))
        .child(row("status", net_status,
                   net_error ? 0xe05c72 : (net_busy ? 0xd9903c : 0x8d94a3)));

    if (!pod_url.empty())
        sidebar = std::move(sidebar).child(row("url", pod_url, 0x6c7383));

    sidebar = std::move(sidebar).child(
        vik::div().px_3().py_2()
            .child(icon_button("connect", net_busy ? "hourglass" : "plugs-connected",
                               false,
                [](Studio& s, const vik::ClickEvent&, vik::Window&,
                   vik::Context<Studio>& c) {
                    s.connect(c.app());
                    c.notify();
                })));

    if (!projects.empty()) {
        auto list = vik::div().flex_col().flex_1().overflow_hidden()
            .child(vik::text("  Projects").text_xs().text_color(vik::rgb(0x6c7383)));
        // Enough to choose from without turning the sidebar into a file
        // browser; a proper browser with search is Phase 2's polish, not its
        // gate.
        const size_t shown = std::min<size_t>(projects.size(), 14);
        for (size_t i = 0; i < shown; ++i) {
            const auto& proj = projects[i];
            const bool active = proj.id == open_project_id;
            list = std::move(list).child(
                vik::div().id("proj").px_3().py_1().cursor_pointer()
                    .bg(vik::rgb(active ? 0x2c3446 : 0x1b1e27))
                    .hover([](vik::StyleRefinement& st) { st.bg(vik::rgb(0x242936)); })
                    .on_click(cx.listener([id = proj.id](Studio& s,
                                                         const vik::ClickEvent&,
                                                         vik::Window&,
                                                         vik::Context<Studio>& c) {
                        s.open_project(id, c.app());
                        c.notify();
                    }))
                    .child(vik::text(proj.title.substr(0, 26))
                               .text_color(vik::rgb(active ? 0xffd58a : 0xc9cedb))));
        }
        sidebar = std::move(sidebar).child(std::move(list));
    }

    return vik::div().size_full().flex_col().bg(vik::rgb(0x14161d))
        .on_key_down(cx.listener(
            [](Studio& s, const vik::KeyDownEvent& e, vik::Window&,
               vik::Context<Studio>& c) {
                if (e.key == "space") s.toggle_play();
                else if (e.key == "r") s.regenerate(c.app());
                else if (e.key == "e") s.export_mix();
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
                   .child(std::move(timeline)))
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

int main(int argc, char** argv) {
    fs::path folder;
    fs::path render_to;
    fs::path bounce_to;
    std::string pod_url, pod_token, pod_project;
    bool probe_only = false;
    double regen_from = -1.0, regen_to = -1.0;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--selftest") selftest = true;
        else if (arg == "--render" && i + 1 < argc) render_to = argv[++i];
        else if (arg == "--bounce" && i + 1 < argc) bounce_to = argv[++i];
        else if (arg == "--connect" && i + 1 < argc) pod_url = argv[++i];
        else if (arg == "--probe") probe_only = true;
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
    if (!bounce_to.empty()) return bounce_offline(folder, bounce_to);
    if (!pod_url.empty() && probe_only)
        return connect_probe(pod_url, pod_token, pod_project);

    if (!g_device.start(g_mixer, 48000))
        std::printf("audio: %s\n", g_device.error().c_str());

    vik::App::run([&](vik::App& app) {
        app.open_window(
            vik::WindowOptions{.title = "musicX Studio", .size = {1280.0f, 800.0f}},
            [&](vik::Window&, vik::App& app) {
                auto handle = app.add_entity<Studio>();
                app.update_entity(handle, [&](Studio& s, vik::Context<Studio>& c) {
                    s.selftest = selftest;
                    s.pod_url = pod_url;
                    s.pod_token = pod_token;
                    s.regen_from = regen_from;
                    s.regen_to = regen_to;
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

    g_device.stop();
    return 0;
}
