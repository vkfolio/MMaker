// The musicmaker API, as plain data.
//
// This is where the server stops being JSON and becomes types the app can hold.
// Everything is blocking and belongs on a worker.
//
// What is deliberately *not* here: any mirror of the server's Project. The
// server is a render service, not a document store -- mix state exists on both
// sides, its saves are whole-file rewrites, and a long job finishing can clobber
// an edit made while it ran. What the app keeps is provenance: enough to name a
// piece of audio, fetch it again, and know what produced it.

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <string>
#include <vector>

#include "http.h"

namespace mx::net {

/// What the engine on the other end can really render.
///
/// A tier whose checkpoint is absent does not fail -- ACE-Step falls back to
/// its primary and renders anyway. Asking, rather than assuming, is the only
/// way the app can avoid offering a quality it will not get; local mode makes
/// this routine rather than exceptional, since a laptop carries the small tier
/// and a pod may carry all three.
struct Capabilities {
    bool                     known = false;
    std::vector<std::string> installed;
    /// tier -> the checkpoint that tier will actually use here.
    std::vector<std::pair<std::string, std::string>> effective;
    /// Tiers that would quietly render as something else.
    std::vector<std::string> falls_back;
    /// ACE-Step task types any installed checkpoint implements -- "cover",
    /// "repaint", "lego", "complete", "extract". A turbo checkpoint carries
    /// only the first three of the seven, so on an ordinary laptop install
    /// Add a Layer and Extend are genuinely unavailable rather than slow.
    std::vector<std::string> tasks;

    bool tier_is_real(const std::string& tier) const {
        for (const auto& name : falls_back)
            if (name == tier) return false;
        return true;
    }

    /// Whether the engine can be asked to do this at all.
    ///
    /// Unknown capabilities answer yes: before the first probe the app must not
    /// grey out everything, and an older engine that does not report tasks was
    /// working before this field existed.
    bool can(const std::string& task) const {
        if (!known || tasks.empty()) return true;
        for (const auto& name : tasks)
            if (name == task) return true;
        return false;
    }
};

struct Health {
    bool        reachable = false;
    bool        auth_required = false;
    bool        ready = false;
    std::string status;
    std::string gpu;
    std::string message;
};

struct VersionRef {
    std::string id;
    std::string audio;          // path relative to the project's audio route
    std::string op;             // "import", "repaint", "layer", ...
    std::string note;
};

struct StemRef {
    std::string id;
    std::string track_class;
    std::string label;
    std::string split_id;
    float       gain_db = 0.0f;
    float       pan = 0.0f;
    bool        muted = false;
    bool        soloed = false;
    std::vector<VersionRef> versions;
    int         current = 0;

    const VersionRef* version() const {
        if (versions.empty()) return nullptr;
        const int i = (current >= 0 && current < static_cast<int>(versions.size()))
                          ? current : static_cast<int>(versions.size()) - 1;
        return &versions[i];
    }
};

struct VariationRef {
    std::string id;
    std::string audio;
    int64_t     seed = 0;
};

struct ProjectSummary {
    std::string id;
    std::string title;
    double      bpm = 120.0;
    int         bars = 32;
    std::string key_scale;
};

struct ProjectDetail {
    ProjectSummary            summary;
    std::vector<VariationRef> variations;
    std::vector<StemRef>      stems;
};

struct JobRef {
    std::string id;
    std::string kind;
    std::string status;         // queued | running | done | error | cancelled
    std::string message;
    std::string error;
    int         queue_position = 0;
};

/// A blocking client. One per worker thread -- it owns a curl handle, which is
/// not shareable across threads.
class ApiClient {
public:
    void set_base(std::string url);
    void set_token(std::string token) { http_.set_token(std::move(token)); }
    const std::string& base() const { return base_; }
    const std::string& last_error() const { return last_error_; }

    Health health();
    Capabilities capabilities();

    /// Ad-hoc generation. Creates a scratch workspace on the pod and renders
    /// into it.
    ///
    /// The pod needs somewhere to put the audio and something to hang stem ids
    /// off, so a server-side project is unavoidable -- but it is scratch, not a
    /// document. The user's project is the local .mmproj; this id is recorded as
    /// provenance so later tools know where the audio came from, and is never
    /// shown as something to browse.
    struct Generated {
        std::string project_id;
        JobRef      job;
        bool        ok = false;
    };
    Generated generate(const std::string& prompt, const std::string& lyrics,
                       int bars, const std::string& quality,
                       bool instrumental, double bpm,
                       const std::string& idempotency_key = {},
                       bool thinking = false);

    /// Separate a rendered take into stems, so each part is editable.
    ///
    /// Tiers match the reference dialog: basic (vocal + instrumental),
    /// professional (6 stems), advanced (everything detected).
    /// Upload local audio as a new scratch project on the engine.
    ///
    /// This is how a recorded take reaches the AI tools at all: they act on
    /// stems the engine holds, and a take recorded here is a file the engine
    /// has never seen. Returns the new project id, empty on failure.
    struct Uploaded {
        std::string project_id;
        int         bpm = 0;          // what the engine detected, 0 if it did not
        std::string key_scale;
        bool        ok = false;
    };
    Uploaded upload_audio(const std::string& file_path, const std::string& title,
                          const std::string& prompt,
                          const std::string& idempotency_key = {});

    /// Confirm the grid, which the upload deliberately leaves unconfirmed --
    /// a wrong key silently corrupts every stem generated against it.
    bool confirm_grid(const std::string& project_id, int bpm,
                      const std::string& key_scale, int bars);

    /// Generate takes against whatever the project holds. With uploaded source
    /// audio this is ACE-Step's `cover`: variations of your own material.
    std::optional<JobRef> variations(const std::string& project_id, int count = 1);

    /// One note on the wire, in seconds relative to its clip.
    ///
    /// Seconds rather than frames: a sample rate is a property of the audio
    /// device, not of the music, and every other time this client sends is
    /// converted at this edge for the same reason.
    struct ScoreNote {
        double  start = 0.0;
        double  length = 0.0;
        int     pitch = 60;
        int     velocity = 96;
        std::string lyric;
    };

    /// Play a score. With a prompt, the sampler render is then reimagined
    /// through the engine; without one you get the sampler alone, which is
    /// offline, immediate and predictable.
    std::optional<JobRef> render_score(const std::string& project_id,
                                       const std::vector<ScoreNote>& notes,
                                       double bpm, int program,
                                       const std::string& label,
                                       const std::string& prompt = {},
                                       const std::string& idempotency_key = {});

    /// A project to work in, with nothing rendered into it.
    std::string create_empty_project(const std::string& title);

    std::optional<JobRef> split(const std::string& project_id,
                                const std::string& variation_id,
                                const std::string& tier = "basic",
                                bool remove_reverb = false);

    /// The remaining per-stem tools. All take the same shape as repaint: a
    /// range in seconds where the server accepts one, and an idempotency key,
    /// because every one of them starts a GPU job that a lost response would
    /// otherwise duplicate.
    std::optional<JobRef> vary(const std::string& project_id,
                               const std::string& stem_id,
                               const std::string& idempotency_key = {});
    std::optional<JobRef> extend(const std::string& project_id,
                                 const std::string& stem_id, int bars,
                                 const std::string& idempotency_key = {});
    std::optional<JobRef> isolate(const std::string& project_id,
                                  const std::string& stem_id,
                                  const std::string& idempotency_key = {});
    std::optional<JobRef> change_voice(const std::string& project_id,
                                       const std::string& stem_id,
                                       const std::string& voice_id,
                                       double start_s = -1.0, double end_s = -1.0,
                                       const std::string& idempotency_key = {});

    /// Make an earlier version current again. Local A/B is instant because
    /// every version is already cached; this is what makes the choice stick.
    bool set_version(const std::string& project_id, const std::string& stem_id,
                     int index);

    struct Voice {
        std::string id;
        std::string label;
        std::string description;
    };
    std::vector<Voice> voices();
    std::vector<ProjectSummary> projects();
    std::optional<ProjectDetail> project(const std::string& id);

    /// Full URL for a project-relative audio path, ready to hand to download().
    std::string audio_url(const std::string& project_id, const std::string& rel) const;

    /// Streams audio to `dest`. Returns false and sets last_error on failure.
    bool fetch_audio(const std::string& project_id, const std::string& rel,
                     const std::string& dest, ProgressFn progress = {});

    /// Regenerate a time range of one stem, in seconds.
    ///
    /// Seconds, not bars: the server used to quantise to whole bars, which made
    /// "regenerate this bit" mean "regenerate the bar it lands in". A selection
    /// the user dragged has to survive the trip.
    std::optional<JobRef> repaint(const std::string& project_id,
                                  const std::string& stem_id,
                                  double start_s, double end_s,
                                  const std::string& prompt = {},
                                  const std::string& idempotency_key = {});

    /// Generate a new part against what the project already has.
    ///
    /// `track_class` is one of the server's fixed set (vocals, drums, bass,
    /// guitar, keyboard, strings, synth, fx, brass, woodwinds, percussion,
    /// backing_vocals) -- the model conditions on the class, so an arbitrary
    /// string is not a description, it is a rejected request.
    std::optional<JobRef> add_layer(const std::string& project_id,
                                    const std::string& track_class,
                                    const std::string& prompt = {},
                                    double start_s = -1.0, double end_s = -1.0,
                                    const std::string& idempotency_key = {});

    std::optional<JobRef> job(const std::string& job_id);
    bool cancel(const std::string& job_id);

    /// Follows a job to completion. `on_update` is called for each change and
    /// may return false to stop following (the job keeps running server-side).
    ///
    /// SSE first, then polling: the stream is cheaper and lower-latency, but a
    /// proxy that buffers or drops it must not strand the job, so a stream that
    /// produces nothing falls back rather than hanging.
    JobRef follow(const std::string& job_id,
                  const std::function<bool(const JobRef&)>& on_update);

    Http& http() { return http_; }

private:
    Http        http_;
    std::string base_;
    std::string last_error_;

    std::string url(const std::string& path) const { return base_ + path; }
};

/// The TLS backend curl actually chose, e.g. "Schannel". Worth surfacing: the
/// claim "we use the Windows certificate store" is otherwise unverifiable.
std::string tls_backend();

}  // namespace mx::net
