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
#include <string>
#include <vector>

#include "http.h"

namespace mx::net {

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
                       const std::string& idempotency_key = {});

    /// Separate a rendered take into stems, so each part is editable.
    std::optional<JobRef> split(const std::string& project_id,
                                const std::string& variation_id,
                                const std::string& tier = "basic");
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
