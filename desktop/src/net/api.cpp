#include "api.h"

#include <chrono>
#include <thread>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace mx::net {
namespace {

using json = nlohmann::json;

/// Reads a field without trusting it to be there or to be the right type.
/// Every one of these came back wrong at least once while this was built:
/// ids arrive namespaced, numbers arrive as strings, objects arrive as null.
template <typename T>
T field(const json& j, const char* key, T fallback) {
    if (!j.is_object()) return fallback;
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    try {
        return it->get<T>();
    } catch (const json::exception&) {
        return fallback;
    }
}

std::optional<json> parse(const Response& r, std::string& error) {
    if (!r.ok()) {
        if (!r.error.empty()) {
            error = r.error;
        } else {
            error = "HTTP " + std::to_string(r.status);
            // FastAPI puts the useful part in `detail`; surfacing "HTTP 401"
            // alone tells the user nothing they can act on.
            try {
                const json body = json::parse(r.body);
                const std::string detail = field<std::string>(body, "detail", "");
                if (!detail.empty()) error += ": " + detail;
            } catch (const json::exception&) {
                if (!r.body.empty()) error += ": " + r.body.substr(0, 160);
            }
        }
        return std::nullopt;
    }
    try {
        return json::parse(r.body);
    } catch (const json::exception& e) {
        error = std::string("malformed JSON: ") + e.what();
        return std::nullopt;
    }
}

VersionRef read_version(const json& j) {
    VersionRef v;
    v.id = field<std::string>(j, "id", "");
    v.audio = field<std::string>(j, "audio", "");
    v.op = field<std::string>(j, "op", "");
    v.note = field<std::string>(j, "note", "");
    return v;
}

StemRef read_stem(const json& j) {
    StemRef s;
    s.id = field<std::string>(j, "id", "");
    s.track_class = field<std::string>(j, "track_class", "");
    s.label = field<std::string>(j, "label", s.track_class);
    s.split_id = field<std::string>(j, "split_id", "");
    s.gain_db = field<float>(j, "gain_db", 0.0f);
    s.pan = field<float>(j, "pan", 0.0f);
    s.muted = field<bool>(j, "muted", false);
    s.soloed = field<bool>(j, "soloed", false);
    s.current = field<int>(j, "current", 0);
    if (auto it = j.find("versions"); it != j.end() && it->is_array())
        for (const auto& v : *it) s.versions.push_back(read_version(v));
    return s;
}

JobRef read_job(const json& j) {
    JobRef job;
    job.id = field<std::string>(j, "id", "");
    job.kind = field<std::string>(j, "kind", "");
    job.status = field<std::string>(j, "status", "");
    job.message = field<std::string>(j, "message", "");
    job.error = field<std::string>(j, "error", "");
    job.queue_position = field<int>(j, "queue_position", 0);
    return job;
}

}  // namespace

std::string tls_backend() {
    const curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    return (info && info->ssl_version) ? info->ssl_version : "none";
}

void ApiClient::set_base(std::string url) {
    while (!url.empty() && (url.back() == '/' || url.back() == ' ')) url.pop_back();
    if (!url.empty() && url.rfind("http", 0) != 0) url = "https://" + url;
    base_ = std::move(url);
}

Health ApiClient::health() {
    Health h;
    const Response r = http_.get(url("/health"));
    auto body = parse(r, last_error_);
    if (!body) { h.message = last_error_; return h; }

    h.reachable = true;
    h.status = field<std::string>(*body, "status", "");
    h.ready = (h.status == "ready");
    h.auth_required = field<bool>(*body, "auth_required", false);
    h.message = field<std::string>(*body, "message", "");
    if (auto it = body->find("gpu"); it != body->end() && it->is_object())
        h.gpu = field<std::string>(*it, "name", "");
    return h;
}

ApiClient::Generated ApiClient::generate(const std::string& prompt,
                                         const std::string& lyrics, int bars,
                                         const std::string& quality,
                                         bool instrumental, double bpm,
                                         const std::string& idempotency_key) {
    Generated out;
    json body;
    body["title"] = "scratch";
    body["prompt"] = prompt;
    body["bars"] = bars;
    body["variations"] = 1;          // one take; more is the pod doing work we discard
    body["quality"] = quality;
    body["instrumental"] = instrumental;
    if (bpm > 0) body["bpm"] = static_cast<int>(bpm);
    // An empty lyrics field means instrumental regardless of the flag, so it is
    // only sent when there is something to sing.
    if (!lyrics.empty()) body["lyrics"] = lyrics;

    const Response r = http_.post(url("/api/projects"), body.dump(), idempotency_key);
    auto parsed = parse(r, last_error_);
    if (!parsed) return out;

    if (auto p = parsed->find("project"); p != parsed->end() && p->is_object())
        out.project_id = field<std::string>(*p, "id", "");
    if (auto j = parsed->find("job"); j != parsed->end() && j->is_object())
        out.job = read_job(*j);
    out.ok = !out.project_id.empty() && !out.job.id.empty();
    if (!out.ok) last_error_ = "the pod accepted the request but named no project";
    return out;
}

std::optional<JobRef> ApiClient::split(const std::string& project_id,
                                       const std::string& variation_id,
                                       const std::string& tier,
                                       bool remove_reverb) {
    json body;
    body["variation_id"] = variation_id;
    body["tier"] = tier;
    body["remove_reverb"] = remove_reverb;
    const Response r = http_.post(
        url("/api/projects/" + url_escape(project_id) + "/split"), body.dump());
    auto parsed = parse(r, last_error_);
    if (!parsed) return std::nullopt;
    auto it = parsed->find("job");
    if (it == parsed->end() || !it->is_object()) {
        last_error_ = "split accepted but no job returned";
        return std::nullopt;
    }
    return read_job(*it);
}

namespace {

/// The tool calls are all the same request with a different path and body, so
/// they share one implementation rather than five near-copies that can drift.
std::optional<JobRef> post_job(Http& http, const std::string& url,
                               const json& body, const std::string& key,
                               std::string& error, const char* what) {
    const Response r = http.post(url, body.dump(), key);
    if (!r.ok()) {
        if (!r.error.empty()) {
            error = r.error;
        } else {
            error = "HTTP " + std::to_string(r.status);
            try {
                const json parsed = json::parse(r.body);
                if (parsed.contains("detail") && parsed["detail"].is_string())
                    error += ": " + parsed["detail"].get<std::string>();
            } catch (const json::exception&) {
            }
        }
        return std::nullopt;
    }
    try {
        const json parsed = json::parse(r.body);
        auto it = parsed.find("job");
        if (it == parsed.end() || !it->is_object()) {
            error = std::string(what) + " was accepted but named no job";
            return std::nullopt;
        }
        JobRef job;
        job.id = it->value("id", "");
        job.kind = it->value("kind", "");
        job.status = it->value("status", "");
        job.message = it->value("message", "");
        job.error = it->value("error", "");
        job.queue_position = it->value("queue_position", 0);
        return job;
    } catch (const json::exception& e) {
        error = std::string("malformed JSON: ") + e.what();
        return std::nullopt;
    }
}

}  // namespace

std::optional<JobRef> ApiClient::vary(const std::string& project_id,
                                      const std::string& stem_id,
                                      const std::string& idempotency_key) {
    return post_job(http_,
                    url("/api/projects/" + url_escape(project_id) + "/stems/" +
                        url_escape(stem_id) + "/vary"),
                    json::object(), idempotency_key, last_error_, "vary");
}

std::optional<JobRef> ApiClient::extend(const std::string& project_id,
                                        const std::string& stem_id, int bars,
                                        const std::string& idempotency_key) {
    json body;
    body["bars"] = bars;
    return post_job(http_,
                    url("/api/projects/" + url_escape(project_id) + "/stems/" +
                        url_escape(stem_id) + "/extend"),
                    body, idempotency_key, last_error_, "extend");
}

std::optional<JobRef> ApiClient::isolate(const std::string& project_id,
                                         const std::string& stem_id,
                                         const std::string& idempotency_key) {
    return post_job(http_,
                    url("/api/projects/" + url_escape(project_id) + "/stems/" +
                        url_escape(stem_id) + "/isolate"),
                    json::object(), idempotency_key, last_error_, "isolate");
}

std::optional<JobRef> ApiClient::change_voice(const std::string& project_id,
                                              const std::string& stem_id,
                                              const std::string& voice_id,
                                              double start_s, double end_s,
                                              const std::string& idempotency_key) {
    json body;
    body["voice_id"] = voice_id;
    if (start_s >= 0.0 && end_s > start_s) {
        body["start_s"] = start_s;
        body["end_s"] = end_s;
    }
    return post_job(http_,
                    url("/api/projects/" + url_escape(project_id) + "/stems/" +
                        url_escape(stem_id) + "/voice"),
                    body, idempotency_key, last_error_, "voice change");
}

bool ApiClient::set_version(const std::string& project_id,
                            const std::string& stem_id, int index) {
    const Response r = http_.post(
        url("/api/projects/" + url_escape(project_id) + "/stems/" +
            url_escape(stem_id) + "/version/" + std::to_string(index)),
        "{}");
    if (!r.ok()) {
        last_error_ = r.error.empty() ? ("HTTP " + std::to_string(r.status)) : r.error;
        return false;
    }
    return true;
}

std::vector<ApiClient::Voice> ApiClient::voices() {
    std::vector<Voice> out;
    const Response r = http_.get(url("/api/voices"));
    auto body = parse(r, last_error_);
    if (!body) return out;
    auto it = body->find("voices");
    if (it == body->end() || !it->is_array()) return out;
    for (const auto& v : *it) {
        Voice voice;
        voice.id = field<std::string>(v, "id", "");
        voice.label = field<std::string>(v, "label", voice.id);
        voice.description = field<std::string>(v, "description", "");
        if (!voice.id.empty()) out.push_back(std::move(voice));
    }
    return out;
}

Capabilities ApiClient::capabilities() {
    Capabilities caps;
    const Response r = http_.get(url("/api/engine"));
    auto body = parse(r, last_error_);
    if (!body) return caps;                 // an older engine has no such route

    caps.known = true;
    if (auto it = body->find("installed"); it != body->end() && it->is_array())
        for (const auto& name : *it)
            if (name.is_string()) caps.installed.push_back(name.get<std::string>());
    if (auto it = body->find("effective_tiers"); it != body->end() && it->is_object())
        for (auto item = it->begin(); item != it->end(); ++item)
            if (item.value().is_string())
                caps.effective.emplace_back(item.key(), item.value().get<std::string>());
    if (auto it = body->find("tiers_that_would_fall_back");
        it != body->end() && it->is_object())
        for (auto item = it->begin(); item != it->end(); ++item)
            caps.falls_back.push_back(item.key());
    return caps;
}

std::vector<ProjectSummary> ApiClient::projects() {
    std::vector<ProjectSummary> out;
    const Response r = http_.get(url("/api/projects"));
    auto body = parse(r, last_error_);
    if (!body) return out;

    auto it = body->find("projects");
    if (it == body->end() || !it->is_array()) return out;
    for (const auto& p : *it) {
        ProjectSummary s;
        s.id = field<std::string>(p, "id", "");
        s.title = field<std::string>(p, "title", "Untitled");
        if (auto g = p.find("grid"); g != p.end() && g->is_object()) {
            s.bpm = field<double>(*g, "bpm", 120.0);
            s.bars = field<int>(*g, "bars", 32);
            s.key_scale = field<std::string>(*g, "key_scale", "");
        }
        if (!s.id.empty()) out.push_back(std::move(s));
    }
    return out;
}

std::optional<ProjectDetail> ApiClient::project(const std::string& id) {
    const Response r = http_.get(url("/api/projects/" + url_escape(id)));
    auto body = parse(r, last_error_);
    if (!body) return std::nullopt;

    auto p = body->find("project");
    if (p == body->end() || !p->is_object()) {
        last_error_ = "the response had no project";
        return std::nullopt;
    }

    ProjectDetail d;
    d.summary.id = field<std::string>(*p, "id", id);
    d.summary.title = field<std::string>(*p, "title", "Untitled");
    if (auto g = p->find("grid"); g != p->end() && g->is_object()) {
        d.summary.bpm = field<double>(*g, "bpm", 120.0);
        d.summary.bars = field<int>(*g, "bars", 32);
        d.summary.key_scale = field<std::string>(*g, "key_scale", "");
    }
    if (auto v = p->find("variations"); v != p->end() && v->is_array()) {
        for (const auto& item : *v) {
            VariationRef ref;
            ref.id = field<std::string>(item, "id", "");
            ref.audio = field<std::string>(item, "audio", "");
            ref.seed = field<int64_t>(item, "seed", 0);
            d.variations.push_back(std::move(ref));
        }
    }
    if (auto s = p->find("stems"); s != p->end() && s->is_array())
        for (const auto& item : *s) d.stems.push_back(read_stem(item));
    return d;
}

std::string ApiClient::audio_url(const std::string& project_id,
                                 const std::string& rel) const {
    // The path is several segments ("stems/split_x/vocals.wav"), so the
    // separators must survive escaping while the segments themselves are
    // escaped. Escaping the whole string would turn the path into one segment.
    std::string encoded;
    std::string segment;
    for (char c : rel) {
        if (c == '/') {
            encoded += url_escape(segment) + "/";
            segment.clear();
        } else {
            segment += c;
        }
    }
    encoded += url_escape(segment);
    return base_ + "/api/projects/" + url_escape(project_id) + "/audio/" + encoded;
}

bool ApiClient::fetch_audio(const std::string& project_id, const std::string& rel,
                            const std::string& dest, ProgressFn progress) {
    const Response r =
        http_.download(audio_url(project_id, rel), dest, std::move(progress));
    if (!r.ok()) {
        last_error_ = r.error.empty() ? ("HTTP " + std::to_string(r.status)) : r.error;
        return false;
    }
    return true;
}

std::optional<JobRef> ApiClient::repaint(const std::string& project_id,
                                         const std::string& stem_id,
                                         double start_s, double end_s,
                                         const std::string& prompt,
                                         const std::string& idempotency_key) {
    json body;
    body["start_s"] = start_s;
    body["end_s"] = end_s;
    if (!prompt.empty()) body["prompt"] = prompt;

    // An idempotency key matters more here than anywhere else: a POST whose
    // response is lost has still started a GPU job, and without the key a retry
    // starts a second one while the first is still running.
    const Response r = http_.post(
        url("/api/projects/" + url_escape(project_id) + "/stems/" +
            url_escape(stem_id) + "/repaint"),
        body.dump(), idempotency_key);
    auto parsed = parse(r, last_error_);
    if (!parsed) return std::nullopt;

    auto it = parsed->find("job");
    if (it == parsed->end() || !it->is_object()) {
        last_error_ = "the server accepted the repaint but named no job";
        return std::nullopt;
    }
    return read_job(*it);
}

std::optional<JobRef> ApiClient::add_layer(const std::string& project_id,
                                           const std::string& track_class,
                                           const std::string& prompt,
                                           double start_s, double end_s,
                                           const std::string& idempotency_key) {
    json body;
    body["track_class"] = track_class;
    if (!prompt.empty()) body["prompt"] = prompt;
    // Omitted rather than sent as null: without a range the server layers over
    // the whole arrangement, which is the sensible default and the only thing
    // the API could express before ranges existed.
    if (start_s >= 0.0 && end_s > start_s) {
        body["start_s"] = start_s;
        body["end_s"] = end_s;
    }

    const Response r = http_.post(
        url("/api/projects/" + url_escape(project_id) + "/layers"),
        body.dump(), idempotency_key);
    auto parsed = parse(r, last_error_);
    if (!parsed) return std::nullopt;

    auto it = parsed->find("job");
    if (it == parsed->end() || !it->is_object()) {
        last_error_ = "the server accepted the layer but named no job";
        return std::nullopt;
    }
    return read_job(*it);
}

std::optional<JobRef> ApiClient::job(const std::string& job_id) {
    const Response r = http_.get(url("/api/jobs/" + url_escape(job_id)));
    auto body = parse(r, last_error_);
    if (!body) return std::nullopt;
    return read_job(*body);
}

bool ApiClient::cancel(const std::string& job_id) {
    const Response r = http_.post(url("/api/jobs/" + url_escape(job_id) + "/cancel"), "{}");
    if (!r.ok()) {
        last_error_ = r.error.empty() ? ("HTTP " + std::to_string(r.status)) : r.error;
        return false;
    }
    return true;
}

JobRef ApiClient::follow(const std::string& job_id,
                         const std::function<bool(const JobRef&)>& on_update) {
    JobRef last;
    last.id = job_id;
    bool finished = false;
    bool keep_going = true;
    int events = 0;

    const Response stream = http_.stream_events(
        url("/api/jobs/" + url_escape(job_id) + "/events"),
        [&](const std::string& payload) {
            json parsed;
            try {
                parsed = json::parse(payload);
            } catch (const json::exception&) {
                return true;                       // not ours; keep listening
            }
            // Heartbeats keep the connection open and carry no state.
            if (field<bool>(parsed, "heartbeat", false)) return true;
            ++events;
            last = read_job(parsed);
            if (last.id.empty()) last.id = job_id;
            keep_going = on_update(last);
            finished = (last.status == "done" || last.status == "error" ||
                        last.status == "cancelled");
            return keep_going && !finished;
        });

    if (finished || !keep_going) return last;

    // The stream told us nothing useful -- buffered by a proxy, dropped, or
    // never supported. Poll instead. This is a fallback, not a race: it only
    // starts once the stream has stopped.
    if (events == 0 && !stream.error.empty()) last.message = "streaming unavailable";
    while (keep_going) {
        auto polled = job(job_id);
        if (!polled) {
            last.status = "error";
            last.error = last_error_;
            on_update(last);
            return last;
        }
        last = *polled;
        keep_going = on_update(last);
        if (last.status == "done" || last.status == "error" ||
            last.status == "cancelled")
            return last;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    return last;
}

}  // namespace mx::net
