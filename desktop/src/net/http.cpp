#include "http.h"

#include <chrono>
#include <cstdio>
#include <filesystem>

#include <curl/curl.h>

namespace mx::net {
namespace {

size_t to_string(char* data, size_t size, size_t count, void* user) {
    auto* out = static_cast<std::string*>(user);
    out->append(data, size * count);
    return size * count;
}

size_t to_file(char* data, size_t size, size_t count, void* user) {
    auto* file = static_cast<std::FILE*>(user);
    return std::fwrite(data, size, count, file) * size;
}

struct ProgressState {
    ProgressFn fn;
    bool       cancelled = false;
};

int on_progress(void* user, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
    auto* state = static_cast<ProgressState*>(user);
    if (!state->fn) return 0;
    if (!state->fn(static_cast<int64_t>(now), static_cast<int64_t>(total))) {
        state->cancelled = true;
        return 1;                       // non-zero aborts the transfer
    }
    return 0;
}

struct EventState {
    const std::function<bool(const std::string&)>* on_event;
    std::string buffer;
    bool        stop = false;
};

/// SSE framing: events are separated by a blank line, and payload lines start
/// with "data:". Anything else in the frame -- comments, retry hints, the
/// heartbeats this server sends -- is skipped rather than delivered as data.
size_t to_events(char* data, size_t size, size_t count, void* user) {
    auto* state = static_cast<EventState*>(user);
    const size_t bytes = size * count;
    if (state->stop) return 0;
    state->buffer.append(data, bytes);

    for (;;) {
        const size_t split = state->buffer.find("\n\n");
        if (split == std::string::npos) break;
        const std::string frame = state->buffer.substr(0, split);
        state->buffer.erase(0, split + 2);

        std::string payload;
        size_t line_start = 0;
        while (line_start <= frame.size()) {
            size_t line_end = frame.find('\n', line_start);
            if (line_end == std::string::npos) line_end = frame.size();
            const std::string line = frame.substr(line_start, line_end - line_start);
            if (line.rfind("data:", 0) == 0) {
                size_t from = 5;
                if (from < line.size() && line[from] == ' ') ++from;
                if (!payload.empty()) payload += "\n";
                payload += line.substr(from);
            }
            if (line_end == frame.size()) break;
            line_start = line_end + 1;
        }

        if (!payload.empty() && !(*state->on_event)(payload)) {
            state->stop = true;
            return 0;                   // short read closes the connection
        }
    }
    return bytes;
}

}  // namespace

struct Http::Impl {
    CURL* curl = nullptr;
};

Http::Http() : impl_(new Impl) {
    static const bool global_init = [] {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        return true;
    }();
    (void)global_init;
    impl_->curl = curl_easy_init();
}

Http::~Http() {
    if (impl_->curl) curl_easy_cleanup(impl_->curl);
    delete impl_;
}

std::string url_escape(const std::string& text) {
    CURL* curl = curl_easy_init();
    char* encoded = curl_easy_escape(curl, text.c_str(), static_cast<int>(text.size()));
    std::string out = encoded ? encoded : text;
    if (encoded) curl_free(encoded);
    curl_easy_cleanup(curl);
    return out;
}

namespace {

curl_slist* build_headers(const std::string& token, bool json,
                          const std::string& idempotency_key,
                          const std::string& accept = {}) {
    curl_slist* headers = nullptr;
    if (!token.empty())
        headers = curl_slist_append(headers, ("X-API-Token: " + token).c_str());
    if (json)
        headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!idempotency_key.empty())
        headers = curl_slist_append(
            headers, ("Idempotency-Key: " + idempotency_key).c_str());
    if (!accept.empty())
        headers = curl_slist_append(headers, ("Accept: " + accept).c_str());
    return headers;
}

void apply_common(CURL* curl, const std::string& url, long timeout_s) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_s);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    // The proxy can idle. Give up on a connection that has moved fewer than
    // 100 bytes for 60s rather than hanging on it forever.
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 100L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
}

}  // namespace

Response Http::get(const std::string& url) {
    Response out;
    if (!impl_->curl) { out.error = "curl unavailable"; return out; }
    const auto started = std::chrono::steady_clock::now();

    curl_easy_reset(impl_->curl);
    apply_common(impl_->curl, url, timeout_s_);
    curl_slist* headers = build_headers(token_, false, {});
    curl_easy_setopt(impl_->curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEFUNCTION, to_string);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEDATA, &out.body);

    const CURLcode code = curl_easy_perform(impl_->curl);
    if (code != CURLE_OK) out.error = curl_easy_strerror(code);
    curl_easy_getinfo(impl_->curl, CURLINFO_RESPONSE_CODE, &out.status);
    curl_slist_free_all(headers);

    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

Response Http::post(const std::string& url, const std::string& json_body,
                    const std::string& idempotency_key) {
    Response out;
    if (!impl_->curl) { out.error = "curl unavailable"; return out; }
    const auto started = std::chrono::steady_clock::now();

    curl_easy_reset(impl_->curl);
    apply_common(impl_->curl, url, timeout_s_);
    curl_slist* headers = build_headers(token_, true, idempotency_key);
    curl_easy_setopt(impl_->curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(impl_->curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(impl_->curl, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(json_body.size()));
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEFUNCTION, to_string);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEDATA, &out.body);

    const CURLcode code = curl_easy_perform(impl_->curl);
    if (code != CURLE_OK) out.error = curl_easy_strerror(code);
    curl_easy_getinfo(impl_->curl, CURLINFO_RESPONSE_CODE, &out.status);
    curl_slist_free_all(headers);

    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

Response Http::post_file(
    const std::string& url, const std::string& file_path,
    const std::vector<std::pair<std::string, std::string>>& fields,
    const std::string& idempotency_key) {
    Response out;
    if (!impl_->curl) { out.error = "curl unavailable"; return out; }
    const auto started = std::chrono::steady_clock::now();

    curl_easy_reset(impl_->curl);
    apply_common(impl_->curl, url, timeout_s_);
    // json=false: curl_mime sets its own multipart Content-Type with the
    // boundary, and a Content-Type we add by hand overrides it with one that
    // names no boundary -- which the server rejects as a malformed body.
    curl_slist* headers = build_headers(token_, false, idempotency_key);
    curl_easy_setopt(impl_->curl, CURLOPT_HTTPHEADER, headers);

    curl_mime* mime = curl_mime_init(impl_->curl);
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filedata(part, file_path.c_str());   // streamed, not buffered
    for (const auto& [name, value] : fields) {
        curl_mimepart* f = curl_mime_addpart(mime);
        curl_mime_name(f, name.c_str());
        curl_mime_data(f, value.c_str(), CURL_ZERO_TERMINATED);
    }
    curl_easy_setopt(impl_->curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEFUNCTION, to_string);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEDATA, &out.body);

    const CURLcode code = curl_easy_perform(impl_->curl);
    if (code != CURLE_OK) out.error = curl_easy_strerror(code);
    curl_easy_getinfo(impl_->curl, CURLINFO_RESPONSE_CODE, &out.status);
    curl_mime_free(mime);
    curl_slist_free_all(headers);

    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

Response Http::download(const std::string& url, const std::string& dest_path,
                        ProgressFn progress) {
    Response out;
    if (!impl_->curl) { out.error = "curl unavailable"; return out; }
    const auto started = std::chrono::steady_clock::now();

    // Write to a temporary name and rename on success. A cache keyed by content
    // hash must never contain a truncated file -- it would be indistinguishable
    // from a complete one forever after.
    const std::string temp_path = dest_path + ".part";
    std::filesystem::create_directories(
        std::filesystem::path(dest_path).parent_path());

    std::FILE* file = std::fopen(temp_path.c_str(), "wb");
    if (!file) { out.error = "cannot write " + temp_path; return out; }

    ProgressState state{std::move(progress), false};

    curl_easy_reset(impl_->curl);
    apply_common(impl_->curl, url, timeout_s_ * 10);   // large files, long time
    curl_slist* headers = build_headers(token_, false, {});
    curl_easy_setopt(impl_->curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEFUNCTION, to_file);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEDATA, file);
    curl_easy_setopt(impl_->curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(impl_->curl, CURLOPT_XFERINFOFUNCTION, on_progress);
    curl_easy_setopt(impl_->curl, CURLOPT_XFERINFODATA, &state);

    const CURLcode code = curl_easy_perform(impl_->curl);
    if (code != CURLE_OK)
        out.error = state.cancelled ? "cancelled" : curl_easy_strerror(code);
    curl_easy_getinfo(impl_->curl, CURLINFO_RESPONSE_CODE, &out.status);
    curl_slist_free_all(headers);
    std::fclose(file);

    std::error_code ec;
    if (out.ok()) {
        std::filesystem::rename(temp_path, dest_path, ec);
        if (ec) out.error = "cannot move the download into place: " + ec.message();
    } else {
        std::filesystem::remove(temp_path, ec);
    }

    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

Response Http::stream_events(const std::string& url,
                             const std::function<bool(const std::string&)>& on_event) {
    Response out;
    if (!impl_->curl) { out.error = "curl unavailable"; return out; }
    const auto started = std::chrono::steady_clock::now();

    EventState state{&on_event, {}, false};

    curl_easy_reset(impl_->curl);
    apply_common(impl_->curl, url, 0L);          // no overall timeout: it streams
    // A job can be quiet for a while; the server heartbeats every 15s, so a
    // full minute of true silence means the stream is dead, not slow.
    curl_easy_setopt(impl_->curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(impl_->curl, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_slist* headers = build_headers(token_, false, {}, "text/event-stream");
    curl_easy_setopt(impl_->curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEFUNCTION, to_events);
    curl_easy_setopt(impl_->curl, CURLOPT_WRITEDATA, &state);

    const CURLcode code = curl_easy_perform(impl_->curl);
    // A deliberate short read shows up as a write error; that is us closing the
    // stream, not a failure.
    if (code != CURLE_OK && !(state.stop && code == CURLE_WRITE_ERROR))
        out.error = curl_easy_strerror(code);
    curl_easy_getinfo(impl_->curl, CURLINFO_RESPONSE_CODE, &out.status);
    curl_slist_free_all(headers);

    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

}  // namespace mx::net
