// HTTP, blocking and thread-confined.
//
// libcurl with schannel rather than a header-only client: RunPod's proxy does
// chunked TLS with idle timeouts, and schannel uses the Windows certificate
// store, so nothing ships a CA bundle that can go stale.
//
// Every call here blocks. That is deliberate -- these run on worker threads,
// and an async HTTP client would add a second concurrency model on top of the
// one the app already has. The UI thread must never call into this.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <vector>

namespace mx::net {

struct Response {
    long        status = 0;
    std::string body;
    std::string error;         // transport-level failure, empty if the request completed
    double      seconds = 0.0;

    bool ok() const { return error.empty() && status >= 200 && status < 300; }
    /// Worth retrying: a transport failure, or a server-side/rate-limit status.
    bool retryable() const {
        return !error.empty() || status == 429 || status == 408 || status >= 500;
    }
};

/// Progress during a download: (bytes so far, total or 0 when unknown).
/// Returning false cancels the transfer.
using ProgressFn = std::function<bool(int64_t, int64_t)>;

class Http {
public:
    Http();
    ~Http();
    Http(const Http&) = delete;
    Http& operator=(const Http&) = delete;

    void set_token(std::string token) { token_ = std::move(token); }
    void set_timeout(long seconds) { timeout_s_ = seconds; }

    Response get(const std::string& url);
    Response post(const std::string& url, const std::string& json_body,
                  const std::string& idempotency_key = {});

    /// Multipart POST: one file plus form fields.
    ///
    /// The upload endpoint takes a file, and the JSON post above cannot carry
    /// one. Streamed from disk by curl rather than read into memory first --
    /// a take is tens of megabytes and there is no reason for it to exist
    /// twice.
    Response post_file(const std::string& url, const std::string& file_path,
                       const std::vector<std::pair<std::string, std::string>>& fields,
                       const std::string& idempotency_key = {});

    /// Streams a URL to a file. The partial file is removed on failure, so a
    /// truncated download is never mistaken for a cached one.
    Response download(const std::string& url, const std::string& dest_path,
                      ProgressFn progress = {});

    /// Server-sent events. `on_event` receives each `data:` payload; returning
    /// false closes the stream. Used for job progress, with polling as the
    /// fallback when the stream goes quiet.
    Response stream_events(const std::string& url,
                           const std::function<bool(const std::string&)>& on_event);

private:
    struct Impl;
    Impl*       impl_;
    std::string token_;
    long        timeout_s_ = 120;
};

/// Percent-encodes one path segment. Stem and project ids are server-supplied
/// strings, and building a URL by concatenation is how a bad id becomes a
/// request for something else entirely.
std::string url_escape(const std::string& text);

}  // namespace mx::net
