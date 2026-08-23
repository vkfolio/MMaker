// Sidecar -- the AI helper subprocess (node desktop/ai/sidecar/sidecar.mjs).
//
// Owns the hidden node process, its stdio pipes, and a reader thread that
// parses the sidecar's ndjson events into a mutex-guarded inbox. The UI thread
// drains that inbox on the same tick it already uses for worker results, and
// writes requests back with send().
//
// Threading contract, the same one the decode and network workers hold: the
// reader thread touches ONLY inbox and exited, under in_mtx. Everything else
// happens on the UI thread. It never sees a Session, a Handle or an entity.
//
// A Win32 Job object with kill-on-close guarantees the node tree dies with the
// app -- the Agent SDK spawns a child of its own, and orphaning it would leave
// a process holding an API session open after the window is gone.
#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace mx::ai {

/// One proposal from the model: a whole answer, never a patch.
struct Proposal {
    std::string id;
    std::string title;
    std::string caption;     // the ACE-Step prompt
    std::string lyrics;      // empty for an instrumental
    int         bpm = 0;
    std::string key_scale;
    std::string notes;       // one sentence on what it chose and why
};

class Sidecar {
public:
    // Win32 handles as void* so this header stays windows.h-free.
    void* process = nullptr;
    void* job = nullptr;
    void* child_stdin_w = nullptr;
    void* child_stdout_r = nullptr;

    std::thread reader;
    std::mutex  in_mtx;                    // guards inbox
    std::deque<nlohmann::json> inbox;      // sidecar -> app events
    std::mutex  out_mtx;                   // serialises WriteFile
    std::atomic<bool> exited{false};

    std::string spawn_error;   // non-empty => spawn failed, unusable
    std::string log_path;      // sidecar stderr (%TEMP%\musicx-sidecar.log)

    /// Spawns node + sidecar.mjs hidden. Always returns an object; check
    /// spawn_error before use, so a missing Node is a message rather than a
    /// crash or a silently dead feature.
    static std::unique_ptr<Sidecar> spawn();

    void send(const nlohmann::json& msg);            // UI thread
    std::vector<nlohmann::json> drain();             // UI thread; swaps inbox
    std::string log_tail(int max_lines = 10) const;  // last lines of stderr

    ~Sidecar();
};

/// Parse a `result` event. Returns false for anything else, so the caller can
/// hand every drained event here without pre-sorting them.
bool parse_result(const nlohmann::json& event, Proposal* out);

}  // namespace mx::ai
