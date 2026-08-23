#include "sidecar.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <filesystem>
#include <fstream>

namespace mx::ai {
namespace {

namespace fs = std::filesystem;

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0,
                                      nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

/// node.exe: MUSICX_NODE override, else PATH.
std::wstring find_node(std::string* err) {
    wchar_t envbuf[MAX_PATH];
    if (GetEnvironmentVariableW(L"MUSICX_NODE", envbuf, MAX_PATH) > 0 && fs::exists(envbuf))
        return envbuf;
    wchar_t found[MAX_PATH];
    if (SearchPathW(nullptr, L"node.exe", nullptr, MAX_PATH, found, nullptr) > 0)
        return found;
    if (err)
        *err = "Node.js not found. Install Node 18+ (nodejs.org), or set MUSICX_NODE "
               "to a node.exe.";
    return {};
}

/// desktop/ai/sidecar: MUSICX_SIDECAR override, else walk up from the exe
/// (build\Release -> desktop), else the working directory.
fs::path find_sidecar(std::string* err) {
    wchar_t envbuf[2048];
    if (GetEnvironmentVariableW(L"MUSICX_SIDECAR", envbuf, 2048) > 0) {
        fs::path p = envbuf;
        if (fs::exists(p / "sidecar.mjs")) return p;
    }
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    fs::path dir = fs::path(exe).parent_path();
    for (int up = 0; up < 5; ++up) {
        const fs::path cand = dir / "ai" / "sidecar";
        if (fs::exists(cand / "sidecar.mjs")) return cand;
        if (!dir.has_parent_path() || dir == dir.parent_path()) break;
        dir = dir.parent_path();
    }
    const fs::path cwd = fs::current_path() / "ai" / "sidecar";
    if (fs::exists(cwd / "sidecar.mjs")) return cwd;
    if (err)
        *err = "AI sidecar not found (desktop\\ai\\sidecar\\sidecar.mjs). "
               "Set MUSICX_SIDECAR to its folder.";
    return {};
}

void close_h(void*& h) {
    if (h) { CloseHandle(static_cast<HANDLE>(h)); h = nullptr; }
}

}  // namespace

std::unique_ptr<Sidecar> Sidecar::spawn() {
    auto s = std::make_unique<Sidecar>();

    std::string err;
    const std::wstring node = find_node(&err);
    if (node.empty()) { s->spawn_error = err; return s; }
    const fs::path dir = find_sidecar(&err);
    if (dir.empty()) { s->spawn_error = err; return s; }
    if (!fs::exists(dir / "node_modules")) {
        s->spawn_error = "AI sidecar dependencies missing -- run `npm install` in " +
                         dir.string();
        return s;
    }

    // Pipes: child ends inheritable, parent ends explicitly not. Leaving the
    // parent end inheritable keeps the pipe alive in the child and the reader
    // never sees EOF.
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE in_r = nullptr, in_w = nullptr;
    HANDLE out_r = nullptr, out_w = nullptr;
    if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0)) {
        s->spawn_error = "CreatePipe failed";
        return s;
    }
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    // stderr to a log file. Module-not-found and key errors surface there, and
    // without it a sidecar that dies on startup dies silently.
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    const std::wstring logw = std::wstring(tmp) + L"musicx-sidecar.log";
    s->log_path = narrow(logw);
    HANDLE errh = CreateFileW(logw.c_str(), GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (errh == INVALID_HANDLE_VALUE) errh = nullptr;

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = errh;

    std::wstring cmd = L"\"" + node + L"\" \"" + widen((dir / "sidecar.mjs").string()) + L"\"";
    PROCESS_INFORMATION pi{};
    // CREATE_SUSPENDED so the job assignment lands before node can spawn the
    // SDK's own child -- otherwise that grandchild escapes kill-on-close.
    const BOOL ok = CreateProcessW(node.c_str(), cmd.data(), nullptr, nullptr,
                                   TRUE /*inherit*/, CREATE_SUSPENDED | CREATE_NO_WINDOW,
                                   nullptr, widen(dir.string()).c_str(), &si, &pi);
    CloseHandle(in_r);
    CloseHandle(out_w);
    if (errh) CloseHandle(errh);
    if (!ok) {
        CloseHandle(in_w);
        CloseHandle(out_r);
        s->spawn_error = "CreateProcess(node) failed (" + std::to_string(GetLastError()) + ")";
        return s;
    }

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
        jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);

    s->process = pi.hProcess;
    s->job = job;
    s->child_stdin_w = in_w;
    s->child_stdout_r = out_r;

    // Reader thread: raw bytes -> lines -> json -> inbox. Nothing else.
    Sidecar* raw = s.get();
    s->reader = std::thread([raw] {
        std::string carry;
        std::vector<char> buf(64 * 1024);
        for (;;) {
            DWORD got = 0;
            if (!ReadFile(static_cast<HANDLE>(raw->child_stdout_r), buf.data(),
                          DWORD(buf.size()), &got, nullptr) ||
                got == 0)
                break;
            carry.append(buf.data(), got);
            size_t nl;
            while ((nl = carry.find('\n')) != std::string::npos) {
                std::string line = carry.substr(0, nl);
                carry.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                auto ev = nlohmann::json::parse(line, nullptr, false);
                if (ev.is_discarded()) continue;
                std::lock_guard<std::mutex> lk(raw->in_mtx);
                raw->inbox.push_back(std::move(ev));
            }
        }
        raw->exited.store(true);
    });

    return s;
}

void Sidecar::send(const nlohmann::json& msg) {
    if (!child_stdin_w || exited.load()) return;
    const std::string line = msg.dump() + "\n";
    std::lock_guard<std::mutex> lk(out_mtx);
    DWORD written = 0;
    WriteFile(static_cast<HANDLE>(child_stdin_w), line.data(), DWORD(line.size()),
              &written, nullptr);
}

std::vector<nlohmann::json> Sidecar::drain() {
    std::lock_guard<std::mutex> lk(in_mtx);
    std::vector<nlohmann::json> out(inbox.begin(), inbox.end());
    inbox.clear();
    return out;
}

std::string Sidecar::log_tail(int max_lines) const {
    std::ifstream f(log_path, std::ios::binary);
    if (!f) return {};
    std::deque<std::string> lines;
    std::string l;
    while (std::getline(f, l)) {
        lines.push_back(l);
        if (int(lines.size()) > max_lines) lines.pop_front();
    }
    std::string out;
    for (auto& s : lines) out += s + "\n";
    return out;
}

Sidecar::~Sidecar() {
    if (process) {
        // Closing stdin is the shutdown signal: the sidecar's readline close
        // handler exits on EOF.
        close_h(child_stdin_w);
        if (WaitForSingleObject(static_cast<HANDLE>(process), 700) != WAIT_OBJECT_0)
            TerminateProcess(static_cast<HANDLE>(process), 1);
    }
    if (reader.joinable()) reader.join();  // the pipe breaks once the child dies
    close_h(child_stdout_r);
    close_h(process);
    close_h(job);  // kill-on-close reaps the SDK's child too
}

bool parse_result(const nlohmann::json& event, Proposal* out) {
    if (!out || !event.is_object()) return false;
    if (event.value("type", std::string{}) != "result") return false;
    out->id = event.value("id", std::string{});
    out->title = event.value("title", std::string{});
    out->caption = event.value("caption", std::string{});
    out->lyrics = event.value("lyrics", std::string{});
    out->bpm = event.value("bpm", 0);
    out->key_scale = event.value("key_scale", std::string{});
    out->notes = event.value("notes", std::string{});
    return !out->caption.empty();
}

}  // namespace mx::ai
