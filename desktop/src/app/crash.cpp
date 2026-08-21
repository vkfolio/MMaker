#include "crash.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <ctime>

#include <windows.h>
#include <dbghelp.h>

namespace mx {
namespace {

const char* exception_name(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:      return "ACCESS_VIOLATION";
        case EXCEPTION_STACK_OVERFLOW:        return "STACK_OVERFLOW";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:    return "INT_DIVIDE_BY_ZERO";
        case EXCEPTION_ILLEGAL_INSTRUCTION:   return "ILLEGAL_INSTRUCTION";
        case EXCEPTION_IN_PAGE_ERROR:         return "IN_PAGE_ERROR";
        case EXCEPTION_PRIV_INSTRUCTION:      return "PRIV_INSTRUCTION";
        case 0xC000041D:                      return "FATAL_USER_CALLBACK_EXCEPTION";
        case 0xE06D7363:                      return "C++ exception (unhandled)";
        default:                              return "unknown";
    }
}

void report(std::FILE* out, EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    std::fprintf(out, "\n=== musicx crashed ===\n");
    std::fprintf(out, "exception : 0x%08lX  %s\n", code, exception_name(code));
    std::fprintf(out, "address   : %p\n", info->ExceptionRecord->ExceptionAddress);
    if (code == EXCEPTION_ACCESS_VIOLATION &&
        info->ExceptionRecord->NumberParameters >= 2) {
        std::fprintf(out, "operation : %s at %p\n",
                     info->ExceptionRecord->ExceptionInformation[0] ? "write" : "read",
                     reinterpret_cast<void*>(
                         info->ExceptionRecord->ExceptionInformation[1]));
    }

    // Symbol lookup is best-effort: without a .pdb this still prints addresses,
    // which is enough to tell one crash site from another.
    HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    SymInitialize(process, nullptr, TRUE);

    void* frames[62];
    const USHORT captured = CaptureStackBackTrace(0, 62, frames, nullptr);
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256];
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = 255;

    std::fprintf(out, "stack (%u frames):\n", captured);
    for (USHORT i = 0; i < captured; ++i) {
        DWORD64 displacement = 0;
        if (SymFromAddr(process, reinterpret_cast<DWORD64>(frames[i]),
                        &displacement, symbol)) {
            IMAGEHLP_LINE64 line{};
            line.SizeOfStruct = sizeof(line);
            DWORD col = 0;
            if (SymGetLineFromAddr64(process, reinterpret_cast<DWORD64>(frames[i]),
                                     &col, &line))
                std::fprintf(out, "  %2u  %s  (%s:%lu)\n", i, symbol->Name,
                             line.FileName, line.LineNumber);
            else
                std::fprintf(out, "  %2u  %s\n", i, symbol->Name);
        } else {
            std::fprintf(out, "  %2u  %p\n", i, frames[i]);
        }
    }
    std::fflush(out);
}

LONG WINAPI on_crash(EXCEPTION_POINTERS* info) {
    report(stderr, info);

    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local) {
        std::string path = std::string(local) + "\MusicMaker\crash.log";
        if (std::FILE* file = std::fopen(path.c_str(), "a")) {
            const std::time_t now = std::time(nullptr);
            std::fprintf(file, "\n--- %s", std::ctime(&now));
            report(file, info);
            std::fclose(file);
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;   // die, but having said something
}

}  // namespace

void install_crash_handler() {
    SetUnhandledExceptionFilter(&on_crash);
    // A stack overflow needs room to run the handler in; without this the
    // guard page is hit again inside the handler and nothing is printed.
    ULONG guarantee = 64 * 1024;
    SetThreadStackGuarantee(&guarantee);
}

}  // namespace mx
