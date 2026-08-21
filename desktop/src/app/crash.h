// Crash diagnostics.
//
// Three wrong guesses at one crash is what this exists to prevent. A fault with
// no output is a fault you debug by intuition; a fault that prints its exception
// code and the function names on the stack is one you fix.
//
// Installed for the whole process, writes to stderr and to a file beside the
// settings, and deliberately does as little as possible in the handler itself --
// a stack overflow leaves very little room to work in.

#pragma once

namespace mx {

/// Installs the handler. Safe to call once, early in main().
void install_crash_handler();

}  // namespace mx
