// Application settings, stored per user.
//
// The pod belongs here, not in the main window. It is a render service the app
// talks to -- like an output device -- not the place your work lives. Putting a
// server project list in the primary navigation made the server look like the
// document store, which is exactly what it is not.
//
// The token is encrypted with DPAPI rather than written as text. It grants full
// access to a machine that costs money to run, and %LOCALAPPDATA% is readable
// by anything running as the user, backed up, and synced. DPAPI ties the
// ciphertext to this Windows account, so a copied settings file is inert.

#pragma once

#include <filesystem>
#include <string>

namespace mx {

struct Settings {
    std::string pod_url;
    std::string pod_token;
    bool        auto_connect = false;
    std::string last_document;      // reopened on launch when present

    /// Where settings live: %LOCALAPPDATA%/MusicMaker/settings.json
    static std::filesystem::path path();

    /// Missing or unreadable settings are not an error -- a first run has none.
    static Settings load();
    bool save() const;
};

/// DPAPI round-trip, base64 in the file. Returns empty on failure rather than
/// throwing: a token that cannot be decrypted (copied from another machine,
/// another user) should present as "not set", not as a crash.
std::string protect_secret(const std::string& plain);
std::string unprotect_secret(const std::string& encoded);

}  // namespace mx
