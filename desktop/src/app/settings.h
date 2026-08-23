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

/// Where renders happen.
///
/// The client only ever talks to a URL, so this is a URL choice and not a
/// second code path. Keeping both URLs rather than one editable field means
/// switching modes does not make you retype the pod address, and the token
/// follows the cloud URL it belongs to.
enum class Mode { Local, Cloud };

struct Settings {
    Mode        mode = Mode::Local;
    std::string local_url = "http://127.0.0.1:8000";
    std::string pod_url;
    std::string pod_token;
    /// Anthropic key for the AI helper. Optional: the Agent SDK will use an
    /// existing Claude Code login on this machine when there is one, which is
    /// why this is not a precondition for the feature.
    std::string ai_key;
    /// Which capture device to record from, by index. -1 is the system
    /// default, which on a laptop is routinely an array microphone the OS has
    /// muted -- so this being settable is what makes recording usable at all.
    int         input_device = -1;
    bool        auto_connect = false;
    std::string last_document;      // reopened on launch when present

    /// Where settings live: %LOCALAPPDATA%/MusicMaker/settings.json
    static std::filesystem::path path();

    /// The URL the current mode points at.
    const std::string& active_url() const {
        return mode == Mode::Cloud ? pod_url : local_url;
    }
    /// Local needs no token: it guards a machine that costs money to run, and
    /// this one is already yours.
    std::string active_token() const {
        return mode == Mode::Cloud ? pod_token : std::string{};
    }

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
