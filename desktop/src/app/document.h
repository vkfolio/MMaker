// .mmproj -- the local project document.
//
// This is the app's document store, and the only one. The pod renders audio; it
// does not hold your work. What lives here is the arrangement plus provenance
// for every piece of audio: which pod project, stem and version it came from,
// and where it sits in the local cache.
//
// Audio is never embedded. A document is small JSON and stays readable, and the
// samples live in the content cache where a second document referring to the
// same render costs nothing. The consequence, stated plainly rather than
// discovered later: a .mmproj alone is not portable to another machine. Moving
// a project means moving the cache with it, or re-fetching from the pod, which
// is exactly what the recorded provenance is for.

#pragma once

#include <filesystem>
#include <string>

#include "session.h"

namespace mx {

struct LoadResult {
    bool        ok = false;
    std::string error;
    int         missing_audio = 0;   // sources whose cached file was gone
};

/// Serialises the session. `schema` is written so a future reader can migrate
/// rather than guess.
bool save_document(const Session& session, const std::filesystem::path& path,
                   std::string* error = nullptr);

/// Reads a document into `session`, replacing whatever it held.
///
/// Audio is *not* decoded here: this returns the arrangement and the list of
/// what needs loading, so the decoding stays on a worker where it belongs.
LoadResult load_document(Session& session, const std::filesystem::path& path);

/// Where documents go by default: Documents/MusicMaker.
std::filesystem::path documents_root();

}  // namespace mx
