// The local audio cache.
//
// Keyed by identity, not by filename: a stem's audio path repeats across
// projects ("stems/split_x/vocals.wav"), so a path-keyed cache would serve one
// project's vocals to another. The key is a hash of project, version and path
// together.
//
// Version ids are the identity that matters. Versions are append-only on the
// server -- a repaint adds one rather than replacing audio in place -- so a
// version id names a fixed piece of audio for good. That property is only true
// because the destructive-split bug was fixed first; before that, ids could be
// reused for different audio, and this cache would have been quietly wrong.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace mx::net {

/// Where cached audio lives: %LOCALAPPDATA%/MusicMaker/cache, falling back to
/// the temp directory when the environment has no such thing.
std::filesystem::path cache_root();

/// A stable 64-bit key. FNV-1a rather than std::hash, which is not required to
/// be stable across runs -- and a cache key that changes when the process
/// restarts is not a cache key.
uint64_t cache_key(const std::string& project_id, const std::string& version_id,
                   const std::string& audio_path);

/// Full path for a cached item. The file may or may not exist.
std::filesystem::path cache_path(uint64_t key, const std::string& extension = ".wav");

}  // namespace mx::net
