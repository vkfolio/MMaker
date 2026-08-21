#include "cache.h"

#include <cstdlib>
#include <sstream>

namespace mx::net {

std::filesystem::path cache_root() {
    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local)
        return std::filesystem::path(local) / "MusicMaker" / "cache";
    return std::filesystem::temp_directory_path() / "MusicMaker" / "cache";
}

uint64_t cache_key(const std::string& project_id, const std::string& version_id,
                   const std::string& audio_path) {
    // FNV-1a, 64-bit. The separators matter: without them "ab"+"c" and
    // "a"+"bc" would hash alike, which is a collision you would only find as a
    // project serving another project's audio.
    uint64_t hash = 1469598103934665603ull;
    const auto absorb = [&hash](const std::string& text) {
        for (unsigned char c : text) {
            hash ^= c;
            hash *= 1099511628211ull;
        }
        hash ^= 0x1F;                     // unit separator between fields
        hash *= 1099511628211ull;
    };
    absorb(project_id);
    absorb(version_id);
    absorb(audio_path);
    return hash;
}

std::filesystem::path cache_path(uint64_t key, const std::string& extension) {
    std::ostringstream name;
    name << std::hex << key << extension;
    return cache_root() / name.str();
}

}  // namespace mx::net
