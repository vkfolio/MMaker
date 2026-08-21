#include "settings.h"

#include <cstdlib>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include <windows.h>
#include <wincrypt.h>

namespace mx {
namespace {

using json = nlohmann::json;

const char kAlphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const std::vector<unsigned char>& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const unsigned a = data[i];
        const unsigned b = (i + 1 < data.size()) ? data[i + 1] : 0;
        const unsigned c = (i + 2 < data.size()) ? data[i + 2] : 0;
        const unsigned triple = (a << 16) | (b << 8) | c;
        out += kAlphabet[(triple >> 18) & 0x3F];
        out += kAlphabet[(triple >> 12) & 0x3F];
        out += (i + 1 < data.size()) ? kAlphabet[(triple >> 6) & 0x3F] : '=';
        out += (i + 2 < data.size()) ? kAlphabet[triple & 0x3F] : '=';
    }
    return out;
}

std::vector<unsigned char> base64_decode(const std::string& text) {
    int table[256];
    for (int& value : table) value = -1;
    for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(kAlphabet[i])] = i;

    std::vector<unsigned char> out;
    int buffer = 0, bits = 0;
    for (unsigned char c : text) {
        if (c == '=' || table[c] < 0) continue;
        buffer = (buffer << 6) | table[c];
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

}  // namespace

std::string protect_secret(const std::string& plain) {
    if (plain.empty()) return {};
    DATA_BLOB in{static_cast<DWORD>(plain.size()),
                 reinterpret_cast<BYTE*>(const_cast<char*>(plain.data()))};
    DATA_BLOB out{};
    if (!CryptProtectData(&in, L"musicX pod token", nullptr, nullptr, nullptr,
                          CRYPTPROTECT_UI_FORBIDDEN, &out))
        return {};
    std::vector<unsigned char> bytes(out.pbData, out.pbData + out.cbData);
    LocalFree(out.pbData);
    return base64_encode(bytes);
}

std::string unprotect_secret(const std::string& encoded) {
    if (encoded.empty()) return {};
    auto bytes = base64_decode(encoded);
    if (bytes.empty()) return {};
    DATA_BLOB in{static_cast<DWORD>(bytes.size()), bytes.data()};
    DATA_BLOB out{};
    if (!CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr,
                            CRYPTPROTECT_UI_FORBIDDEN, &out))
        return {};                       // wrong user or machine: treat as unset
    std::string plain(reinterpret_cast<char*>(out.pbData), out.cbData);
    LocalFree(out.pbData);
    return plain;
}

std::filesystem::path Settings::path() {
    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local)
        return std::filesystem::path(local) / "MusicMaker" / "settings.json";
    return std::filesystem::temp_directory_path() / "MusicMaker" / "settings.json";
}

Settings Settings::load() {
    Settings s;
    std::ifstream file(path());
    if (!file) return s;                 // a first run has no settings
    try {
        json j;
        file >> j;
        s.pod_url = j.value("pod_url", "");
        s.auto_connect = j.value("auto_connect", false);
        s.last_document = j.value("last_document", "");
        s.pod_token = unprotect_secret(j.value("pod_token_dpapi", ""));
    } catch (const json::exception&) {
        return Settings{};               // corrupt settings are not fatal
    }
    return s;
}

bool Settings::save() const {
    std::error_code ec;
    std::filesystem::create_directories(path().parent_path(), ec);
    json j;
    j["pod_url"] = pod_url;
    j["auto_connect"] = auto_connect;
    j["last_document"] = last_document;
    // Named for what it is, so nobody mistakes the value for something they can
    // paste elsewhere -- and so a future reader knows why it will not decrypt
    // on another machine.
    j["pod_token_dpapi"] = protect_secret(pod_token);

    std::ofstream file(path(), std::ios::trunc);
    if (!file) return false;
    file << j.dump(2) << "\n";
    return file.good();
}

}  // namespace mx
