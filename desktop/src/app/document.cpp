#include "document.h"

#include <cstdlib>
#include <fstream>

#include <nlohmann/json.hpp>

namespace mx {
namespace {

using json = nlohmann::json;
constexpr int kSchema = 1;

}  // namespace

std::filesystem::path documents_root() {
    if (const char* profile = std::getenv("USERPROFILE"); profile && *profile)
        return std::filesystem::path(profile) / "Documents" / "MusicMaker";
    return std::filesystem::current_path();
}

bool save_document(const Session& session, const std::filesystem::path& path,
                   std::string* error) {
    json doc;
    doc["schema"] = kSchema;
    doc["title"] = session.title;
    doc["bpm"] = session.bpm;
    doc["beats_per_bar"] = session.beats_per_bar;
    doc["rate"] = session.rate;
    doc["loop_start"] = session.loop_start;
    doc["loop_end"] = session.loop_end;

    for (const auto& t : session.tracks) {
        doc["tracks"].push_back({{"id", t.id},
                                 {"name", t.name},
                                 {"gain", t.gain},
                                 {"pan", t.pan},
                                 {"muted", t.muted},
                                 {"soloed", t.soloed},
                                 {"color", t.color}});
    }
    for (const auto& s : session.sources) {
        // Provenance is the point of this record. Without it a document can
        // never recover its audio, and the cache path alone is a local detail
        // that means nothing on another machine.
        doc["sources"].push_back({{"id", s.id},
                                  {"label", s.label},
                                  {"project_id", s.project_id},
                                  {"stem_id", s.stem_id},
                                  {"version_id", s.version_id},
                                  {"local_path", s.local_path.string()},
                                  {"source_rate", s.source_rate}});
    }
    for (const auto& sc : session.scores) {
        json notes = json::array();
        for (const auto& n : sc.notes)
            notes.push_back({{"start", n.start},
                             {"length", n.length},
                             {"pitch", n.pitch},
                             {"lyric", n.lyric}});
        doc["scores"].push_back({{"id", sc.id},
                                 {"voice_id", sc.voice_id},
                                 {"voice_label", sc.voice_label},
                                 {"language", sc.language},
                                 {"key", sc.key},
                                 {"notes", std::move(notes)}});
    }
    for (const auto& c : session.clips) {
        doc["clips"].push_back({{"id", c.id},
                                {"track_id", c.track_id},
                                {"source_id", c.source_id},
                                {"score_id", c.score_id},
                                {"start_frame", c.start_frame},
                                {"length", c.length},
                                {"source_offset", c.source_offset},
                                {"fade_in", c.fade_in},
                                {"fade_out", c.fade_out},
                                {"gain", c.gain},
                                {"name", c.name}});
    }
    // Ids must survive a round trip, or a reload would hand out ids that are
    // already in use and two objects would share an identity.
    doc["next"] = {{"track", session.next_track},
                   {"clip", session.next_clip},
                   {"source", session.next_source},
                   {"score", session.next_score}};

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    // Write beside the target and rename, so an interrupted save cannot destroy
    // the last good document -- the failure mode that loses work.
    const auto temp = path.string() + ".saving";
    {
        std::ofstream file(temp, std::ios::trunc);
        if (!file) {
            if (error) *error = "cannot write " + temp;
            return false;
        }
        file << doc.dump(2) << "\n";
        if (!file.good()) {
            if (error) *error = "write failed";
            return false;
        }
    }
    std::filesystem::rename(temp, path, ec);
    if (ec) {
        if (error) *error = "cannot replace the document: " + ec.message();
        return false;
    }
    return true;
}

LoadResult load_document(Session& session, const std::filesystem::path& path) {
    LoadResult out;
    std::ifstream file(path);
    if (!file) {
        out.error = "cannot open " + path.string();
        return out;
    }

    json doc;
    try {
        file >> doc;
    } catch (const json::exception& e) {
        out.error = std::string("not a readable project: ") + e.what();
        return out;
    }

    const int schema = doc.value("schema", 0);
    if (schema > kSchema) {
        out.error = "this project was written by a newer version (schema " +
                    std::to_string(schema) + ")";
        return out;
    }

    session = Session{};
    session.title = doc.value("title", "Untitled");
    session.bpm = doc.value("bpm", 120.0);
    session.beats_per_bar = doc.value("beats_per_bar", 4);
    session.rate = doc.value("rate", 48000u);
    session.loop_start = doc.value("loop_start", 0LL);
    session.loop_end = doc.value("loop_end", 0LL);

    for (const auto& t : doc.value("tracks", json::array())) {
        Track track;
        track.id = t.value("id", 0u);
        track.name = t.value("name", "");
        track.gain = t.value("gain", 1.0f);
        track.pan = t.value("pan", 0.0f);
        track.muted = t.value("muted", false);
        track.soloed = t.value("soloed", false);
        track.color = t.value("color", 0xff4f8ef7u);
        session.tracks.push_back(std::move(track));
    }
    for (const auto& s : doc.value("sources", json::array())) {
        Source source;
        source.id = s.value("id", 0u);
        source.label = s.value("label", "");
        source.project_id = s.value("project_id", "");
        source.stem_id = s.value("stem_id", "");
        source.version_id = s.value("version_id", "");
        source.local_path = s.value("local_path", "");
        source.source_rate = s.value("source_rate", 0u);
        if (!source.local_path.empty() &&
            !std::filesystem::exists(source.local_path))
            ++out.missing_audio;
        session.sources.push_back(std::move(source));
    }
    for (const auto& sc : doc.value("scores", json::array())) {
        Score score;
        score.id = sc.value("id", 0u);
        score.voice_id = sc.value("voice_id", "");
        score.voice_label = sc.value("voice_label", "");
        score.language = sc.value("language", "en");
        score.key = sc.value("key", "C Major");
        for (const auto& n : sc.value("notes", json::array())) {
            Note note;
            note.start = n.value("start", 0LL);
            note.length = n.value("length", 0LL);
            note.pitch = static_cast<uint8_t>(n.value("pitch", 60));
            note.lyric = n.value("lyric", "");
            score.notes.push_back(std::move(note));
        }
        session.scores.push_back(std::move(score));
    }
    for (const auto& c : doc.value("clips", json::array())) {
        Clip clip;
        clip.id = c.value("id", 0u);
        clip.track_id = c.value("track_id", 0u);
        clip.source_id = c.value("source_id", 0u);
        clip.score_id = c.value("score_id", 0u);
        clip.start_frame = c.value("start_frame", 0LL);
        clip.length = c.value("length", 0LL);
        clip.source_offset = c.value("source_offset", 0LL);
        clip.fade_in = c.value("fade_in", 0LL);
        clip.fade_out = c.value("fade_out", 0LL);
        clip.gain = c.value("gain", 1.0f);
        clip.name = c.value("name", "");
        session.clips.push_back(std::move(clip));
    }

    if (auto it = doc.find("next"); it != doc.end() && it->is_object()) {
        session.next_track = it->value("track", 1u);
        session.next_clip = it->value("clip", 1u);
        session.next_source = it->value("source", 1u);
        session.next_score = it->value("score", 1u);
    }
    // Whatever the file said, never hand out an id that is already taken.
    for (const auto& t : session.tracks)
        session.next_track = std::max(session.next_track, t.id + 1);
    for (const auto& c : session.clips)
        session.next_clip = std::max(session.next_clip, c.id + 1);
    for (const auto& s : session.sources)
        session.next_source = std::max(session.next_source, s.id + 1);
    for (const auto& sc : session.scores)
        session.next_score = std::max(session.next_score, sc.id + 1);

    out.ok = true;
    return out;
}

}  // namespace mx
