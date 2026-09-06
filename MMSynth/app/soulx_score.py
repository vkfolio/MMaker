"""MMSynth notes to SoulX-Singer's score metadata.

The model's target format is a list of segments, each a set of parallel
space-separated strings, one entry per *note-sized unit*:

    duration    seconds, and they must tile the segment with no gaps
    text        the word, or <SP> for a rest
    phoneme     en_T-W-IH1-NG for a note, <SP> for a rest
    note_pitch  MIDI number, 0 for a rest
    note_type   1 rest, 2 a syllable starting, 3 the same syllable continuing
    f0          frame-level pitch -- omitted here, and optional for score control

Two details are worth stating because they are not obvious from the format and
are load-bearing:

**Rests are entries, not gaps.** The durations tile the segment, so silence
between notes has to be written down as a `<SP>` entry or every note after it
drifts early by the length of the gap. This is the single easiest way to get a
melody that is right in the piano roll and wrong in the audio.

**A held syllable repeats its phoneme with note_type 3.** That is how their own
English example writes "you're you're" across two notes, and it is exactly what
MMSynth's `-` means. So a melisma is not a special case here; it is the same
mechanism the model was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import g2p, lyrics

LANGUAGE = "English"
REST = g2p.REST

# A gap shorter than this is absorbed into the note before it. Writing a 4 ms
# rest helps nobody and gives the model a unit it cannot sing.
MIN_REST = 0.03

# Segments are inferred one at a time. Long ones cost memory and drift; their
# own examples sit around 5-10 s. Split at a rest once past this.
MAX_SEGMENT_S = 24.0
SPLIT_REST_S = 0.35


@dataclass
class Entry:
    duration: float
    text: str
    phoneme: str
    pitch: int
    note_type: int


def entries_for(notes) -> list[Entry]:
    """One entry per note, with rests written in between."""
    out: list[Entry] = []
    previous_phoneme = REST
    cursor = 0.0

    for note in sorted(notes, key=lambda n: n.start):
        gap = note.start - cursor
        if gap >= MIN_REST:
            out.append(Entry(gap, REST, REST, 0, 1))
        elif gap > 0 and out:
            # Absorb a sliver into whatever came before, so the tiling stays
            # exact without inventing an unsingable unit.
            out[-1].duration += gap

        label = (note.lyric or "").strip() or lyrics.DEFAULT_LYRIC
        if label == lyrics.HOLD:
            # A hold repeats the previous phoneme. With nothing before it there
            # is no syllable to continue, so it becomes an ordinary `da` rather
            # than a continuation of silence.
            if previous_phoneme != REST:
                phoneme, note_type, text = previous_phoneme, 3, label
            else:
                sounds = lyrics.phonemes_for(lyrics.DEFAULT_LYRIC)
                phoneme, note_type, text = g2p.token(sounds), 2, lyrics.DEFAULT_LYRIC
        else:
            sounds = list(getattr(note, "phonemes", None) or []) \
                or lyrics.phonemes_for(label)
            phoneme, note_type = g2p.token(sounds), 2
            text = lyrics.base_word(label)

        out.append(Entry(max(0.03, note.length), text, phoneme,
                         int(note.pitch), note_type))
        previous_phoneme = phoneme
        cursor = note.start + note.length

    return out


def _segment(entries: list[Entry], start_s: float, index: int) -> dict:
    """One segment, in the shape the model's DataProcessor reads."""
    # Their own examples open and close on a rest, which gives the model room to
    # start and stop rather than beginning mid-phonation.
    if not entries or entries[0].phoneme != REST:
        entries = [Entry(0.2, REST, REST, 0, 1)] + entries
    if entries[-1].phoneme != REST:
        entries = entries + [Entry(0.3, REST, REST, 0, 1)]

    total = sum(e.duration for e in entries)
    return {
        "index": f"mmsynth_{index}",
        "language": LANGUAGE,
        "time": [int(round(start_s * 1000)), int(round((start_s + total) * 1000))],
        "duration": " ".join(f"{e.duration:.2f}" for e in entries),
        "text": " ".join(e.text for e in entries),
        "phoneme": " ".join(e.phoneme for e in entries),
        "note_pitch": " ".join(str(e.pitch) for e in entries),
        "note_type": " ".join(str(e.note_type) for e in entries),
    }


def build_target(notes) -> list[dict]:
    """The full target metadata: a list of segments covering the melody.

    Split at a long rest once a segment gets past MAX_SEGMENT_S, never mid-word,
    because a segment boundary inside a syllable is an audible seam.
    """
    entries = entries_for(notes)
    if not entries:
        return []

    segments: list[dict] = []
    current: list[Entry] = []
    elapsed = 0.0            # seconds already committed to earlier segments
    running = 0.0

    for entry in entries:
        current.append(entry)
        running += entry.duration
        long_enough = running >= MAX_SEGMENT_S
        breakable = entry.phoneme == REST and entry.duration >= SPLIT_REST_S
        if long_enough and breakable:
            segments.append(_segment(current, elapsed, len(segments)))
            elapsed += sum(e.duration for e in current)
            current, running = [], 0.0

    if current:
        segments.append(_segment(current, elapsed, len(segments)))
    return segments
