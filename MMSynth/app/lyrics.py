"""Words on notes.

The model is ACE Studio's, because it is better than the obvious one:

    every note carries a lyric      a note is never silent by accident
    a new note is born singing      the default is `da`
    `-` holds the previous syllable a melisma, said explicitly
    `twinkle#1` `twinkle#2`         a syllable that still names its word
    phonemes per note               editable, because English spelling lies

The convention this replaces was "an empty lyric means hold". It cannot work: an
empty lyric is also what "nobody has typed here yet" looks like, and those two
states must not be indistinguishable. `-` is a hold; `da` is a default; a word is
a word.

Everything here is a pure function over a list of notes. The document lives in
the client, so distributing a line returns what the notes *would* become and the
client decides whether to keep it. That is what makes "warn before overwriting"
possible without the service holding any state to warn about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import g2p

DEFAULT_LYRIC = "da"
HOLD = "-"

_WORD = re.compile(r"[A-Za-z']+")
_PART = re.compile(r"^(.*)#(\d+)$")


def words(text: str) -> list[str]:
    """Words, in order. Punctuation carries nothing a singer can sing."""
    return _WORD.findall(text or "")


def base_word(label: str) -> str:
    """`twinkle#2` -> `twinkle`. A plain label is already its own word."""
    match = _PART.match(label or "")
    return match.group(1) if match else (label or "")


def part_index(label: str) -> int:
    """Which syllable of the word this is, 1-based. 1 when there is only one."""
    match = _PART.match(label or "")
    return int(match.group(2)) if match else 1


def phonemes_for(label: str) -> list[str]:
    """The phonemes a label implies, before anyone edits them."""
    text = (label or "").strip()
    if not text or text == HOLD:
        return []
    pieces = g2p.syllables(base_word(text))
    if not pieces:
        return []
    index = min(part_index(text), len(pieces)) - 1
    return list(pieces[index][1])


def line_syllables(line: str) -> list[tuple[str, list[str]]]:
    """A typed line as one (label, phonemes) pair per syllable."""
    out: list[tuple[str, list[str]]] = []
    for word in words(line):
        out.extend(g2p.syllables(word))
    return out


def hand_set(note) -> bool:
    """Whether someone has deliberately put this note's words there.

    Derived rather than flagged, so no extra field has to be kept honest through
    a save, a load and a round trip: a lyric that is not the default and not a
    hold was typed, and phonemes that disagree with what the lyric implies were
    edited. Both are exactly the work that a lyric-bar overwrite destroys, which
    is why this is the thing to count.
    """
    lyric = (getattr(note, "lyric", "") or "").strip()
    if lyric and lyric not in (DEFAULT_LYRIC, HOLD):
        return True
    supplied = list(getattr(note, "phonemes", None) or [])
    if supplied and supplied != phonemes_for(lyric or DEFAULT_LYRIC):
        return True
    return False


@dataclass
class Assignment:
    """What one note would become."""
    lyric: str
    phonemes: list[str]


@dataclass
class Distribution:
    notes: list[Assignment] = field(default_factory=list)
    # Syllables with no note to land on. An ordinary mid-edit state, not an
    # error: you type the line first and draw the rest of the melody after.
    pending: list[str] = field(default_factory=list)
    held: int = 0
    replaced: int = 0
    note: str = ""


def distribute(notes, line: str) -> Distribution:
    """Spread a typed line across the notes, one syllable each.

    Notes past the end of the line hold the last syllable rather than falling
    silent, which is what a singer does with a long tail. Syllables past the end
    of the melody wait.
    """
    syllables = line_syllables(line)
    replaced = sum(1 for n in notes if hand_set(n))

    out: list[Assignment] = []
    held = 0
    for i, _note in enumerate(notes):
        if i < len(syllables):
            label, sounds = syllables[i]
            out.append(Assignment(lyric=label, phonemes=list(sounds)))
        elif syllables:
            out.append(Assignment(lyric=HOLD, phonemes=[]))
            held += 1
        else:
            out.append(Assignment(lyric=DEFAULT_LYRIC,
                                  phonemes=phonemes_for(DEFAULT_LYRIC)))

    pending = [label for label, _ in syllables[len(notes):]]

    said = [f"{min(len(syllables), len(notes))} syllable(s) on {len(notes)} note(s)"]
    if held:
        said.append(f"the last one held across {held} more")
    if pending:
        said.append(f"{len(pending)} waiting for notes: "
                    + " ".join(pending[:6]) + ("..." if len(pending) > 6 else ""))
    if replaced:
        said.append(f"this replaces {replaced} note(s) you set by hand")

    return Distribution(notes=out, pending=pending, held=held,
                        replaced=replaced, note="; ".join(said))


def ensure_defaults(notes) -> int:
    """Give every lyric-less note a `da`, and every note its phonemes.

    Called on the way in, so a melody drawn without a thought about words is
    still singable and still shows a phoneme row. Returns how many were filled.
    """
    filled = 0
    for note in notes:
        lyric = (note.lyric or "").strip()
        if not lyric:
            note.lyric = DEFAULT_LYRIC
            filled += 1
            lyric = DEFAULT_LYRIC
        if not getattr(note, "phonemes", None):
            note.phonemes = phonemes_for(lyric)
    return filled


def current_line(notes) -> str:
    """Rebuild the lyric bar from the notes.

    Holds contribute nothing and repeated parts of one word collapse back into
    it, so a line that was distributed and then read back is the line that was
    typed.
    """
    out: list[str] = []
    last_word = None
    last_part = 0
    for note in notes:
        label = (note.lyric or "").strip()
        if not label or label == HOLD:
            continue
        word, part = base_word(label), part_index(label)
        if word == last_word and part == last_part + 1:
            last_part = part
            continue
        out.append(word)
        last_word, last_part = word, part
    return " ".join(out)


@dataclass
class Segment:
    """One sung syllable, and the notes it is sung across."""
    lyric: str
    phonemes: list[str]
    first: int
    count: int = 1


def segments(notes) -> list[Segment]:
    """Group notes into what actually gets sung.

    A hold joins the segment before it. A hold with nothing before it is
    dropped: there is no previous syllable for it to extend, and inventing one
    would put a word on a pitch nobody chose.
    """
    out: list[Segment] = []
    for i, note in enumerate(notes):
        label = (note.lyric or "").strip() or DEFAULT_LYRIC
        if label == HOLD:
            if out:
                out[-1].count += 1
            continue
        sounds = list(getattr(note, "phonemes", None) or []) or phonemes_for(label)
        out.append(Segment(lyric=label, phonemes=sounds, first=i))
    return out
