"""English spelling to ARPAbet, and ARPAbet to syllables.

What this is for: the row of phonemes above each note, and the string the singing
model is actually conditioned on. The user can see what will be sung and correct
it, which every singing tool needs because English spelling does not determine
English pronunciation.

Two backends, in order:

1. **g2p_en** (Apache-2.0), which is what SoulX-Singer's own preprocessing uses.
   When it is installed -- it is, on the pod -- its answer is authoritative,
   because matching the tool the model's training data was built with removes a
   whole class of mismatch nobody would ever debug.
2. **Rules**, below, for a laptop with nothing installed. They are wrong
   sometimes. That is survivable precisely because the result is editable: this
   produces the first guess, not the answer.

Everything is checked against SoulX-Singer's own `phone_set.json`, vendored at
data/phone_set.json. An unknown phone is not a bad noise later -- it is a
KeyError inside the model's data processor, so it has to be caught here.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

VOWELS = "aeiou"

# ARPAbet nuclei, without stress digits. Everything else is a consonant, which
# is all the syllable splitter needs to know.
NUCLEI = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"}

# Words the rules get wrong and that turn up in lyrics constantly. Short on
# purpose: a long table here is a dictionary in disguise, worse maintained. Only
# consulted when g2p_en is absent.
EXCEPTIONS = {
    "a": ["AH0"], "the": ["DH", "AH0"], "to": ["T", "UW1"], "do": ["D", "UW1"],
    "of": ["AH1", "V"], "you": ["Y", "UW1"], "your": ["Y", "AO1", "R"],
    "are": ["AA1", "R"], "i": ["AY1"], "my": ["M", "AY1"], "by": ["B", "AY1"],
    "is": ["IH1", "Z"], "was": ["W", "AH1", "Z"], "one": ["W", "AH1", "N"],
    "once": ["W", "AH1", "N", "S"], "two": ["T", "UW1"], "who": ["HH", "UW1"],
    "what": ["W", "AH1", "T"], "want": ["W", "AA1", "N", "T"],
    "where": ["W", "EH1", "R"], "there": ["DH", "EH1", "R"],
    "here": ["HH", "IH1", "R"], "were": ["W", "ER1"],
    "have": ["HH", "AE1", "V"], "give": ["G", "IH1", "V"],
    "live": ["L", "IH1", "V"], "love": ["L", "AH1", "V"],
    "come": ["K", "AH1", "M"], "some": ["S", "AH1", "M"],
    "done": ["D", "AH1", "N"], "gone": ["G", "AO1", "N"],
    "none": ["N", "AH1", "N"], "eye": ["AY1"], "eyes": ["AY1", "Z"],
    "heart": ["HH", "AA1", "R", "T"],
    "wonder": ["W", "AH1", "N", "D", "ER0"],
    "little": ["L", "IH1", "T", "AH0", "L"],
    "twinkle": ["T", "W", "IH1", "NG", "K", "AH0", "L"],
    "star": ["S", "T", "AA1", "R"], "how": ["HH", "AW1"],
    "da": ["D", "AA1"], "la": ["L", "AA1"], "na": ["N", "AA1"],
    "oh": ["OW1"], "ah": ["AA1"], "mm": ["M"],
}

DIGRAPHS: list[tuple[str, list[str]]] = [
    ("tch", ["CH"]), ("dge", ["JH"]), ("igh", ["AY"]), ("ough", ["AH", "F"]),
    ("augh", ["AE", "F"]), ("eigh", ["EY"]), ("tion", ["SH", "AH", "N"]),
    ("sion", ["ZH", "AH", "N"]),
    ("ch", ["CH"]), ("sh", ["SH"]), ("th", ["TH"]), ("ph", ["F"]),
    ("wh", ["W"]), ("ck", ["K"]), ("ng", ["NG"]), ("qu", ["K", "W"]),
    ("ai", ["EY"]), ("ay", ["EY"]), ("ea", ["IY"]), ("ee", ["IY"]),
    ("ie", ["IY"]), ("oa", ["OW"]), ("oo", ["UW"]), ("ou", ["AW"]),
    ("ow", ["OW"]), ("oi", ["OY"]), ("oy", ["OY"]), ("au", ["AO"]),
    ("aw", ["AO"]), ("ei", ["EY"]), ("eu", ["Y", "UW"]), ("ue", ["UW"]),
    ("ui", ["UW"]),
    ("ar", ["AA", "R"]), ("or", ["AO", "R"]), ("er", ["ER"]),
    ("ir", ["ER"]), ("ur", ["ER"]),
]

SHORT = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}
LONG = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}

SIMPLE = {
    "b": ["B"], "d": ["D"], "f": ["F"], "g": ["G"], "h": ["HH"], "j": ["JH"],
    "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"], "p": ["P"], "r": ["R"],
    "s": ["S"], "t": ["T"], "v": ["V"], "w": ["W"], "z": ["Z"], "x": ["K", "S"],
}

_CLEAN = re.compile(r"[^a-z']")
_DIGIT = re.compile(r"\d")

REST = "<SP>"


def bare(phone: str) -> str:
    """A phone without its stress digit. `AH0` -> `AH`."""
    return _DIGIT.sub("", phone)


@lru_cache(maxsize=1)
def phone_set() -> set[str]:
    """SoulX-Singer's vocabulary, vendored.

    Loaded so that an unknown phone is caught here, where it can be repaired,
    rather than in the model's data processor, where it is a KeyError several
    layers down with no mention of the word that caused it.
    """
    path = Path(os.environ.get("MMSYNTH_PHONESET", DATA / "phone_set.json"))
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _g2p_en():
    """g2p_en, if it is installed. It is on the pod; it may not be locally."""
    try:
        from g2p_en import G2p
        return G2p()
    except Exception:                      # noqa: BLE001 -- nltk data may be absent
        return None


# ---------------------------------------------------------------------------
# The fallback rules
# ---------------------------------------------------------------------------

def _magic_e(word: str) -> bool:
    """vowel + single consonant + final e: the vowel is long, the e is silent."""
    return (len(word) >= 3 and word.endswith("e")
            and word[-2] not in VOWELS and word[-3] in VOWELS)


def rules(word: str) -> list[str]:
    """Letter-to-sound, left to right, longest match first. Unstressed."""
    out: list[str] = []
    i, n = 0, len(word)
    long_vowel = _magic_e(word)

    while i < n:
        c = word[i]
        if i == 0 and word[:2] in ("kn", "gn", "pn", "wr"):
            out.append("N" if word[1] == "n" else "R")
            i += 2
            continue
        if i == n - 1 and c == "e" and out and long_vowel:
            break
        if i == n - 1 and c == "e" and n > 2 and word[-2] not in VOWELS:
            break

        matched = False
        for spelling, phones in DIGRAPHS:
            if word.startswith(spelling, i):
                # "ow" ends a short word as AW ("how"), and sits inside one as
                # OW ("slow"). Neither is reliable, which is why the row above
                # the note is editable.
                out.extend(["AW"] if (spelling == "ow" and i + 2 == n and n <= 4)
                           else phones)
                i += len(spelling)
                matched = True
                break
        if matched:
            continue

        if c in VOWELS:
            last = not any(v in word[i + 1:] for v in VOWELS)
            out.append(LONG[c] if (long_vowel and last) else SHORT[c])
        elif c == "y":
            if i == 0:
                out.append("Y")
            elif i == n - 1:
                out.append("IY" if any(v in word[:i] for v in VOWELS) else "AY")
            else:
                out.append("IH")
        elif c == "c":
            out.append("S" if i + 1 < n and word[i + 1] in "eiy" else "K")
        elif c == "g":
            out.append("JH" if i + 1 < n and word[i + 1] in "eiy" else "G")
        else:
            out.extend(SIMPLE.get(c, []))
        i += 1

    return out or ["AH"]


def _stress(sounds: list[str]) -> list[str]:
    """Put primary stress on the first nucleus and none on the rest.

    Wrong for most words of three syllables or more, right for most of one and
    two, and always in the phone set. The alternative is no digit at all, which
    is not a phone SoulX-Singer knows.
    """
    out, first = [], True
    for p in sounds:
        if p in NUCLEI:
            out.append(p + ("1" if first else "0"))
            first = False
        else:
            out.append(p)
    return out


def repair(sounds: list[str]) -> list[str]:
    """Force a sequence into the model's vocabulary.

    A vowel missing its digit gets one; anything still unknown is dropped rather
    than passed on. Dropping is the lesser evil: a missing phone slurs one
    syllable, an unknown one stops the render.
    """
    known = phone_set()
    if not known:
        return sounds
    out = []
    for p in sounds:
        if f"en_{p}" in known:
            out.append(p)
        elif bare(p) in NUCLEI and f"en_{bare(p)}0" in known:
            out.append(bare(p) + "0")
        elif f"en_{bare(p)}" in known:
            out.append(bare(p))
    return out


# ---------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------

def phonemes(word: str) -> list[str]:
    """ARPAbet with stress, for one word."""
    clean = _CLEAN.sub("", word.lower())
    if not clean:
        return []
    engine = _g2p_en()
    if engine is not None:
        got = [p for p in engine(clean) if p.strip() and p[0].isalpha()]
        if got:
            return repair(got)
    if clean in EXCEPTIONS:
        return repair(list(EXCEPTIONS[clean]))
    return repair(_stress(rules(clean)))


def split_syllables(sounds: list[str]) -> list[list[str]]:
    """Cut a phoneme sequence at syllable boundaries.

    One syllable per nucleus. A consonant run between two nuclei splits so the
    first consonant closes the syllable before and the rest open the one after
    -- T W IH1 NG | K AH0 L. A single consonant goes wholly to the syllable it
    opens, which is what makes "ba-by" rather than "bab-y".
    """
    nuclei = [i for i, p in enumerate(sounds) if bare(p) in NUCLEI]
    if len(nuclei) <= 1:
        return [list(sounds)] if sounds else []

    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        run = b - a - 1
        cuts.append(b - run if run <= 1 else a + 2)
    pieces, previous = [], 0
    for cut in cuts:
        pieces.append(sounds[previous:cut])
        previous = cut
    pieces.append(sounds[previous:])
    return [p for p in pieces if p]


def syllables(word: str) -> list[tuple[str, list[str]]]:
    """A word as (label, phonemes) per syllable.

    The label keeps the whole word and says which part this is --
    `twinkle#1`, `twinkle#2` -- so a note still reads as the word it belongs to
    rather than as an orthographic fragment nobody typed. A one-syllable word is
    just itself.
    """
    clean = _CLEAN.sub("", word.lower())
    if not clean:
        return []
    parts = split_syllables(phonemes(clean))
    if len(parts) <= 1:
        return [(word, parts[0] if parts else [])]
    return [(f"{word}#{i + 1}", part) for i, part in enumerate(parts)]


def token(sounds: list[str]) -> str:
    """One note's phonemes as SoulX-Singer wants them: `en_T-W-IH1-NG`.

    A rest is `<SP>`, which is the model's own symbol rather than ours.
    """
    if not sounds:
        return REST
    return "en_" + "-".join(sounds)


def display(sounds: list[str]) -> str:
    """The row above the note: [t][w][ih1][ng], as ACE Studio draws it."""
    return "".join(f"[{p.lower()}]" for p in sounds)
