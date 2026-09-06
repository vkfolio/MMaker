"""Reading a melody out of audio.

Two jobs need this and neither can be done any other way:

  * **Voice presets.** SoulX-Singer conditions on a reference clip *and that
    clip's score* -- its notes and phonemes, not just its waveform. So a clip
    somebody drops in has to be transcribed before it can become a voice.
  * **Singing over a track.** A melody that exists only as a recording has to
    become notes before it can be sung with different words.

This is pitch tracking, so it is an estimate. `pyin` because the input is a
single melodic line, which is exactly what it is for, and because it reports
*voicing* -- silence stays silence instead of becoming a confident wrong note.

The output is deliberately coarse. Notes are rounded to semitones and short
fragments are absorbed, because the consumer is a singing model that will
re-perform them, not a notation editor. Chasing vibrato into separate notes
would produce a score that is more accurate and less useful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import RenderError
from .score import Note

SR = 24000            # SoulX's rate; nothing here needs more
MIN_NOTE_S = 0.09     # shorter than this is an artefact of the tracker
MIN_REST_S = 0.12     # a gap shorter than this is phrasing, not a rest
MIN_SPLIT_S = 0.16    # two attacks closer than this are one attack, twice detected


def _frames(path: Path):
    try:
        import librosa
    except ImportError as exc:                                  # pragma: no cover
        raise RenderError("librosa is needed to read a melody from audio") from exc

    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if y.size < SR // 4:
        raise RenderError("that clip is too short to read a melody from")

    f0, voiced, _prob = librosa.pyin(
        y, sr=sr,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr)
    midi = np.full(f0.shape, np.nan)
    ok = voiced & ~np.isnan(f0)
    midi[ok] = librosa.hz_to_midi(f0[ok])

    # Onsets, separately. A repeated note at the same pitch is invisible in the
    # pitch contour -- C C G G reads as one long C and one long G -- because
    # what distinguishes a repeat is the attack, not the pitch. Without this,
    # "twinkle twinkle" comes back as four notes instead of seven and the
    # melody quietly loses half its rhythm.
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    onsets = np.asarray(onsets, dtype=float)
    # One attack often fires the detector twice -- the transient and its decay
    # a few frames later. Left alone that splits a single note in two, so
    # onsets closer together than a note could be are collapsed.
    if onsets.size > 1:
        keep = [onsets[0]]
        for t in onsets[1:]:
            if t - keep[-1] >= MIN_SPLIT_S:
                keep.append(t)
        onsets = np.asarray(keep)
    return times, midi, len(y) / sr, onsets


def melody(path) -> tuple[list[Note], str]:
    """Notes, and a sentence about how much to trust them.

    The sentence matters: this is an estimate, and a UI that presents an
    estimate as a fact is lying by omission.
    """
    path = Path(path)
    times, midi, duration, onsets = _frames(path)
    if not np.any(~np.isnan(midi)):
        raise RenderError("no pitched sound found in that clip")

    step = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    rounded = np.where(np.isnan(midi), np.nan, np.round(midi))

    notes: list[Note] = []
    start_i = None
    for i in range(len(rounded) + 1):
        here = rounded[i] if i < len(rounded) else np.nan
        prev = rounded[i - 1] if i > 0 else np.nan
        same = (not np.isnan(here) and not np.isnan(prev) and here == prev)
        if not same:
            if start_i is not None and not np.isnan(prev):
                begin, end = times[start_i], times[min(i, len(times) - 1)]
                if end - begin >= MIN_NOTE_S:
                    notes.append(Note(start=float(begin),
                                      length=float(max(MIN_NOTE_S, end - begin)),
                                      pitch=int(np.clip(prev, 0, 127)),
                                      velocity=96))
            start_i = i if not np.isnan(here) else None

    if not notes:
        raise RenderError("the pitch was too unstable to read notes from")

    def struck_between(a: float, b: float) -> bool:
        """Is there an attack strictly inside this span?"""
        return bool(np.any((onsets > a + 0.04) & (onsets < b - 0.02)))

    # Join neighbours of the same pitch separated by a tracker dropout -- unless
    # something was struck in the gap, which means it was played twice.
    merged: list[Note] = [notes[0]]
    for n in notes[1:]:
        last = merged[-1]
        end = last.start + last.length
        gap = n.start - end
        if n.pitch == last.pitch and gap < MIN_REST_S and not struck_between(end - 0.02, n.start + 0.02):
            last.length = (n.start + n.length) - last.start
        else:
            merged.append(n)

    # Split a held note wherever it was struck again. This is what keeps a
    # repeated pitch from collapsing into one long note.
    split: list[Note] = []
    for n in merged:
        cuts = [t for t in onsets if n.start + MIN_SPLIT_S < t < n.end - MIN_SPLIT_S]
        at = n.start
        for cut in cuts:
            split.append(Note(start=at, length=cut - at, pitch=n.pitch,
                              velocity=n.velocity))
            at = cut
        split.append(Note(start=at, length=(n.start + n.length) - at,
                          pitch=n.pitch, velocity=n.velocity))

    covered = sum(n.length for n in split)
    repeats = len(split) - len(merged)
    said = (f"read {len(split)} note(s) from {duration:.1f}s of audio, "
            f"{100 * covered / max(duration, 1e-6):.0f}% pitched"
            + (f", {repeats} of them repeated notes found by their attack" if repeats else "")
            + ". Pitch tracking is an estimate -- check it before relying on it.")
    return split, said
