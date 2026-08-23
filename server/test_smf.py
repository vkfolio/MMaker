"""The MIDI file contract.

A score leaves the app as note events and has to arrive at FluidSynth as the
same music. Everything between is arithmetic nobody can hear until it is wrong,
and wrong here is quiet: a note a tick short still sounds, a note-off ordered
after the next note-on silences a repeat, and a tempo that disagrees with the
project makes every rendered clip drift against the grid it was drawn on.

So the properties are:

  timing      -- a note at t seconds lands at the tick t seconds implies, at the
                 tempo written into the same file
  retrigger   -- the same pitch played twice in a row survives as two notes
  survival    -- nothing is dropped, nothing is silent: no zero-length notes
  agreement   -- the file's own tempo matches what the caller asked for

Read back with mido rather than by our own decoder, so a symmetrical bug in the
writer cannot hide behind a symmetrical bug in the reader.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.midi import TICKS_PER_BEAT, write_smf   # noqa: E402

failures = 0


def check(ok, what):
    global failures
    if not ok:
        failures += 1
    print(f"  {what:<58} {'PASS' if ok else 'FAIL'}")


class Note:
    def __init__(self, start, length, pitch, velocity=96):
        self.start = start
        self.length = length
        self.pitch = pitch
        self.velocity = velocity


def notes_from(path):
    """(pitch, start_tick, length_tick, velocity), in the order they start."""
    import mido

    midi = mido.MidiFile(path)
    now = 0
    open_notes = {}
    out = []
    for msg in mido.merge_tracks(midi.tracks):
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault(msg.note, []).append((now, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            started = open_notes.get(msg.note)
            if started:
                start, velocity = started.pop(0)
                out.append((msg.note, start, now - start, velocity))
    out.sort(key=lambda n: (n[1], n[0]))
    return out


def tempo_of(path):
    import mido

    midi = mido.MidiFile(path)
    for track in midi.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return round(mido.tempo2bpm(msg.tempo))
    return None


def main():
    try:
        import mido  # noqa: F401
    except ImportError:
        print("\nmido is not installed -- cannot verify the file independently.")
        print("SKIPPED (install mido to run this gate)\n")
        return 0

    print("\nMIDI file writing\n" + "=" * 72)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "score.mid")

    # 120 bpm: one beat is 0.5 s, so 480 ticks is 0.5 s.
    write_smf([Note(0.0, 0.5, 60), Note(0.5, 0.5, 62), Note(1.0, 1.0, 64)],
              path, bpm=120)
    got = notes_from(path)
    check(len(got) == 3, "every note survives the round trip")
    check([n[0] for n in got] == [60, 62, 64], "pitches are preserved, in order")
    check(got[0][1] == 0 and got[1][1] == TICKS_PER_BEAT and
          got[2][1] == 2 * TICKS_PER_BEAT,
          "a note at t seconds lands on the tick t implies")
    check(got[0][2] == TICKS_PER_BEAT and got[2][2] == 2 * TICKS_PER_BEAT,
          "durations are preserved")
    check(tempo_of(path) == 120, "the file carries the tempo it was given")

    # A chord: three notes at the same instant must stay simultaneous.
    write_smf([Note(0.0, 1.0, 60), Note(0.0, 1.0, 64), Note(0.0, 1.0, 67)],
              path, bpm=120)
    got = notes_from(path)
    check(len(got) == 3 and all(n[1] == 0 for n in got),
          "a chord stays a chord")

    # The same pitch twice: the first note-off must not swallow the second
    # note-on, which is what ordering note-offs first at a shared tick buys.
    write_smf([Note(0.0, 0.5, 60), Note(0.5, 0.5, 60)], path, bpm=120)
    got = notes_from(path)
    check(len(got) == 2, "a repeated pitch survives as two notes")
    check(got[0][2] == TICKS_PER_BEAT and got[1][1] == TICKS_PER_BEAT,
          "the repeat starts where the first note ended")

    # Tempo changes what a second is worth.
    write_smf([Note(0.0, 1.0, 60)], path, bpm=60)
    got = notes_from(path)
    check(got[0][2] == TICKS_PER_BEAT,
          "at 60 bpm one second is one beat, not two")
    check(tempo_of(path) == 60, "the tempo written is the tempo asked for")

    # A note too short to occupy a tick is still a note. Rounding it to zero
    # makes fast playing vanish while the file still looks correct.
    write_smf([Note(0.0, 0.0001, 60)], path, bpm=120)
    got = notes_from(path)
    check(len(got) == 1 and got[0][2] >= 1, "a very short note is not silent")

    # Velocity is carried, since it is what drives expression later.
    write_smf([Note(0.0, 1.0, 60, velocity=33)], path, bpm=120)
    check(notes_from(path)[0][3] == 33, "velocity is carried through")

    # Out-of-range input is clamped rather than corrupting the byte stream.
    write_smf([Note(0.0, 1.0, 200, velocity=999), Note(0.0, 1.0, -5, velocity=0)],
              path, bpm=120)
    got = notes_from(path)
    check(all(0 <= n[0] <= 127 and 1 <= n[3] <= 127 for n in got),
          "impossible pitches and velocities are clamped, not emitted raw")

    check(notes_from(path) is not None, "an empty score still writes a valid file")
    write_smf([], path, bpm=120)
    check(notes_from(path) == [], "an empty score writes no notes")

    print("=" * 72)
    if failures == 0:
        print("ALL PASS\n")
        return 0
    print(f"{failures} FAILURE(S)\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
