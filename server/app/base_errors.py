"""Errors shared across modules, kept here to avoid import cycles."""


class MusicmakerError(RuntimeError):
    """Base for everything this service raises deliberately."""


class RenderError(MusicmakerError):
    """A rendering step (MIDI, mixdown, export) failed."""


class NotReady(MusicmakerError):
    """The project is not in a state where this operation makes sense."""
