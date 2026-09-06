"""The two failures a caller can do something about.

Everything else is a bug and should raise normally rather than being dressed up
as an API error.
"""


class MMSynthError(RuntimeError):
    """Base for anything we raise deliberately."""


class NotReady(MMSynthError):
    """The request is fine; this machine cannot serve it yet.

    A missing soundfont, a model that has not finished downloading, an engine
    that has not been wired. Always say which of those it is -- "render failed"
    sends someone reading logs for an hour.
    """


class RenderError(MMSynthError):
    """A tool we shelled out to, or a model we called, failed."""


class NotSupported(MMSynthError):
    """This engine cannot do this at all, and never will."""
