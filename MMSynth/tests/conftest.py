"""Test setup.

The environment has to be set before app.config is imported, because settings
are read once at class-definition time. conftest runs first, which is what makes
that safe.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["MMSYNTH_ENGINE"] = "stub"
os.environ["MMSYNTH_STUB_MODELS"] = "1"
os.environ.setdefault("MMSYNTH_DATA_DIR",
                      tempfile.mkdtemp(prefix="mmsynth-test-"))
os.environ.setdefault("MMSYNTH_MODELS_DIR",
                      tempfile.mkdtemp(prefix="mmsynth-models-"))
# The token middleware is exercised in its own test; everything else would
# otherwise need a header on every request.
os.environ.pop("MMSYNTH_API_TOKEN", None)
