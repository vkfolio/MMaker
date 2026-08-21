"""Phase 1 tests: the boot flow, run without a GPU via stub mode.

    python -m pytest server/test_boot.py -q      (or just run this file)
"""

import os
import tempfile
import time
from pathlib import Path

os.environ["MUSICMAKER_STUB_MODELS"] = "1"
_TMP = Path(tempfile.mkdtemp(prefix="mm_test_"))
os.environ["MUSICMAKER_MODELS_DIR"] = str(_TMP / "models")
os.environ["MUSICMAKER_DATA_DIR"] = str(_TMP / "data")

from fastapi.testclient import TestClient      # noqa: E402
from app import bootstrap                       # noqa: E402
from app.main import app                        # noqa: E402


def _wait_ready(client, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/health").json()
        if body["status"] in ("ready", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError("bootstrap never settled")


def test_boot_reaches_ready():
    with TestClient(app) as client:
        body = _wait_ready(client)
        assert body["status"] == "ready", body["message"]
        assert body["stub_models"] is True


def test_only_enabled_models_download():
    with TestClient(app) as client:
        models = {m["name"]: m for m in _wait_ready(client)["models"]}
        assert models["acestep"]["status"] == "ready"
        # Later-phase models must not be fetched before they are needed.
        for name in ("demucs", "seedvc", "stableaudio"):
            assert models[name]["status"] == "skipped", name


def test_capabilities_track_ready_models():
    """Stub mode serves every capability -- that is what makes the API
    testable without a GPU. Real mode gates on which weights are present."""
    with TestClient(app) as client:
        _wait_ready(client)
        info = client.get("/api/info").json()
        assert all(info["capabilities"].values())
        assert info["engines"]["forced_stub"] is True
        assert info["engines"]["music"] == "stub"


def test_engine_endpoint_reports_tier_models():
    """The tiers we would ask for must be inspectable, so a silent fallback to
    the default model is checkable rather than inferred from timings."""
    with TestClient(app) as client:
        _wait_ready(client)
        body = client.get("/api/engine").json()
        assert body["engine"] == "stub"          # stub has no model list
        assert "requested_tiers" not in body or body["requested_tiers"]


def test_ui_is_served():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "musicmaker" in res.text


def test_second_boot_skips_download():
    """Weights already on the volume must not be re-fetched."""
    with TestClient(app) as client:
        _wait_ready(client)
    started = time.time()
    bootstrap.run()                       # simulate a pod restart
    assert time.time() - started < 1.0, "re-downloaded already-present weights"
    assert bootstrap.state.status == "ready"


def test_failure_is_reported_not_swallowed():
    """Runs last by name on purpose -- it leaves a deliberately broken
    registry in bootstrap.state, which start_background() resets."""
    registry = _TMP / "bad.yaml"
    registry.write_text(
        "models:\n"
        "  - name: broken\n    label: Broken\n    kind: nonsense\n"
        "    repo_id: x/y\n    approx_gb: 1\n    enabled: true\n"
        "    phase: 2\n    purpose: fail\n", encoding="utf-8")
    from app.config import settings
    original = settings.registry_path
    settings.registry_path = registry
    try:
        bootstrap.run()
        assert bootstrap.state.status == "error"
        assert "unknown model kind" in bootstrap.state.models[0].error
    finally:
        settings.registry_path = original


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:                      # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print("\nall passed" if not failures else f"\n{failures} failed")
    raise SystemExit(1 if failures else 0)
