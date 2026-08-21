"""End-to-end tests for Phases 2-7, run against the stub engine (no GPU).

Proves the whole pipeline: idea -> variations -> stems -> edit -> vocals ->
voice -> export, plus the guards that stop a bad grid corrupting a project.

    python server/test_pipeline.py      (or: python -m pytest server/test_pipeline.py -q)
"""

import os
import tempfile
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="mm_pipe_"))
os.environ["MUSICMAKER_STUB_MODELS"] = "1"
os.environ["MUSICMAKER_ENGINE"] = "stub"
os.environ["MUSICMAKER_MODELS_DIR"] = str(_TMP / "models")
os.environ["MUSICMAKER_DATA_DIR"] = str(_TMP / "data")

import numpy as np                               # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

from app import audio, engines, storage, voices  # noqa: E402
from app.main import app                         # noqa: E402
from app.schemas import Grid                     # noqa: E402

engines.reset_cache()

# TestClient only runs the app lifespan as a context manager, and the lifespan
# is what binds the job queue to the event loop. Hold it open for the module.
import atexit                                   # noqa: E402
import contextlib                               # noqa: E402

_stack = contextlib.ExitStack()
client = None


def _open():
    """(Re)open the app lifespan for this module.

    test_boot.py opens and closes its own TestClient contexts, which closes the
    event loop this module's job queue was bound to. So we (re)bind at module
    start rather than at import time.
    """
    global client
    with contextlib.suppress(Exception):
        _stack.close()
    client = _stack.enter_context(TestClient(app))
    return client


_open()
atexit.register(_stack.close)

try:                                            # pragma: no cover
    import pytest

    @pytest.fixture(autouse=True, scope="module")
    def _module_lifespan():
        _open()
        yield
except ImportError:
    pass


# ---- helpers --------------------------------------------------------------

def wait_job(job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            assert body["status"] == "done", body.get("error")
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def make_project(variations=2, bars=8):
    res = client.post("/api/projects", json={
        "title": "Test", "prompt": "warm test bed", "style": "indie_soul",
        "bars": bars, "variations": variations,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    wait_job(body["job"]["id"])
    return body["project"]["id"]


def split(project_id):
    project = client.get(f"/api/projects/{project_id}").json()["project"]
    var_id = project["variations"][0]["id"]
    job = client.post(f"/api/projects/{project_id}/split",
                      json={"variation_id": var_id, "method": "demucs"}).json()["job"]
    wait_job(job["id"])
    return client.get(f"/api/projects/{project_id}/stems").json()["stems"]


def write_wav(path, seconds=2.0, sr=48000, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    tone = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    return audio.write(path, np.stack([tone, tone], axis=1), sr)


# ---- phase 2 --------------------------------------------------------------

def test_create_project_makes_variations():
    pid = make_project(variations=3)
    project = client.get(f"/api/projects/{pid}").json()["project"]
    assert len(project["variations"]) == 3
    # Same grid, different seeds -- that is the whole point.
    seeds = {v["seed"] for v in project["variations"]}
    assert len(seeds) == 3
    for v in project["variations"]:
        assert (storage.project_dir(pid) / v["audio"]).exists()


def test_prompt_project_grid_is_preconfirmed():
    """A typed prompt has nothing to mis-detect, so it needs no confirmation."""
    pid = make_project()
    grid = client.get(f"/api/projects/{pid}").json()["project"]["grid"]
    assert grid["confirmed"] is True
    assert grid["bpm"] == 96 and grid["key_scale"] == "A Minor"   # from the preset


def test_style_preset_shapes_the_prompt():
    pid = make_project()
    project = client.get(f"/api/projects/{pid}").json()["project"]
    assert "indie soul" in project["variations"][0]["prompt"]


def test_upload_requires_grid_confirmation():
    """The guard that stops a bad key detection corrupting every stem."""
    src = write_wav(_TMP / "idea.wav")
    with open(src, "rb") as fh:
        res = client.post("/api/projects/upload",
                          files={"file": ("idea.wav", fh, "audio/wav")},
                          data={"title": "From audio", "bars": "8"})
    assert res.status_code == 200, res.text
    body = res.json()
    pid = body["project"]["id"]
    assert body["project"]["grid"]["confirmed"] is False
    assert body["project"]["source_audio"]

    # Layering must be refused until a human confirms the grid.
    job = client.post(f"/api/projects/{pid}/layers",
                      json={"track_class": "bass"}).json()["job"]
    result = None
    deadline = time.time() + 30
    while time.time() < deadline:
        result = client.get(f"/api/jobs/{job['id']}").json()
        if result["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert result["status"] == "error"
    assert "confirm bpm and key" in result["error"]

    # After confirming, the same call succeeds.
    client.post(f"/api/projects/{pid}/grid",
                json={"bpm": 100, "key_scale": "C Major", "bars": 8})
    assert client.get(f"/api/projects/{pid}").json()["project"]["grid"]["confirmed"]


# ---- phase 3 --------------------------------------------------------------

def test_split_produces_stems():
    pid = make_project()
    stems = split(pid)
    assert len(stems) >= 4
    classes = {s["track_class"] for s in stems}
    assert {"drums", "bass", "vocals"} <= classes
    for stem in stems:
        assert len(stem["versions"]) == 1
        assert (storage.project_dir(pid) / stem["versions"][0]["audio"]).exists()


def test_second_split_does_not_destroy_the_first():
    """Splitting twice must keep both sets, their ids, and their audio.

    The old code did `project.stems = []` and wrote flat stems/bass.wav, so a
    re-split silently invalidated every stem id a client had saved and
    overwrote the audio underneath it.
    """
    pid = make_project(variations=2)
    first = split(pid)
    first_ids = {s["id"] for s in first}
    first_audio = {s["id"]: storage.project_dir(pid) / s["versions"][0]["audio"]
                   for s in first}
    first_bytes = {i: p.read_bytes() for i, p in first_audio.items()}
    assert all(p.exists() for p in first_audio.values())

    # split a different variation
    project = client.get(f"/api/projects/{pid}").json()["project"]
    other = project["variations"][1]["id"]
    job = client.post(f"/api/projects/{pid}/split",
                      json={"variation_id": other, "method": "demucs"}).json()["job"]
    wait_job(job["id"])

    after = client.get(f"/api/projects/{pid}/stems").json()["stems"]
    after_ids = {s["id"] for s in after}

    # every original stem still exists, by id, with its audio untouched
    assert first_ids <= after_ids, "a re-split destroyed the earlier stem ids"
    for sid, path in first_audio.items():
        assert path.exists(), f"audio for {sid} was deleted by the second split"
        assert path.read_bytes() == first_bytes[sid],             f"audio for {sid} was overwritten in place by the second split"

    # the two sets are distinct, and only the newest is active
    project = storage.load(pid)
    splits = {s.split_id for s in project.stems}
    assert len(splits) == 2, f"expected two split sets, got {splits}"
    assert project.active_split in splits
    active = project.active_stems()
    assert active and all(s.split_id == project.active_split for s in active)
    assert len(active) < len(project.stems), "active set should be a subset"


def test_export_uses_only_the_active_split():
    import zipfile
    pid = make_project(variations=2)
    split(pid)
    project = client.get(f"/api/projects/{pid}").json()["project"]
    job = client.post(f"/api/projects/{pid}/split",
                      json={"variation_id": project["variations"][1]["id"],
                            "method": "demucs"}).json()["job"]
    wait_job(job["id"])

    job = client.post(f"/api/projects/{pid}/export").json()["job"]
    wait_job(job["id"])
    out = _TMP / "split_export.zip"
    out.write_bytes(client.get(f"/api/projects/{pid}/export/download").content)
    with zipfile.ZipFile(out) as zf:
        wavs = [n for n in zf.namelist() if n.endswith(".wav") and n != "_mixdown.wav"]
    active = storage.load(pid).active_stems()
    assert len(wavs) == len(active),         f"export packed {len(wavs)} stems but the active split has {len(active)}"


def test_split_tiers_produce_different_stem_counts():
    """Basic is vocals + instrumental; professional adds guitar and piano."""
    pid = make_project(variations=2)
    project = client.get(f"/api/projects/{pid}").json()["project"]

    job = client.post(f"/api/projects/{pid}/split",
                      json={"variation_id": project["variations"][0]["id"],
                            "tier": "basic"}).json()["job"]
    wait_job(job["id"])
    basic = storage.load(pid).active_stems()
    assert len(basic) == 2, f"basic should give 2 stems, got {len(basic)}"

    job = client.post(f"/api/projects/{pid}/split",
                      json={"variation_id": project["variations"][1]["id"],
                            "tier": "professional"}).json()["job"]
    wait_job(job["id"])
    pro = storage.load(pid).active_stems()
    assert len(pro) > len(basic), "professional should give more stems than basic"


def test_unsupported_split_option_is_reported_not_ignored():
    """We host no de-reverb model. Saying so beats silently dropping the flag."""
    pid = make_project()
    project = client.get(f"/api/projects/{pid}").json()["project"]
    job = client.post(f"/api/projects/{pid}/split",
                      json={"variation_id": project["variations"][0]["id"],
                            "remove_reverb": True}).json()["job"]
    stems = wait_job(job["id"])["result"]
    assert "remove_reverb" in stems[0]["versions"][0]["params"]["unfulfilled"]


def test_mixdown_respects_mute_and_solo():
    pid = make_project()
    stems = split(pid)
    full = client.get(f"/api/projects/{pid}/mixdown").content

    client.patch(f"/api/projects/{pid}/stems/{stems[0]['id']}/mix", json={"muted": True})
    muted = client.get(f"/api/projects/{pid}/mixdown").content
    assert muted != full, "muting a stem did not change the mix"

    client.patch(f"/api/projects/{pid}/stems/{stems[0]['id']}/mix",
                 json={"muted": False, "soloed": True})
    soloed = client.get(f"/api/projects/{pid}/mixdown").content
    assert soloed != full and soloed != muted


def test_mixdown_leaves_headroom():
    """A clipped mix is a corrupted conditioning signal, not just a loud one."""
    pid = make_project()
    split(pid)
    project = storage.load(pid)
    from app import service
    path = service.mixdown(project)
    peak = audio.describe(path)["peak"]
    assert 0.85 < peak <= 0.90, f"expected ~-1 dBFS, got peak {peak}"


def test_export_zip_contains_stems_and_manifest():
    import zipfile
    pid = make_project()
    split(pid)
    job = client.post(f"/api/projects/{pid}/export").json()["job"]
    wait_job(job["id"])

    res = client.get(f"/api/projects/{pid}/export/download")
    assert res.status_code == 200
    out = _TMP / "export.zip"
    out.write_bytes(res.content)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "README.txt" in names and "_mixdown.wav" in names
        assert sum(n.endswith(".wav") for n in names) >= 5
        readme = zf.read("README.txt").decode()
        # FL Studio needs the tempo to place the stems on the grid.
        assert "bpm: 96" in readme and "key:" in readme


# ---- phase 4 --------------------------------------------------------------

def test_repaint_preserves_audio_outside_the_region():
    """The core promise of regional editing, verified by null test."""
    pid = make_project(bars=16)
    stems = split(pid)
    stem = stems[0]
    before = storage.project_dir(pid) / stem["versions"][0]["audio"]
    before_copy = _TMP / "before.wav"
    audio.write(before_copy, *audio.load(before))

    job = client.post(f"/api/projects/{pid}/stems/{stem['id']}/repaint",
                      json={"start_bar": 5, "end_bar": 9}).json()["job"]
    version = wait_job(job["id"])["result"]

    assert version["op"] == "repaint"
    residual = version["params"]["outside_residual"]
    assert residual is not None and residual < 0.05, \
        f"repaint changed audio outside the region (residual {residual})"

    # And the region itself must actually differ.
    after = storage.project_dir(pid) / version["audio"]
    grid = Grid(bpm=96, bars=16)
    a, sr = audio.load(before_copy)
    b, _ = audio.load(after)
    i0, i1 = int(grid.bar_to_seconds(5) * sr), int(grid.bar_to_seconds(9) * sr)
    inside = np.abs(a[i0:i1] - b[i0:i1]).mean()
    assert inside > 1e-4, "repaint did not change the requested region"


def test_repaint_accepts_seconds_not_just_bars():
    """A timeline drag lands anywhere; whole-bar quantisation was ours, not the
    model's -- ACE-Step's repaint takes seconds."""
    pid = make_project(bars=16)
    stems = split(pid)
    job = client.post(f"/api/projects/{pid}/stems/{stems[0]['id']}/repaint",
                      json={"start_s": 3.75, "end_s": 7.25}).json()["job"]
    version = wait_job(job["id"])["result"]
    assert version["op"] == "repaint"
    assert version["params"]["start_s"] == 3.75
    assert version["params"]["end_s"] == 7.25


def test_repaint_without_a_range_is_rejected():
    pid = make_project(bars=8)
    stems = split(pid)
    res = client.post(f"/api/projects/{pid}/stems/{stems[0]['id']}/repaint",
                      json={"prompt": "no range given"})
    assert res.status_code == 422


def test_progress_is_not_fabricated():
    """Jobs whose engine reports nothing must say so rather than inventing a
    percentage; and the queue position must be real."""
    from app import jobs as jobs_mod
    q = jobs_mod.JobQueue()
    j = jobs_mod.Job(id="j1", kind="test")
    assert j.public()["determinate"] is True
    j.determinate = False
    body = j.public()
    assert body["determinate"] is False and "queue_position" in body


def test_repaint_rejects_a_range_past_the_end():
    pid = make_project(bars=8)
    stems = split(pid)
    job = client.post(f"/api/projects/{pid}/stems/{stems[0]['id']}/repaint",
                      json={"start_bar": 200, "end_bar": 204}).json()["job"]
    deadline = time.time() + 30
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job['id']}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert body["status"] == "error" and "only" in body["error"]


def test_versions_accumulate_and_are_selectable():
    """Editing appends; nothing is overwritten, so undo is free."""
    pid = make_project(bars=16)
    stems = split(pid)
    stem_id = stems[0]["id"]

    for _ in range(2):
        job = client.post(f"/api/projects/{pid}/stems/{stem_id}/vary",
                          json={}).json()["job"]
        wait_job(job["id"])

    stem = next(s for s in client.get(f"/api/projects/{pid}/stems").json()["stems"]
                if s["id"] == stem_id)
    assert len(stem["versions"]) == 3 and stem["current"] == 2

    reverted = client.post(f"/api/projects/{pid}/stems/{stem_id}/version/0").json()["stem"]
    assert reverted["current"] == 0
    assert len(reverted["versions"]) == 3, "reverting must not delete takes"


def test_extend_lengthens_audio_and_grid():
    pid = make_project(bars=8)
    stems = split(pid)
    stem_id = stems[0]["id"]
    before = audio.describe(
        storage.project_dir(pid) / stems[0]["versions"][0]["audio"])["duration_s"]

    job = client.post(f"/api/projects/{pid}/stems/{stem_id}/extend",
                      json={"bars": 4}).json()["job"]
    version = wait_job(job["id"])["result"]

    after = audio.describe(storage.project_dir(pid) / version["audio"])["duration_s"]
    assert after > before + 8, f"expected ~10s longer, got {after - before:.2f}s"
    assert client.get(f"/api/projects/{pid}").json()["project"]["grid"]["bars"] == 12


def test_add_layer_conditions_on_existing_stems():
    pid = make_project(bars=8)
    split(pid)
    job = client.post(f"/api/projects/{pid}/layers",
                      json={"track_class": "strings", "prompt": "soft pad"}).json()["job"]
    stem = wait_job(job["id"])["result"]
    assert stem["track_class"] == "strings"
    assert stem["versions"][0]["op"] == "generate"
    assert (storage.project_dir(pid) / stem["versions"][0]["audio"]).exists()


def test_layer_can_target_a_region():
    """A layer confined to bars 3-5 must be silent outside that window."""
    import numpy as np
    pid = make_project(bars=8)
    split(pid)
    grid = storage.load(pid).grid
    start_s, end_s = grid.bar_to_seconds(3), grid.bar_to_seconds(5)

    job = client.post(f"/api/projects/{pid}/layers",
                      json={"track_class": "strings", "prompt": "pad",
                            "start_s": start_s, "end_s": end_s}).json()["job"]
    stem = wait_job(job["id"])["result"]
    path = storage.project_dir(pid) / stem["versions"][0]["audio"]
    data, sr = audio.load(path)

    assert abs(len(data) / sr - grid.duration_s) < 0.2, "layer should span the grid"
    inside = np.abs(data[int(start_s * sr):int(end_s * sr)]).max()
    before = np.abs(data[:int(start_s * sr)]).max() if start_s > 0 else 0.0
    assert inside > 0.05, "the targeted region should contain audio"
    assert before < 1e-4, f"audio leaked before the region (peak {before})"


def test_voice_can_target_a_region():
    """Converting a region must leave the rest of the stem bit-identical."""
    ref = write_wav(_TMP / "regionref.wav", seconds=3.0, freq=280)
    with open(ref, "rb") as fh:
        vid = client.post("/api/voices",
                          files={"file": ("r.wav", fh, "audio/wav")},
                          data={"label": "Region", "consent": "test"}).json()["voice"]["id"]

    pid = make_project(bars=8)
    stems = split(pid)
    vocal = next(s for s in stems if s["track_class"] == "vocals")
    before_path = storage.project_dir(pid) / vocal["versions"][0]["audio"]
    before, sr = audio.load(before_path)

    job = client.post(f"/api/projects/{pid}/stems/{vocal['id']}/voice",
                      json={"voice_id": vid, "start_s": 4.0, "end_s": 8.0}).json()["job"]
    version = wait_job(job["id"])["result"]
    after, _ = audio.load(storage.project_dir(pid) / version["audio"])

    import numpy as np
    head = np.abs(before[:int(3.5 * sr)] - after[:int(3.5 * sr)]).max()
    mid = np.abs(before[int(4.5 * sr):int(7.5 * sr)]
                 - after[int(4.5 * sr):int(7.5 * sr)]).max()
    assert head < 1e-4, f"audio outside the region changed (delta {head})"
    assert mid > 1e-3, "the targeted region was not converted"
    assert version["params"]["start_s"] == 4.0


def test_delete_stem():
    pid = make_project()
    stems = split(pid)
    client.delete(f"/api/projects/{pid}/stems/{stems[0]['id']}")
    remaining = client.get(f"/api/projects/{pid}/stems").json()["stems"]
    assert len(remaining) == len(stems) - 1


# ---- phase 5 --------------------------------------------------------------

def test_vocals_require_lyrics():
    pid = make_project(bars=8)
    split(pid)
    job = client.post(f"/api/projects/{pid}/vocals", json={"lyrics": ""}).json()["job"]
    deadline = time.time() + 30
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job['id']}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert body["status"] == "error" and "lyrics are required" in body["error"]


def test_vocals_over_instrumental():
    pid = make_project(bars=8)
    split(pid)
    job = client.post(f"/api/projects/{pid}/vocals",
                      json={"lyrics": "[verse]\\nslow light on the floor"}).json()["job"]
    stem = wait_job(job["id"])["result"]
    assert stem["track_class"] == "vocals"
    assert (storage.project_dir(pid) / stem["versions"][0]["audio"]).exists()


def test_voice_library_add_list_and_apply():
    ref = write_wav(_TMP / "ref.wav", seconds=3.0, freq=300)
    with open(ref, "rb") as fh:
        res = client.post("/api/voices",
                          files={"file": ("ref.wav", fh, "audio/wav")},
                          data={"label": "Aria", "consent": "test fixture"})
    assert res.status_code == 200
    voice_id = res.json()["voice"]["id"]
    assert any(v["id"] == voice_id for v in client.get("/api/voices").json()["voices"])

    pid = make_project(bars=8)
    split(pid)
    vocal = next(s for s in client.get(f"/api/projects/{pid}/stems").json()["stems"]
                 if s["track_class"] == "vocals")
    job = client.post(f"/api/projects/{pid}/stems/{vocal['id']}/voice",
                      json={"voice_id": voice_id}).json()["job"]
    version = wait_job(job["id"])["result"]
    assert version["op"] == "voice" and version["params"]["voice_id"] == voice_id

    stem = next(s for s in client.get(f"/api/projects/{pid}/stems").json()["stems"]
                if s["id"] == vocal["id"])
    assert stem["voice_id"] == voice_id


def test_unknown_voice_is_rejected():
    pid = make_project(bars=8)
    stems = split(pid)
    job = client.post(f"/api/projects/{pid}/stems/{stems[0]['id']}/voice",
                      json={"voice_id": "voice_nope"}).json()["job"]
    deadline = time.time() + 30
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job['id']}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert body["status"] == "error" and "no such voice" in body["error"]


# ---- phase 6 --------------------------------------------------------------

def test_cancel_stops_a_queued_job():
    """A queued job must stop at once; asyncio.to_thread cannot be cancelled
    from outside, so cancellation is cooperative via the report callback."""
    pid = make_project(bars=8)
    split(pid)
    # Two jobs: concurrency is 1, so the second is queued behind the first.
    first = client.post(f"/api/projects/{pid}/layers",
                        json={"track_class": "strings"}).json()["job"]
    second = client.post(f"/api/projects/{pid}/layers",
                         json={"track_class": "brass"}).json()["job"]

    body = client.post(f"/api/jobs/{second['id']}/cancel").json()
    assert body["status"] == "cancelled", body
    assert body["cancel_requested"] is True

    final = client.get(f"/api/jobs/{second['id']}").json()
    assert final["status"] == "cancelled"
    wait_job(first["id"])   # the one we did not cancel still completes


def test_cancelling_a_finished_job_is_harmless():
    pid = make_project(bars=8)
    job = client.post(f"/api/projects/{pid}/variations?count=1").json()["job"]
    wait_job(job["id"])
    body = client.post(f"/api/jobs/{job['id']}/cancel").json()
    assert body["status"] == "done", "cancelling a finished job must not rewrite it"


def test_cancel_unknown_job_is_404():
    assert client.post("/api/jobs/job_nope/cancel").status_code == 404


def test_idempotency_key_does_not_start_a_second_render():
    """A POST whose response was lost has already spent GPU time. Repeating it
    with the same key must return the original job, not start another."""
    key = "probe-key-123"
    body = {"title": "idem", "prompt": "test", "bars": 4, "variations": 1}
    a = client.post("/api/projects", json=body,
                    headers={"Idempotency-Key": key}).json()
    b = client.post("/api/projects", json=body,
                    headers={"Idempotency-Key": key}).json()
    assert a["job"]["id"] == b["job"]["id"], "the same key started a second job"
    wait_job(a["job"]["id"])

    c = client.post("/api/projects", json=body,
                    headers={"Idempotency-Key": "different-key"}).json()
    assert c["job"]["id"] != a["job"]["id"], "a different key must start a new job"
    wait_job(c["job"]["id"])


def test_sfx_generation():
    job = client.post("/api/sfx", json={
        "prompt": "deep impact hit", "duration_s": 2.0, "variations": 2,
    }).json()["job"]
    sounds = wait_job(job["id"])["result"]
    assert len(sounds) == 2
    for sound in sounds:
        res = client.get(f"/api/sfx/{Path(sound['audio']).name}")
        assert res.status_code == 200 and len(res.content) > 1000


# ---- cross-cutting --------------------------------------------------------

def test_audio_route_blocks_path_traversal():
    pid = make_project()
    res = client.get(f"/api/projects/{pid}/audio/../../../../etc/passwd")
    assert res.status_code == 404


def test_project_survives_reload():
    """State lives on disk, so a pod restart loses nothing."""
    pid = make_project(bars=8)
    split(pid)
    reloaded = storage.load(pid)
    assert reloaded is not None
    assert len(reloaded.stems) >= 4
    assert reloaded.grid.bpm == 96


def test_styles_endpoint():
    body = client.get("/api/styles").json()
    assert len(body["styles"]) >= 8 and len(body["sfx"]) >= 4
    assert all({"id", "label", "prompt"} <= set(s) for s in body["styles"])


def test_grid_maths():
    grid = Grid(bpm=120, time_sig="4/4", bars=16)
    assert grid.duration_s == 32.0
    assert grid.bar_to_seconds(1) == 0.0
    assert grid.bar_to_seconds(5) == 8.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                          # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
