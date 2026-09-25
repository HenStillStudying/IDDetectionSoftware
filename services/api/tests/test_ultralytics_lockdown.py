"""Importing ultralytics monkey-patches PIL.Image.open process-wide; on any
image that fails to open it used to `pip install pi-heif` from PyPI inside
the running service, triggered by user uploads. ktp_detection/__init__.py
now forces ultralytics offline with auto-install off. These run in a
subprocess (ultralytics reads its settings once, at import) and only where
ultralytics is installed — CI deliberately doesn't install it."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")

ROOT = Path(__file__).resolve().parents[3]


def _run(code: str) -> subprocess.CompletedProcess:
    env_paths = [str(ROOT / "services" / "api"), str(ROOT / "services" / "detection")]
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "PYTHONPATH": __import__("os").pathsep.join(env_paths)},
    )


def test_importing_ktp_detection_turns_ultralytics_auto_install_and_telemetry_off():
    proc = _run("""
        import os
        os.environ.pop("YOLO_OFFLINE", None); os.environ.pop("YOLO_AUTOINSTALL", None)
        import ktp_detection
        from ultralytics.utils import checks
        import ultralytics.utils as u
        print("AUTOINSTALL", checks.AUTOINSTALL, "ONLINE", u.ONLINE)
    """)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "AUTOINSTALL False ONLINE False" in proc.stdout


def test_junk_upload_with_ultralytics_loaded_never_runs_pip():
    # Any pip invocation fails loudly — ultralytics' AutoUpdate installs via
    # subprocess.run / subprocess.check_output (utils/checks.py). Popen is
    # left alone: libraries subclass it at import time.
    proc = _run("""
        import os, subprocess
        os.environ.pop("YOLO_OFFLINE", None); os.environ.pop("YOLO_AUTOINSTALL", None)
        def make_trap(real):
            def trap(*a, **k):
                if "pip" in str(a) + str(k):
                    raise SystemExit("PIP WAS CALLED: " + str(a))
                return real(*a, **k)
            return trap
        subprocess.run = make_trap(subprocess.run)
        subprocess.check_output = make_trap(subprocess.check_output)
        import ktp_detection  # applies ultralytics' Image.open patch, as in the real API
        from app.pipeline import KtpExtractionPipeline
        from app.stub_models import StubDetectionService, StubOcrService
        r = KtpExtractionPipeline(StubDetectionService(), StubOcrService()).run(b"this is not an image")
        print("STATUS", r.status.value)
    """)
    assert "PIP WAS CALLED" not in proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "STATUS invalid_image" in proc.stdout
