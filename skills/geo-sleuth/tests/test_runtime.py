#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy"]
# ///
"""Offline runtime regressions. Run: uv run skills/geo-sleuth/tests/test_runtime.py"""
from __future__ import annotations

import asyncio
import http.server
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _net
import _browser
import baidu_pano
import doctor
import poi
import revimg


class NetworkTests(unittest.TestCase):
    def test_route_precedence(self):
        with patch.dict(os.environ, {"GEO_PROXY": "http://localhost:8123"}, clear=True):
            self.assertEqual(_net.resolve_proxy(), "http://localhost:8123")
            self.assertEqual(_net.resolve_proxy("http://localhost:8234"), "http://localhost:8234")
            self.assertIsNone(_net.resolve_proxy("direct"))
            self.assertIsNone(_net.resolve_proxy(""))
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://localhost:9999"}, clear=True):
            self.assertIsNone(_net.resolve_proxy())

    @unittest.skipUnless(shutil.which("curl"), "curl required")
    def test_curl_reaches_direct_and_explicit_proxy_despite_ambient_settings(self):
        paths = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
                                         "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": "*", "GEO_PROXY": "http://127.0.0.1:1"}):
                self.assertEqual(_net.fetch_bytes(base + "/direct", "direct"), b"ok")
                self.assertEqual(_net.fetch_bytes("http://example.invalid/proxied", base), b"ok")
            self.assertEqual(paths, ["/direct", "http://example.invalid/proxied"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_model_route_overrides_ambient_variables(self):
        with patch.dict(os.environ, {"GEO_PROXY": "http://chosen:8080", "https_proxy": "http://old:8080", "NO_PROXY": "*"}, clear=True):
            _net.model_proxy_env()
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://chosen:8080")
            self.assertNotIn("https_proxy", os.environ)
            self.assertNotIn("NO_PROXY", os.environ)
            _net.model_proxy_env("direct")
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertEqual(os.environ["NO_PROXY"], "*")

    def test_baidu_near_forwards_proxy(self):
        with patch.object(baidu_pano, "fetch_bytes", return_value=b'{"content":null}') as fetch:
            self.assertIsNone(baidu_pano.near(35, 110, proxy="http://chosen:8080"))
            self.assertEqual(fetch.call_args.args[1], "http://chosen:8080")

    def test_all_poi_sources_forward_proxy(self):
        with patch.object(poi, "_curl", return_value='{}') as fetch:
            poi.search_so("school", None, 2, "direct")
            self.assertEqual(fetch.call_args.kwargs["proxy"], "direct")
            poi.search_sug("school", "http://chosen:8080")
            self.assertEqual(fetch.call_args.kwargs["proxy"], "http://chosen:8080")
        with patch.object(poi, "_curl", return_value='[]') as fetch:
            poi.search_osm("school", None, 2, "http://chosen:8080", "us")
            self.assertEqual(fetch.call_args.kwargs["proxy"], "http://chosen:8080")

    def test_intake_forwards_route_to_both_engines(self):
        import intake
        from PIL import Image
        commands = []
        def run(cmd, **kwargs):
            commands.append(cmd)
            return 0, "", ""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / "photo.jpg"
            Image.new("RGB", (100, 100), "white").save(photo)
            # Preparation writes files used by intake; run the actual small local helpers.
            original = intake._run
            def local_or_search(cmd, **kwargs):
                if any(str(part).endswith("revimg.py") for part in cmd):
                    return run(cmd, **kwargs)
                return original([sys.executable, *cmd[2:]], **kwargs)
            with patch.object(intake, "_run", side_effect=local_or_search), patch.object(sys, "argv", ["intake.py", str(photo), "--out-dir", str(root / "out"), "--no-ocr", "--max-variants", "1", "--proxy", "direct"]):
                intake.main()
            self.assertEqual(len(commands), 2)
            for cmd in commands:
                self.assertEqual(cmd[cmd.index("--proxy") + 1], "direct")
            self.assertEqual({cmd[cmd.index("--engines") + 1] for cmd in commands}, {"baidu", "yandex"})


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_chromium_fallback_uses_same_route(self):
        browser = object()
        launch = AsyncMock(side_effect=[RuntimeError("Chrome absent"), browser])
        p = SimpleNamespace(chromium=SimpleNamespace(launch=launch))
        actual, label = await _browser.launch_browser(p, "socks5h://localhost:8123")
        self.assertIs(actual, browser)
        self.assertEqual(label, "Playwright Chromium")
        for call in launch.call_args_list:
            self.assertEqual(call.kwargs["proxy"]["server"], "socks5://localhost:8123")
        self.assertEqual(launch.call_args_list[0].kwargs["channel"], "chrome")
        self.assertNotIn("channel", launch.call_args_list[1].kwargs)

    async def test_explicit_direct_overrides_saved_browser_proxy(self):
        launch = AsyncMock(return_value=object())
        p = SimpleNamespace(chromium=SimpleNamespace(launch=launch))
        with patch.dict(os.environ, {"GEO_PROXY": "http://old:8080"}):
            await _browser.launch_browser(p, "direct")
        self.assertNotIn("proxy", launch.call_args.kwargs)
        self.assertIn("--no-proxy-server", launch.call_args.kwargs["args"])

    async def test_both_browsers_missing_gives_install_fix(self):
        p = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(side_effect=RuntimeError("missing"))))
        with self.assertRaisesRegex(RuntimeError, "uvx playwright install chromium"):
            await _browser.launch_browser(p, "direct")

    async def test_reverse_search_all_engines_get_selected_route(self):
        browser = SimpleNamespace(new_context=AsyncMock(), close=AsyncMock())
        manager = AsyncMock()
        manager.__aenter__.return_value = object()
        # Avoid requiring Playwright in this offline test's environment.
        fake = SimpleNamespace(async_playwright=lambda: manager)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"playwright.async_api": fake}), patch.object(revimg, "launch_browser", AsyncMock(return_value=(browser, "test"))) as launch, patch.object(revimg, "_text", AsyncMock(return_value={"links": []})), patch.object(revimg, "_baidu", AsyncMock(return_value={"links": []})) as baidu, patch.object(revimg, "_yandex", AsyncMock(return_value={"links": []})):
            await revimg.run([Path("photo.jpg")], ["baidu", "yandex"], Path(folder), "direct", ["school"], ["bing"])
            self.assertEqual(launch.await_count, 3)
            self.assertTrue(all(call.args[1] == "direct" for call in launch.await_args_list))
            self.assertEqual(baidu.await_args.args[-1], "direct")


class DoctorTests(unittest.TestCase):
    def test_probe_distinguishes_transport_failure_and_service_block(self):
        for returncode, stdout, status in [(0, "200", "PASS"), (0, "429", "WARN"), (7, "000", "FAIL")]:
            with self.subTest(status=status), patch.object(doctor.subprocess, "run", return_value=SimpleNamespace(returncode=returncode, stdout=stdout)):
                row = doctor.probe(("service", "https://example.invalid"), "direct")
                self.assertEqual(row["status"], status)

    def test_local_checks_do_not_probe_network_or_print_proxy_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(doctor, "browser_check", AsyncMock(return_value=doctor.check("browser", "PASS", "ready"))), patch.object(doctor, "probe") as probe, patch.dict(os.environ, {"GEO_PROXY": "http://user:secret@localhost:8080"}):
            report = doctor.diagnose(False, None)
            probe.assert_not_called()
            self.assertTrue(report["ok"])
            self.assertNotIn("secret", json.dumps(report))
            self.assertEqual(report["connection"], "configured proxy")

    def test_uv_older_than_the_inline_override_floor_is_flagged(self):
        self.assertEqual(doctor.uv_version("uv 0.6.0 (x86_64-unknown-linux-gnu)"), (0, 6, 0))
        self.assertLess(doctor.uv_version("uv 0.5.0"), doctor.UV_MIN)
        self.assertIsNone(doctor.uv_version("command not found"))
        old_uv = SimpleNamespace(returncode=0, stdout=b"uv 0.5.0 (abc)\n", stderr=b"")
        with patch.object(doctor.subprocess, "run", return_value=old_uv), \
                patch.object(doctor, "browser_check", AsyncMock(return_value=doctor.check("browser", "PASS", "ready"))):
            report = doctor.diagnose(False, None)
        uv_row = next(row for row in report["checks"] if row["name"] == "uv")
        self.assertEqual(uv_row["status"], "WARN")
        self.assertIn("older than 0.6", uv_row["detail"])


class LookupTests(unittest.TestCase):
    def test_driving_side_rejects_unknown_word_instead_of_defaulting_to_right(self):
        import clues
        res = clues.lookup_driving_side("日本", None)
        self.assertEqual(res["matches"], [])
        self.assertIn("left", res["note"])

    def test_driving_side_left_right_and_by_country(self):
        import clues
        left = clues.lookup_driving_side("left", None)
        right = clues.lookup_driving_side("right", None)
        self.assertTrue(left["matches"] and all(m["side"] == "left" for m in left["matches"]))
        self.assertTrue(right["matches"] and all(m["side"] == "right" for m in right["matches"]))
        japan = clues.lookup_driving_side(None, "日本")
        self.assertEqual([m["side"] for m in japan["matches"]], ["left"])


class IntakeStatusTests(unittest.TestCase):
    def test_failure_status_keeps_the_fix_hint_at_the_front(self):
        import intake
        message = ("Neither Google Chrome nor Playwright Chromium could start. Run `uvx playwright install chromium` and retry. "
                   + "chromium: BrowserType.launch: Executable doesn't exist at /x/y " * 10)
        brief = intake._brief(message)
        self.assertIn("uvx playwright install chromium", brief)
        self.assertLessEqual(len(brief), 310)
        self.assertEqual(intake._brief("short\n  message"), "short message")
        traceback = ("Traceback (most recent call last):\n  File \"revimg.py\", line 339, in <module>\n    main()\n"
                     "RuntimeError: Neither Google Chrome nor Playwright Chromium could start. Run `uvx playwright install chromium` and retry. "
                     + "chromium: Executable doesn't exist " * 10)
        self.assertIn("uvx playwright install chromium", intake._brief(traceback))
        self.assertNotIn("Traceback", intake._brief(traceback))


class _ZeroScorer:
    def __init__(self, pos, neg):
        pass

    def score(self, ims, multi_scale):
        import numpy as np
        return np.zeros(len(ims))


class SatScanTests(unittest.TestCase):
    def _run_grid(self, cell_image):
        import sat_scan
        args = SimpleNamespace(proxy="direct", cache=None, preset="track", query=None, neg=None, zoom=17, size=320,
                               source="google", seeds=None, multi_scale=False, seed_radius=500, seed_bonus=0.1,
                               top=3, out=None, sheet=None, cols=4, heat=None)
        points = {"r00c00": (23.0, 113.0), "r00c01": (23.0, 113.004)}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}), \
                patch.object(sat_scan, "prefetch", lambda *a, **k: None), \
                patch.object(sat_scan, "cell_image", cell_image), \
                patch.object(sat_scan, "Scorer", _ZeroScorer), \
                patch("sys.stdout", new_callable=io.StringIO), \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            args.cache = folder
            sat_scan.run(points, args)
        return err.getvalue()

    def test_all_blank_cells_say_the_fetch_failed(self):
        from PIL import Image
        err = self._run_grid(lambda *a, **k: Image.new("RGB", (320, 320), "gray"))  # missing tiles stay gray
        self.assertIn("every cell is blank", err)

    def test_real_imagery_gives_no_blank_warning(self):
        import numpy as np
        from PIL import Image
        noise = np.random.default_rng(0).integers(0, 255, (320, 320, 3), dtype=np.uint8)
        err = self._run_grid(lambda *a, **k: Image.fromarray(noise))
        self.assertNotIn("every cell is blank", err)


class ScriptBehaviourTests(unittest.TestCase):
    scripts = Path(__file__).resolve().parents[1] / "scripts"

    def test_exif_help_prints_usage_and_exits_zero(self):
        result = subprocess.run([sys.executable, str(self.scripts / "exif.py"), "--help"],
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Read photo metadata", result.stdout)

    def test_ocr_header_uses_headless_opencv(self):
        # rapidocr-onnxruntime pulls opencv-python, whose import needs libGL.so.1 (absent on minimal Linux)
        header = (self.scripts / "ocr.py").read_text(encoding="utf-8").split('"""', 1)[0]
        self.assertIn('"opencv-python-headless"', header)
        self.assertIn("override-dependencies = [\"opencv-python; sys_platform == 'never'\"]", header)

    def test_overpass_failure_message_carries_the_curl_reason(self):
        import osm
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            closed_port = s.getsockname()[1]  # nothing listens here once the socket is closed
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(osm, "ENDPOINTS", [f"http://127.0.0.1:{closed_port}/api/interpreter"]):
            with self.assertRaises(SystemExit) as caught:
                osm.run("[out:json];node(1);out;", None, Path(folder), timeout=5, rounds=1)
        self.assertIn("curl:", str(caught.exception))

    def test_street_view_failure_is_announced_not_silent(self):
        import gsv
        failed = SimpleNamespace(returncode=7, stdout=b"", stderr=b"curl: (7) Failed to connect")
        with patch.object(gsv.subprocess, "run", return_value=failed), patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(gsv._curl("https://example.invalid/x", "direct"), b"")
        self.assertIn("curl exit 7", err.getvalue())


if __name__ == "__main__":
    unittest.main()
