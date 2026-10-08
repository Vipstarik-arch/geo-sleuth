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
import math
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


class TerrainTileTests(unittest.TestCase):
    """Missing elevation tiles are padded with 0 m, which reads as flat ground at sea level: they must be counted and reported."""

    def _failed_curl(self, returncode=7):
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=b"curl: (7) Failed to connect to s3.amazonaws.com port 443")

    def test_failed_tile_download_reports_curl_reason_and_leaves_no_file(self):
        import terrain
        with tempfile.TemporaryDirectory() as folder, patch.object(terrain.subprocess, "run", return_value=self._failed_curl()):
            cache = Path(folder)
            arr, err = terrain._fetch(13, 100, 200, cache, "direct")
            self.assertEqual(arr.shape, (256, 256))
            self.assertIn("curl: (7)", err)                     # -sS: curl's own reason, not an empty string
            self.assertFalse((cache / "terrarium_13_100_200.png").exists())

    def test_every_tile_missing_exits_instead_of_reporting_a_flat_horizon(self):
        import terrain
        with tempfile.TemporaryDirectory() as folder, patch.object(terrain.subprocess, "run", return_value=self._failed_curl()), \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            with self.assertRaises(SystemExit) as caught:
                terrain.DEM((39.9042, 116.4074), 300, 13, Path(folder), "direct")
        self.assertIn("No elevation tile", str(caught.exception))
        self.assertIn("WARNING terrain:", err.getvalue())
        self.assertIn("curl: (7)", err.getvalue())

    def test_partial_tile_failure_warns_and_is_written_to_the_profile_json(self):
        import numpy as np
        import terrain
        calls = {"n": 0}

        def half_fail(z, x, y, cache, proxy):
            calls["n"] += 1
            if calls["n"] % 2:
                return np.zeros((256, 256), dtype=np.float32), "curl exit 7: Failed to connect"
            return np.full((256, 256), 120.0, dtype=np.float32), ""

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "prof.json"
            argv = ["terrain.py", "--cache", str(Path(folder) / "cache"), "profile", "--at", "39.9042,116.4074",
                    "--heading", "90", "--range", "3000", "--width", "60", "--zoom", "12", "--out", str(out)]
            with patch.object(terrain, "_fetch", side_effect=half_fail), patch.object(sys, "argv", argv), \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
                terrain.main()                                  # must not exit: only some tiles are missing
            self.assertIn("WARNING terrain:", stderr.getvalue())
            self.assertIn("elevation tiles missing", stdout.getvalue())
            meta = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(meta["elevation_tiles"]["total"], calls["n"])
            self.assertGreater(meta["elevation_tiles"]["failed"], 0)
            self.assertLess(meta["elevation_tiles"]["failed"], calls["n"])
            self.assertIn("Failed to connect", meta["elevation_tiles"]["first_error"])

    def test_region_mosaic_counts_tiles_and_stops_when_none_arrives(self):
        import numpy as np
        import terrain
        need = {(1, 1), (1, 2), (2, 1), (2, 2)}
        with tempfile.TemporaryDirectory() as folder, patch.object(
                terrain, "_fetch", return_value=(np.full((256, 256), 500.0, dtype=np.float32), "")), \
                patch("sys.stderr", new_callable=io.StringIO):
            mos = terrain._Mosaic(need, 10, Path(folder), "direct", 2)
        self.assertEqual((mos.total_tiles, mos.failed_tiles), (4, 0))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(terrain, "_fetch", return_value=(np.zeros((256, 256), dtype=np.float32), "curl exit 7: Failed to connect")), \
                patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                terrain._Mosaic(need, 10, Path(folder), "direct", 2)
        self.assertIn("No elevation tile", str(caught.exception))


class PoseSelfCheckTests(unittest.TestCase):
    """The self-check table must use the same projection as the solver: H/2 + f*tan(dep+pitch) ignores the
    horizontal offset and reported 10–50 px mismatches for off-axis points on a perfect solution."""

    def setUp(self):
        import numpy as np
        self.np = np
        import pose
        self.pose = pose
        self.lat0, self.lon0 = 23.6500, 113.0500
        # the camera must NOT sit at the frame origin: sanity() and project() share the frame, and an offset
        # camera is what catches absolute-vs-camera-relative ENU mistakes (the first version of this fix had one)
        self.e0, self.n0 = 836.0, 564.0
        self.params = np.array([self.e0, self.n0, 50.0, 58.0, -4.0, 0.0, 1500.0])
        self.frame = {"lat0": self.lat0, "lon0": self.lon0, "image_size": [1200, 900]}
        self.pose_dict = {"_params": self.params, "_frame": self.frame}

    def _point(self, name, bearing_deg, dist_m, h):
        kx, ky = self.pose._frame(self.lat0)
        e = self.e0 + dist_m * math.sin(math.radians(bearing_deg))
        n = self.n0 + dist_m * math.cos(math.radians(bearing_deg))
        ll = [self.lat0 + n / ky, self.lon0 + e / kx]
        uv, _ = self.pose.project(self.params, self.np.array([[e, n, h]]), 1200, 900)
        return {"name": name, "ll": ll, "h": h, "px": [float(uv[0][0]), float(uv[0][1])]}, (float(uv[0][0]), float(uv[0][1]))

    def test_off_axis_points_are_not_reported_as_mismatched(self):
        # 20° off the optical axis, close by: the deprecated flat formula lands ~30 px away from the true row
        off, _ = self._point("off_axis", 78.0, 140.0, 5.0)
        on, _ = self._point("on_axis", 58.0, 500.0, 40.0)
        lines = self.pose.sanity(self.pose_dict, [on, off])
        rows = {line.split()[0]: line for line in lines if line.startswith("  ")}
        self.assertIn("off by 0 px", rows["on_axis"])
        self.assertIn("off by 0 px", rows["off_axis"])

    def test_the_flat_formula_would_have_flagged_the_off_axis_point(self):
        off, (col, row) = self._point("off_axis", 78.0, 140.0, 5.0)
        self.assertTrue(0 <= col < 1200 and 0 <= row < 900)     # inside the frame, so the old table compared rows
        dep = math.degrees(math.atan2(50.0 - off["h"], 140.0))
        naive = 900 / 2 + 1500.0 * math.tan(math.radians(dep - 4.0))
        self.assertGreater(abs(naive - row), 20.0)              # this gap is what the old table printed as "off by N px"
        line = next(line for line in self.pose.sanity(self.pose_dict, [off]) if line.startswith("  "))
        self.assertIn("off by 0 px", line)

    def test_points_behind_the_camera_are_named_as_such(self):
        behind = {"name": "behind", "ll": [self.lat0 + (self.n0 - 110.0) / 110540.0, self.lon0 + self.e0 / self.pose._frame(self.lat0)[0]],
                  "h": 10.0, "px": [600, 500]}
        line = next(line for line in self.pose.sanity(self.pose_dict, [behind]) if line.startswith("  "))
        self.assertIn("behind the camera", line)


class ScriptBehaviourTests(unittest.TestCase):
    scripts = Path(__file__).resolve().parents[1] / "scripts"

    def test_shared_fetch_reports_curls_reason(self):
        failed = SimpleNamespace(returncode=35, stdout=b"", stderr=b"curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL")
        with patch.object(_net.subprocess, "run", return_value=failed):
            with self.assertRaises(RuntimeError) as caught:
                _net.fetch_bytes("https://example.invalid/x", "direct")
        self.assertIn("curl exit 35", str(caught.exception))
        self.assertIn("SSL_ERROR_SYSCALL", str(caught.exception))

    def test_terrain_tile_requests_keep_curls_error_text(self):
        header = (self.scripts / "terrain.py").read_text(encoding="utf-8")
        self.assertIn('["curl", "-q", "-sS", "-m", "60", "-o", str(p)', header)
        self.assertIn("_report_tiles", header)

    def test_tile_download_failure_is_announced_and_partial_files_are_dropped(self):
        import tiles
        failed = SimpleNamespace(returncode=7, stdout=b"", stderr=b"curl: (7) Failed to connect")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tile.jpg"
            path.write_bytes(b"x" * 500)                         # truncated file from an earlier failed run
            with patch.object(tiles.subprocess, "run", return_value=failed):
                self.assertFalse(tiles._get("https://example.invalid/t.jpg", path, "direct"))
            self.assertFalse(path.exists())

    def test_clue_update_reports_the_curl_reason(self):
        import clues
        failed = SimpleNamespace(returncode=35, stdout=b"", stderr=b"curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL")
        with patch.object(clues.subprocess, "run", return_value=failed):
            with self.assertRaises(SystemExit) as caught:
                clues._fetch("https://example.invalid/x", "direct")
        self.assertIn("curl exit 35", str(caught.exception))
        self.assertIn("SSL_ERROR_SYSCALL", str(caught.exception))

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

    def test_street_view_curl_treats_http_errors_as_failures(self):
        # without -f an HTTP error page is written out as if it were the image, and the tile silently becomes a broken picture
        source = (self.scripts / "gsv.py").read_text(encoding="utf-8")
        self.assertIn('"curl", "-q", "-sS", "-f", "-m", "40"', source.split("def render", 1)[0])


class PierColumnTests(unittest.TestCase):
    """The pier column must be the centre of the bar: the spacing geometry measures span/position from it."""

    def test_flat_topped_bar_reports_its_centre_not_its_right_edge(self):
        import imgprep
        d = [0.0] * 60
        for c in (10, 34):                                  # two 8 px flat-topped piers, as in a real photo
            d[c:c + 8] = [200.0] * 8
        cols = [i for i, _ in imgprep._peaks(d, 1, 20)]
        self.assertEqual(cols, [14, 38])                    # centres 13.5 and 37.5 (banker's rounding)
        self.assertNotIn(17, cols)                          # 17 is the bar's last column, where the old code pointed

    def test_isolated_spike_keeps_its_own_column(self):
        import imgprep
        d = [0.0] * 20
        d[7] = 90.0
        self.assertEqual([i for i, _ in imgprep._peaks(d, 1, 20)], [7])

    def test_piers_on_a_synthetic_deck_land_on_the_bar_centre(self):
        from PIL import Image
        import imgprep
        cols = [40, 120, 200]
        im = Image.new("L", (260, 100), 0)
        for c in cols:
            for x in range(c, c + 10):
                for y in range(50, 100):
                    im.putpixel((x, y), 200)
        res = imgprep.piers(im.convert("RGB"), (50, 100), None, 5, 30, 25, "bright")
        self.assertEqual([p["col"] for p in res["piers"]], [44, 124, 204])   # 44.5, 124.5, 204.5 rounded


class FailedRenderTests(unittest.TestCase):
    """A render that failed must come back as None with a reason: match.py ranks every non-None tile as a candidate."""

    scripts = Path(__file__).resolve().parents[1] / "scripts"

    def test_street_view_render_returns_none_not_a_grey_placeholder(self):
        import gsv
        failed = SimpleNamespace(returncode=35, stdout=b"", stderr=b"curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL")
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gsv.subprocess, "run", return_value=failed), \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertIsNone(gsv.render("CAoSK0l3dummy0000000000", 90.0, 0.0, 90.0, 640, 480, "direct", Path(folder)))
            self.assertEqual(list(Path(folder).iterdir()), [])          # nothing left behind for the next run to open
        self.assertIn("no image rendered", err.getvalue())

    def test_street_view_drops_a_truncated_cache_entry(self):
        import gsv
        with tempfile.TemporaryDirectory() as folder:
            cached = Path(folder) / "CAoSK0l3dummy0000000000_90_0_90_640x480.jpg"
            cached.write_bytes(b"not a jpeg" * 400)                     # > 2000 bytes, but not an image
            failed = SimpleNamespace(returncode=7, stdout=b"", stderr=b"curl: (7) Failed to connect")
            with patch.object(gsv.subprocess, "run", return_value=failed), patch("sys.stderr", new_callable=io.StringIO):
                self.assertIsNone(gsv.render("CAoSK0l3dummy0000000000", 90.0, 0.0, 90.0, 640, 480, "direct", Path(folder)))
            self.assertFalse(cached.exists())

    def test_street_view_sheet_marks_failed_tiles_and_warns(self):
        import gsv
        items = [{"id": "CAoSK0l3dummy0000000000", "heading": 90.0}, {"id": "CAoSseconddummy00000000", "heading": 180.0}]
        with tempfile.TemporaryDirectory() as folder, patch.object(gsv, "render", return_value=None), \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            out = Path(folder) / "sheet.jpg"
            gsv.sheet(items, out, "direct", Path(folder) / "cache")
            self.assertTrue(out.exists())
        self.assertIn("2/2 tiles", err.getvalue())
        self.assertIn("not evidence of no coverage", err.getvalue())

    def test_baidu_render_returns_none_and_says_it_is_not_evidence_of_absence(self):
        import baidu_pano
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(baidu_pano, "_get", side_effect=RuntimeError("Request failed (curl exit 35: SSL_ERROR_SYSCALL)")), \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertIsNone(baidu_pano.render("0123456789abcdef0123456789", 90.0, cache=Path(folder) / "cache", proxy="direct"))
        self.assertIn("not evidence that the panorama is absent", err.getvalue())

    def test_baidu_render_rejects_an_answer_that_is_not_an_image(self):
        import baidu_pano
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(baidu_pano, "_get", return_value=b"<html>captcha</html>"), \
                patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertIsNone(baidu_pano.render("0123456789abcdef0123456789", 90.0, cache=Path(folder) / "cache", proxy="direct"))
        self.assertIn("did not return an image", err.getvalue())

    def test_match_refuses_an_empty_candidate_list_with_a_clear_message(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.new("RGB", (200, 200), (120, 160, 90)).save(folder / "query.jpg")
            (folder / "items.json").write_text("[]", encoding="utf-8")
            result = subprocess.run([sys.executable, str(self.scripts / "match.py"), "rank",
                                     "--query", str(folder / "query.jpg"), "--items", str(folder / "items.json"),
                                     "--out", str(folder / "rank.json")],
                                    capture_output=True, text=True, encoding="utf-8", timeout=300)
        self.assertEqual(result.returncode, 1, result.stderr[-500:])
        self.assertIn("0 candidates", result.stderr)
        self.assertNotIn("IndexError", result.stderr)


if __name__ == "__main__":
    unittest.main()
