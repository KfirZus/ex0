"""
test_Ex0.py - automated tests for Ex0.py, organised exactly like the Software Test
Documentation (STD v1.1.0).  Test IDs from the STD are part of the test names
(e.g. ``test_UT_Parse_04_...`` is UT-Parse-04) and are checked against the
Requirements Traceability Matrix in section 7.

    Section 0  Shared helpers & fixtures
    Section 1  Smoke testing             @smoke        ST-01 .. ST-02
    Section 2  Unit - data parsing       @unit         UT-Parse-01 .. 04  (+ reader / CSV edge cases)
    Section 3  Unit - logic & sampling   @unit         UT-Logic-01 .. 03  (+ geometry, projector, video)
    Section 4  End-to-end                @e2e          E2E-01 .. 02       (+ pipeline / CLI)
    Section 5  Regression                @regression   REG-01 .. 02
    Section 6  Performance/load/stress   @performance  PF-01..02, LD-01..02, STR-01..03
    Section 7  Requirements traceability

Run (from the folder containing Ex0.py):
    python -m pytest                                   everything (spec sizes scaled down)
    python -m pytest -m "smoke or unit"                fast checks
    python -m pytest -m "not performance"              what runs on every commit
    python -m pytest -m performance                    benchmarks / load / stress
    python -m pytest -k UT_Parse_04                    one STD test case
    FULL_SCALE=1 python -m pytest -m performance       real spec sizes (500^3 / 1023^3 / 5000 / 500,000)
                                                       (set the variable with `set` on Windows cmd,
                                                        `$env:FULL_SCALE=1` in PowerShell)

Environment variables
    FULL_SCALE=1            use the sizes written in the STD instead of CI-friendly ones
    PERF_MIN_FPS=30         throughput threshold for PF-01 / LD-01
    PERF_TARGET_HARDWARE=1  also run PF-02 (>= 100 FPS)
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

import Ex0
import generate_fixtures as gf
from Ex0 import (NrrdReader, Plane, PlaneCsvParser, PlaneGeometry,
                 PlaneProjector, VideoRenderer, Volume, main, run_pipeline)

cv2 = pytest.importorskip("cv2")

smoke = pytest.mark.smoke
unit = pytest.mark.unit
e2e = pytest.mark.e2e
regression = pytest.mark.regression
performance = pytest.mark.performance


# =========================================================================== #
# SECTION 0 - SHARED HELPERS & FIXTURES
# =========================================================================== #

FULL_SCALE = os.environ.get("FULL_SCALE") == "1"
N_STD = 500 if FULL_SCALE else 96            # "500^3" volumes of the STD
N_MAX = 1023 if FULL_SCALE else 128          # "1023^3" maximum-size volume
PERF_MIN_FPS = float(os.environ.get("PERF_MIN_FPS", 30))
EX0_PATH = Path(Ex0.__file__).resolve()

NRRD_NAMES = {np.dtype(np.uint8): "uchar", np.dtype(np.uint16): "ushort",
              np.dtype(np.uint32): "uint", np.dtype(np.int8): "int8",
              np.dtype(np.int16): "short", np.dtype(np.int32): "int"}


def available_ram():
    """Free RAM in bytes, or None when it cannot be determined."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def require_ram(nbytes: int) -> None:
    free = available_ram()
    if free is not None and free < nbytes:
        pytest.skip(f"needs ~{nbytes / 1e9:.1f} GB free RAM, only {free / 1e9:.1f} GB available")


def write_nrrd(path: Path, arr: np.ndarray, *, encoding="raw", endian="little",
               type_name=None, extra_header="", byte_skip=0, magic="NRRD0004",
               dimension=None, sizes=None) -> Path:
    """Write `arr` (shape z, y, x) as an NRRD file with optional header tweaks."""
    z, y, x = arr.shape
    base = np.dtype(arr.dtype).newbyteorder("=")
    tname = type_name or NRRD_NAMES[base]
    dim = 3 if dimension is None else dimension
    sz = sizes or f"{x} {y} {z}"
    hdr = f"{magic}\ntype: {tname}\ndimension: {dim}\nsizes: {sz}\n"
    if arr.dtype.itemsize > 1:
        hdr += f"endian: {endian}\n"
    hdr += f"encoding: {encoding}\n{extra_header}\n"
    dt = arr.dtype.newbyteorder("<" if endian == "little" else ">")
    raw = b"\0" * byte_skip + np.ascontiguousarray(arr.astype(dt)).tobytes()
    if encoding == "gzip":
        raw = gzip.compress(raw)
    path.write_bytes(hdr.encode() + raw)
    return path


def ramp_volume(nx, ny, nz, dtype=np.uint16) -> np.ndarray:
    """Volume whose values encode position, so lookups are easy to verify."""
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    if np.dtype(dtype).itemsize >= 4:
        return (z * 10_000 + y * 100 + x).astype(dtype)
    return ((z * 7 + y * 3 + x) % 250 + 1).astype(dtype)


def make_geom(points) -> PlaneGeometry:
    return PlaneGeometry.from_points(np.asarray(points, dtype=np.float64))


def reference_project(M: np.ndarray, geom: PlaneGeometry) -> np.ndarray:
    """Slow, obviously-correct pixel-by-pixel implementation of the algorithm."""
    nz, ny, nx = M.shape
    E = np.zeros((geom.rows, geom.cols), dtype=M.dtype)
    o, u, w = (geom.origin.astype(np.float32), geom.u.astype(np.float32),
               geom.w.astype(np.float32))
    for i in range(geom.rows):
        for j in range(geom.cols):
            p = o + np.float32(i) * u + np.float32(j) * w
            x, y, z = (int(v) for v in np.rint(p))
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                E[i, j] = M[z, y, x]
    return E


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frames, info


def probe_video(path) -> dict:
    """Frame count / codec / playability of an MP4.

    Uses FFmpeg's ffprobe when installed (as the STD asks); otherwise OpenCV only.
    `decoded` is always the number of frames OpenCV could actually decode.
    """
    info = {"decoded": len(read_frames(path)[0]), "codec": None, "frames": None}
    exe = shutil.which("ffprobe")
    if exe:
        out = subprocess.run(
            [exe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames,codec_name,width,height", "-of", "json", str(path)],
            capture_output=True, text=True, check=True).stdout
        stream = json.loads(out)["streams"][0]
        info["codec"] = stream["codec_name"]
        info["frames"] = int(stream["nb_read_frames"])
    return info


def run_cli(*args, cwd=None):
    """Run `python Ex0.py <args>` exactly as a user would."""
    return subprocess.run([sys.executable, str(EX0_PATH), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd or EX0_PATH.parent)


# ---- session-scoped data fixtures (names follow the STD) ------------------- #

@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def fixture_vol_8bit(fixture_dir):
    return gf.write_phantom_nrrd(fixture_dir / "fixture_vol_8bit.nrrd", N_STD, np.uint8)


@pytest.fixture(scope="session")
def fixture_vol_16bit(fixture_dir):
    return gf.write_phantom_nrrd(fixture_dir / "fixture_vol_16bit.nrrd", N_STD, np.uint16)


@pytest.fixture(scope="session")
def fixture_vol_32bit(fixture_dir):
    return gf.write_phantom_nrrd(fixture_dir / "fixture_vol_32bit.nrrd", N_STD, np.uint32)


@pytest.fixture(scope="session")
def fixture_traj_triangle(fixture_dir):
    return gf.write_trajectory(fixture_dir / "fixture_traj_triangle.csv", 100, "triangle", N_STD, seed=1)


@pytest.fixture(scope="session")
def fixture_traj_trapezoid(fixture_dir):
    return gf.write_trajectory(fixture_dir / "fixture_traj_trapezoid.csv", 50, "trapezoid", N_STD, seed=2)


@pytest.fixture(scope="session")
def fixture_malformed_traj(fixture_dir):
    return gf.write_malformed_trajectory(fixture_dir / "fixture_malformed_traj.csv")


@pytest.fixture(scope="session")
def phantom_500_u8():
    """The 500^3 8-bit reference volume (always full size: used by REG-01 and PF-*)."""
    return Volume(gf.build_phantom(500, 500, 500, np.uint8))


@pytest.fixture
def small_volume() -> Volume:
    return Volume(ramp_volume(20, 16, 12, np.uint8))


@pytest.fixture
def sample_files(tmp_path):
    """Small sphere volume + 10 axial planes; used by the pipeline/CLI tests."""
    z, y, x = np.mgrid[0:30, 0:40, 0:50]
    vol = (((x - 25) ** 2 + (y - 20) ** 2 + (z - 15) ** 2) < 12 ** 2).astype(np.uint16) * 50000
    nrrd = write_nrrd(tmp_path / "vol.nrrd", vol, encoding="gzip")
    csv = tmp_path / "planes.csv"
    csv.write_text("\n".join(f"0,0,{k},0,39,{k},49,39,{k}" for k in range(0, 30, 3)) + "\n")
    return nrrd, csv, tmp_path / "out" / "result.mp4"


# =========================================================================== #
# SECTION 1 - SMOKE TESTING (STD 3.1)
# =========================================================================== #

@smoke
class TestSmoke:
    def test_ST_01_basic_cli_invocation(self, sample_files):
        """ST-01: valid NRRD + CSV -> exit code 0 and out.mp4 exists."""
        nrrd, csv, mp4 = sample_files
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 0, r.stderr
        assert mp4.is_file() and mp4.stat().st_size > 0

    def test_ST_02_missing_arguments(self, sample_files):
        """ST-02: `python Ex0.py valid.nrrd` -> non-zero exit and usage text on stderr."""
        nrrd, _, _ = sample_files
        r = run_cli(nrrd)
        assert r.returncode != 0
        assert "usage" in r.stderr.lower()

    def test_ST_02_no_arguments_at_all(self):
        r = run_cli()
        assert r.returncode != 0 and "usage" in r.stderr.lower()

    def test_ST_02_unknown_option_rejected(self, sample_files):
        nrrd, csv, mp4 = sample_files
        r = run_cli(nrrd, csv, mp4, "--bogus")
        assert r.returncode != 0

    def test_smoke_help_flag(self):
        r = run_cli("--help")
        assert r.returncode == 0 and "nrrd" in r.stdout.lower()


# =========================================================================== #
# SECTION 2 - UNIT TESTS: DATA PARSING (STD 3.2.1)
# =========================================================================== #

@unit
class TestDataParsingSTD:
    @pytest.mark.parametrize("fixture_name,dtype", [
        ("fixture_vol_8bit", np.uint8), ("fixture_vol_16bit", np.uint16), ("fixture_vol_32bit", np.uint32)])
    def test_UT_Parse_01_nrrd_matrix_types(self, request, fixture_name, dtype):
        """UT-Parse-01: M is loaded with the right bit depth, shape and contents."""
        path = request.getfixturevalue(fixture_name)
        vol = NrrdReader.read(str(path))
        assert vol.data.dtype == np.dtype(dtype)
        assert vol.data.shape == (N_STD, N_STD, N_STD)
        for z in (0, N_STD // 2, N_STD - 1):            # spot-check contents against the generator
            np.testing.assert_array_equal(vol.data[z], gf.phantom_slice(z, N_STD, N_STD, N_STD, dtype))

    def test_UT_Parse_02_csv_triangle_parse(self, fixture_traj_triangle):
        """UT-Parse-02: 3 points per line -> valid 3-point structures."""
        planes = PlaneCsvParser.parse(str(fixture_traj_triangle))
        assert len(planes) == 100
        for p in planes:
            assert p.points.shape == (3, 3) and np.all(np.isfinite(p.points))
            a, b, c = p.points
            assert np.linalg.norm(b - a) == pytest.approx(np.linalg.norm(c - b), abs=0.05)  # isosceles

    def test_UT_Parse_03_csv_trapezoid_parse(self, fixture_traj_trapezoid):
        """UT-Parse-03: 4 points per line -> valid 4-point structures (isosceles trapezoid)."""
        planes = PlaneCsvParser.parse(str(fixture_traj_trapezoid))
        assert len(planes) == 50
        for p in planes:
            assert p.points.shape == (4, 3) and np.all(np.isfinite(p.points))
            a1, b, a2, c = p.points
            top, base = a2 - a1, c - b
            assert np.linalg.norm(np.cross(top, base)) < 0.05 * np.linalg.norm(top) * np.linalg.norm(base)  # parallel
            assert np.linalg.norm(b - a1) == pytest.approx(np.linalg.norm(c - a2), abs=0.05)               # equal legs

    def test_UT_Parse_04_malformed_csv_handling(self, fixture_malformed_traj, caplog):
        """UT-Parse-04: bad lines are skipped with a warning; nothing crashes."""
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            planes = PlaneCsvParser.parse(str(fixture_malformed_traj))
        assert [p.points.shape[0] for p in planes] == [3, 4, 3]
        assert [p.line_no for p in planes] == [2, 10, 11]
        skipped = {1, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19}   # blank line 6 is silent
        warned = {int(m) for m in re.findall(r"CSV line (\d+) skipped", caplog.text)}
        assert warned == skipped

    def test_UT_Parse_04_malformed_csv_through_cli(self, fixture_malformed_traj, fixture_vol_8bit, tmp_path):
        r = run_cli(fixture_vol_8bit, fixture_malformed_traj, tmp_path / "o.mp4")
        assert r.returncode == 0 and "Traceback" not in r.stderr
        assert probe_video(tmp_path / "o.mp4")["decoded"] == 3


@unit
class TestNrrdReader:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    @pytest.mark.parametrize("encoding", ["raw", "gzip"])
    def test_roundtrip_unsigned_types(self, tmp_path, dtype, encoding):
        arr = ramp_volume(7, 5, 4, dtype)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, encoding=encoding)))
        assert vol.data.shape == (4, 5, 7)
        assert vol.data.dtype.itemsize == np.dtype(dtype).itemsize
        np.testing.assert_array_equal(vol.data, arr)

    @pytest.mark.parametrize("endian", ["little", "big"])
    def test_endianness(self, tmp_path, endian):
        arr = np.arange(24, dtype=np.uint16).reshape(2, 3, 4) * 1000
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, endian=endian)))
        np.testing.assert_array_equal(vol.data, arr)

    @pytest.mark.parametrize("encoding", ["raw", "gzip"])
    def test_big_endian_32bit(self, tmp_path, encoding):
        arr = (np.arange(60, dtype=np.uint32).reshape(3, 4, 5) * 70_000_000)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, endian="big", encoding=encoding)))
        np.testing.assert_array_equal(vol.data, arr)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
    def test_signed_types_tolerated(self, tmp_path, dtype):
        arr = (np.arange(24).reshape(2, 3, 4) - 12).astype(dtype)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr)))
        np.testing.assert_array_equal(vol.data, arr)

    @pytest.mark.parametrize("name,expected", [
        ("unsigned char", np.uint8), ("uint8_t", np.uint8),
        ("unsigned short", np.uint16), ("uint16_t", np.uint16),
        ("unsigned int", np.uint32), ("uint32_t", np.uint32)])
    def test_type_aliases(self, tmp_path, name, expected):
        arr = np.arange(24, dtype=expected).reshape(2, 3, 4)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, type_name=name)))
        assert vol.data.dtype.itemsize == np.dtype(expected).itemsize
        np.testing.assert_array_equal(vol.data, arr)

    def test_axis_order_x_is_fastest(self, tmp_path):
        arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)   # z=2, y=3, x=4
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr)))
        assert vol.shape_xyz == (4, 3, 2)
        np.testing.assert_array_equal(vol.data.reshape(-1)[:4], [0, 1, 2, 3])
        assert vol.data[1, 2, 3] == 23

    def test_result_is_c_contiguous(self, tmp_path):
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", ramp_volume(6, 5, 4, np.uint8))))
        assert vol.data.flags["C_CONTIGUOUS"]

    def test_gzip_result_is_writable_independent_array(self, tmp_path):
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", ramp_volume(6, 5, 4, np.uint8), encoding="gzip")))
        assert vol.data.flags["WRITEABLE"]

    def test_gzip_larger_than_read_chunk(self, tmp_path):
        """Volume bigger than the 16 MB decompression slice exercises the chunk loop."""
        arr = ramp_volume(260, 260, 260, np.uint8)             # 17.6 MB
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, encoding="gzip")))
        np.testing.assert_array_equal(vol.data, arr)

    def test_comments_and_key_value_lines_ignored(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        extra = "# a comment\nmyKey:=myValue\nspace: left-posterior-superior\n"
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, extra_header=extra)))
        np.testing.assert_array_equal(vol.data, arr)

    def test_type_names_case_insensitive(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, type_name="UChar")))
        np.testing.assert_array_equal(vol.data, arr)

    @pytest.mark.parametrize("encoding", ["raw", "gzip"])
    def test_byte_skip(self, tmp_path, encoding):
        arr = ramp_volume(4, 4, 4, np.uint8)
        p = write_nrrd(tmp_path / "v.nrrd", arr, encoding=encoding, extra_header="byte skip: 10\n", byte_skip=10)
        np.testing.assert_array_equal(NrrdReader.read(str(p)).data, arr)

    def test_extra_trailing_bytes_ignored_for_raw(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        p = write_nrrd(tmp_path / "v.nrrd", arr)
        with open(p, "ab") as f:
            f.write(b"\xff" * 50)
        np.testing.assert_array_equal(NrrdReader.read(str(p)).data, arr)

    # ---- error handling ---- #
    def test_bad_magic(self, tmp_path):
        p = write_nrrd(tmp_path / "v.nrrd", ramp_volume(3, 3, 3, np.uint8), magic="PNG0004")
        with pytest.raises(ValueError, match="not an NRRD"):
            NrrdReader.read(str(p))

    def test_missing_sizes(self, tmp_path):
        p = tmp_path / "v.nrrd"
        p.write_bytes(b"NRRD0004\ntype: uchar\nencoding: raw\n\n" + bytes(8))
        with pytest.raises(ValueError, match="sizes"):
            NrrdReader.read(str(p))

    def test_missing_type(self, tmp_path):
        p = tmp_path / "v.nrrd"
        p.write_bytes(b"NRRD0004\nsizes: 2 2 2\nencoding: raw\n\n" + bytes(8))
        with pytest.raises(ValueError, match="type"):
            NrrdReader.read(str(p))

    def test_unsupported_type(self, tmp_path):
        p = tmp_path / "v.nrrd"
        p.write_bytes(b"NRRD0004\ntype: float\ndimension: 3\nsizes: 2 2 2\nencoding: raw\n\n" + bytes(32))
        with pytest.raises(ValueError, match="Unsupported NRRD type"):
            NrrdReader.read(str(p))

    def test_unsupported_encoding(self, tmp_path):
        p = tmp_path / "v.nrrd"
        p.write_bytes(b"NRRD0004\ntype: uchar\ndimension: 3\nsizes: 2 2 2\nencoding: bzip2\n\n" + bytes(8))
        with pytest.raises(ValueError, match="Unsupported NRRD encoding"):
            NrrdReader.read(str(p))

    @pytest.mark.parametrize("dim,sizes", [(2, "4 4"), (4, "2 2 2 2")])
    def test_non_3d_rejected(self, tmp_path, dim, sizes):
        p = tmp_path / "v.nrrd"
        p.write_bytes(f"NRRD0004\ntype: uchar\ndimension: {dim}\nsizes: {sizes}\nencoding: raw\n\n".encode() + bytes(16))
        with pytest.raises(ValueError, match="3D"):
            NrrdReader.read(str(p))

    def test_detached_data_file_rejected(self, tmp_path):
        p = tmp_path / "v.nrrd"
        p.write_bytes(b"NRRD0004\ntype: uchar\ndimension: 3\nsizes: 2 2 2\nencoding: raw\ndata file: v.raw\n\n")
        with pytest.raises(ValueError, match="Detached"):
            NrrdReader.read(str(p))

    def test_truncated_raw_data(self, tmp_path):
        p = write_nrrd(tmp_path / "v.nrrd", ramp_volume(6, 6, 6, np.uint8))
        p.write_bytes(p.read_bytes()[:-50])
        with pytest.raises(ValueError, match="expected"):
            NrrdReader.read(str(p))

    def test_truncated_gzip_data(self, tmp_path):
        p = write_nrrd(tmp_path / "v.nrrd", ramp_volume(20, 20, 20, np.uint16), encoding="gzip")
        p.write_bytes(p.read_bytes()[:-30])
        with pytest.raises(Exception):
            NrrdReader.read(str(p))

    def test_gzip_with_too_little_data(self, tmp_path):
        arr = ramp_volume(6, 6, 6, np.uint8)
        p = write_nrrd(tmp_path / "v.nrrd", arr, encoding="gzip", sizes="6 6 7")   # header lies about z
        with pytest.raises(ValueError, match="expected"):
            NrrdReader.read(str(p))

    def test_oversized_header_fails_before_allocating(self, tmp_path):
        """Header claims 2048^3 uint32 (32 GiB) but the file is tiny: must fail fast, not allocate."""
        p = gf.write_oversized_stub(tmp_path / "huge.nrrd")
        t = time.perf_counter()
        with pytest.raises(ValueError, match="expected"):
            NrrdReader.read(str(p))
        assert time.perf_counter() - t < 2

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            NrrdReader.read(str(tmp_path / "nope.nrrd"))

    def test_dtype_helper_default_little_endian(self):
        dt = NrrdReader._dtype({"type": "ushort"})
        assert dt.itemsize == 2 and dt.byteorder in ("<", "=")

    def test_dtype_helper_big_endian(self):
        assert NrrdReader._dtype({"type": "ushort", "endian": "big"}).byteorder == ">"


@unit
class TestPlaneCsvParser:
    TRI = "0,0,0,0,10,0,20,10,0"
    TRAP = "5,0,0,0,10,0,15,0,0,20,10,0"

    def test_parse_line_triangle(self):
        p = PlaneCsvParser.parse_line(self.TRI, 1)
        assert p.points.shape == (3, 3) and p.line_no == 1
        np.testing.assert_array_equal(p.points[2], [20, 10, 0])

    def test_parse_line_trapezoid(self):
        assert PlaneCsvParser.parse_line(self.TRAP).points.shape == (4, 3)

    def test_floats_negatives_exponent(self):
        p = PlaneCsvParser.parse_line("-1.5,2e1,0.25, 0,0,0, 1,1,1")
        np.testing.assert_allclose(p.points[0], [-1.5, 20, 0.25])

    @pytest.mark.parametrize("line", ["", "   ", "\n", "# comment", "  # indented comment"])
    def test_blank_and_comment_lines_return_none(self, line):
        assert PlaneCsvParser.parse_line(line) is None

    @pytest.mark.parametrize("sep_line", [
        "0;0;0;0;10;0;20;10;0", "0\t0\t0\t0\t10\t0\t20\t10\t0", "0, 0 ,0,0,10,0,20 ,10,0"])
    def test_alternative_separators_and_spaces(self, sep_line):
        assert PlaneCsvParser.parse_line(sep_line).points.shape == (3, 3)

    def test_trailing_comma_tolerated(self):
        assert PlaneCsvParser.parse_line(self.TRI + ",").points.shape == (3, 3)

    @pytest.mark.parametrize("n", [0, 1, 6, 8, 10, 11, 13, 15])
    def test_wrong_value_count_skipped(self, n, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line(",".join(["1"] * n), 3) is None
        if n:
            assert "line 3" in caplog.text

    def test_header_line_skipped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line("ax,ay,az,bx,by,bz,cx,cy,cz", 1) is None
        assert "non-numeric" in caplog.text

    def test_warn_false_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line("junk,line", 1, warn=False) is None
        assert caplog.text == ""

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_non_finite_values_skipped(self, bad):
        assert PlaneCsvParser.parse_line(f"{bad},0,0,0,10,0,20,10,0") is None

    def test_null_byte_line_skipped(self):
        assert PlaneCsvParser.parse_line("1,2,\x00,4,5,6,7,8,9") is None

    def test_parse_file_keeps_order_and_skips_bad(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ax,ay\n" + self.TRI + "\n\n# c\n1,2,3\n" + self.TRAP + "\n")
        planes = PlaneCsvParser.parse(str(f))
        assert [p.points.shape[0] for p in planes] == [3, 4]
        assert [p.line_no for p in planes] == [2, 6]

    def test_parse_file_with_bom_and_crlf(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_bytes(b"\xef\xbb\xbf" + (self.TRI + "\r\n" + self.TRI + "\r\n").encode())
        assert len(PlaneCsvParser.parse(str(f))) == 2

    def test_parse_file_without_trailing_newline(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text(self.TRI)
        assert len(PlaneCsvParser.parse(str(f))) == 1

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("")
        with pytest.raises(ValueError, match="no valid planes"):
            PlaneCsvParser.parse(str(f))

    def test_only_invalid_lines_raises(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("a,b,c\n1,2\n")
        with pytest.raises(ValueError, match="no valid planes"):
            PlaneCsvParser.parse(str(f))

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            PlaneCsvParser.parse(str(tmp_path / "nope.csv"))

    def test_iter_planes_is_a_lazy_generator(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text((self.TRI + "\n") * 10)
        it = PlaneCsvParser.iter_planes(str(f))
        assert inspect.isgenerator(it)
        assert next(it).line_no == 1 and next(it).line_no == 2

    def test_iter_planes_empty_file_yields_nothing(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("")
        assert list(PlaneCsvParser.iter_planes(str(f))) == []

    def test_large_file(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text((self.TRI + "\n") * 5000)
        assert len(PlaneCsvParser.parse(str(f))) == 5000


@unit
class TestCoplanarity:
    """A trapezoid's 4 points must lie in one plane (a'' within COPLANAR_TOL of the plane of a', b, c)."""

    @staticmethod
    def line(dz: float = 0.0) -> str:
        """Trapezoid a'(5,0,0) b(0,10,0) a''(15,0,dz) c(20,10,0): a'' is exactly |dz| off the z=0 plane."""
        return f"5,0,0,0,10,0,15,0,{dz},20,10,0"

    def test_tolerance_constant(self):
        assert Ex0.COPLANAR_TOL == 0.5

    def test_coplanar_trapezoid_accepted(self):
        assert PlaneCsvParser.parse_line(self.line(0)).points.shape == (4, 3)

    @pytest.mark.parametrize("dz", [0.0, 0.1, 0.4, 0.5, -0.5])
    def test_within_tolerance_accepted(self, dz):
        assert PlaneCsvParser.parse_line(self.line(dz)) is not None

    @pytest.mark.parametrize("dz", [0.51, 0.6, 1, 3, 15, -0.6, -15, 1000])
    def test_outside_tolerance_rejected(self, dz):
        assert PlaneCsvParser.parse_line(self.line(dz)) is None

    def test_off_plane_distance_is_geometric_distance(self):
        pts = PlaneCsvParser.parse_line(self.line(0.5)).points
        assert PlaneCsvParser.off_plane_distance(pts) == pytest.approx(0.5)
        pts = np.array([[5, 0, 0], [0, 10, 0], [15, 0, 3.0], [20, 10, 0]], float)
        assert PlaneCsvParser.off_plane_distance(pts) == pytest.approx(3.0)

    def test_distance_is_measured_perpendicular_to_a_tilted_plane(self):
        """Plane z = x (45 degrees): moving a'' along the plane normal by d gives distance d."""
        a1, b, c = np.array([5.0, 0, 5]), np.array([0.0, 10, 0]), np.array([20.0, 10, 20])
        a2 = a1 + (c - b)                                       # coplanar completion
        n = np.cross(b - a1, c - b)
        n /= np.linalg.norm(n)
        for d in (0.0, 0.3, 2.5):
            assert PlaneCsvParser.off_plane_distance(np.array([a1, b, a2 + n * d, c])) == pytest.approx(d)

    @pytest.mark.parametrize("seed", range(20))
    def test_generated_isosceles_trapezoids_in_any_orientation_are_accepted(self, seed):
        rng = np.random.default_rng(seed)
        line = gf._fmt(gf.make_trapezoid(rng, 200))             # CSV text, rounded to 3 decimals
        assert PlaneCsvParser.parse_line(line) is not None

    def test_triangles_are_never_checked_for_coplanarity(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert PlaneCsvParser.parse_line(",".join(f"{v:.3f}" for v in rng.uniform(-100, 100, 9))) is not None

    def test_collinear_base_points_have_no_plane_to_compare_with(self):
        pts = np.array([[0, 0, 0], [5, 5, 5], [7, 7, 9], [10, 10, 10]], float)   # a', b, c on one line
        assert PlaneCsvParser.off_plane_distance(pts) == 0.0                     # no crash / division by zero
        assert PlaneCsvParser.parse_line("0,0,0,5,5,5,7,7,9,10,10,10") is None   # ...but rejected as degenerate

    def test_rejection_is_logged_with_line_number_and_distance(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line(self.line(15), 4) is None
        assert "line 4 skipped" in caplog.text and "not coplanar" in caplog.text and "15.00" in caplog.text

    def test_warn_false_is_silent_for_rejected_trapezoid(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line(self.line(15), 4, warn=False) is None
        assert caplog.text == ""

    def test_file_keeps_good_planes_around_a_rejected_one(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("\n".join([self.line(0), self.line(15), self.line(0.2)]) + "\n")
        planes = PlaneCsvParser.parse(str(f))
        assert [p.line_no for p in planes] == [1, 3]

    def test_file_with_only_non_coplanar_trapezoids_raises(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text(self.line(15) + "\n" + self.line(-9) + "\n")
        with pytest.raises(ValueError, match="no valid planes"):
            PlaneCsvParser.parse(str(f))


@unit
class TestDegeneratePlanes:
    """Identical / collinear points do not span a plane: the CSV line is skipped, not rendered."""

    @pytest.mark.parametrize("line", [
        "5,5,5,5,5,5,5,5,5",                  # all three points identical
        "5,5,5,5,5,5,9,9,5",                  # a == b  (zero-length first edge)
        "5,5,5,9,9,5,9,9,5",                  # b == c  (zero-length second edge)
        "0,0,0,5,5,5,0,0,0",                  # c == a  (edges point back and forth along one line)
        "0,0,0,5,5,5,10,10,10",               # three collinear points
        "0,0,0,5,5,5,2,2,2",                  # collinear, c between a and b
        "0,0,0,10,0,0,20,0.3,0",              # almost collinear: only 0.3 voxels thick
        "0,0,0,0.4,0,0,0.4,10,0",             # first edge only 0.4 voxels long
        "0,0,0,10,0,0,10,0.4,0",              # second edge only 0.4 voxels long
        "0,0,0,5,5,5,7,7,9,10,10,10",         # trapezoid whose base points a', b, c are collinear
        "5,0,0,5,0,0,15,0,0,20,10,0",         # trapezoid with a' == b
    ])
    def test_degenerate_line_is_skipped(self, line):
        assert PlaneCsvParser.parse_line(line, 1) is None

    @pytest.mark.parametrize("line", [
        "0,0,0,10,0,0,10,0.5,0",              # second edge exactly at the 0.5 minimum
        "0,0,0,10,0,0,20,0.6,0",              # 0.6 voxels thick
        "0,0,0,0.5,0,0,0.5,10,0",             # first edge exactly 0.5
        "0,0,0,1,0,0,1,1,0",                  # tiny but real (1 x 1)
        "-5,-5,-5,-5,10,-5,20,10,-5",
    ])
    def test_thin_but_real_plane_is_accepted(self, line):
        assert PlaneCsvParser.parse_line(line, 1) is not None

    def test_degeneracy_reports_the_reason(self):
        assert "edge is only" in PlaneCsvParser.degeneracy(np.array([[1, 1, 1]] * 3, float))
        assert "one line" in PlaneCsvParser.degeneracy(np.array([[0, 0, 0], [5, 5, 5], [10, 10, 10]], float))
        assert PlaneCsvParser.degeneracy(np.array([[0, 0, 0], [0, 10, 0], [20, 10, 0]], float)) is None

    def test_rejection_is_logged_with_line_number_and_reason(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line("5,5,5,5,5,5,5,5,5", 7) is None
        assert "line 7 skipped" in caplog.text and "degenerate plane" in caplog.text

    def test_warn_false_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert PlaneCsvParser.parse_line("5,5,5,5,5,5,5,5,5", 7, warn=False) is None
        assert caplog.text == ""

    def test_generated_planes_are_never_degenerate(self):
        rng = np.random.default_rng(3)
        for make in (gf.make_triangle, gf.make_trapezoid):
            for _ in range(200):
                assert PlaneCsvParser.parse_line(gf._fmt(make(rng, 100))) is not None

    @pytest.mark.parametrize("bad", [
        "(0,0,0),(0,10,0),(20,10,0)",         # parentheses
        '"0","0","0","0","10","0","20","10","0"',   # quoted numbers
        "0,0,0,0,10,0,20,10,0 # note",        # trailing text
        ",,,,,,,,,",                          # only commas
        "0,0,0,0,10,0,20,10,1,5",             # decimal comma (10 values)
    ])
    def test_other_badly_typed_lines_are_skipped(self, bad):
        assert PlaneCsvParser.parse_line(bad, 1) is None

    def test_file_keeps_good_planes_around_degenerate_ones(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("0,0,0,0,10,0,20,10,0\n5,5,5,5,5,5,5,5,5\n0,0,0,5,5,5,10,10,10\n1,1,1,1,11,1,21,11,1\n")
        assert [p.line_no for p in PlaneCsvParser.parse(str(f))] == [1, 4]

    def test_file_with_only_degenerate_planes_raises(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("5,5,5,5,5,5,5,5,5\n0,0,0,5,5,5,10,10,10\n")
        with pytest.raises(ValueError, match="no valid planes"):
            PlaneCsvParser.parse(str(f))


# =========================================================================== #
# SECTION 3 - UNIT TESTS: ALGORITHMIC LOGIC & SAMPLING (STD 3.2.2)
# =========================================================================== #

@unit
class TestAlgorithmSTD:
    def test_UT_Logic_01_vector_normalization(self):
        """UT-Logic-01: direction vectors have length exactly 1.0 (pytest.approx)."""
        g = make_geom([[0, 0, 0], [3, 4, 0], [3, 4, 12]])
        np.testing.assert_allclose(g.u, [0.6, 0.8, 0.0], atol=1e-6)
        np.testing.assert_allclose(g.w, [0.0, 0.0, 1.0], atol=1e-6)
        assert np.linalg.norm(g.u) == pytest.approx(1.0)
        assert np.linalg.norm(g.w) == pytest.approx(1.0)

    def test_UT_Logic_01_vector_normalization_random_planes(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            g = make_geom(rng.uniform(-1000, 1000, (3, 3)))
            assert np.linalg.norm(g.u) == pytest.approx(1.0, abs=1e-6)
            assert np.linalg.norm(g.w) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("seed", range(10))
    def test_UT_Logic_02_matrix_E_dimensions(self, seed):
        """UT-Logic-02: E.shape == (Dab, Dbc), computed independently with math.dist."""
        rng = np.random.default_rng(seed)
        pts = rng.uniform(10, 90, (3, 3))
        vol = Volume(np.ones((100, 100, 100), np.uint8))
        E = PlaneProjector(vol).project(PlaneGeometry.from_points(pts))
        dab, dbc = math.dist(pts[0], pts[1]), math.dist(pts[1], pts[2])
        assert E.shape == (max(1, round(dab)), max(1, round(dbc)))

    def test_UT_Logic_02_matrix_E_dimensions_axis_aligned(self, small_volume):
        E = PlaneProjector(small_volume).project(make_geom([[0, 0, 3], [0, 10, 3], [15, 10, 3]]))
        assert E.shape == (10, 15)

    def test_UT_Logic_02_trapezoid_dimensions(self):
        vol = Volume(np.ones((40, 40, 40), np.uint8))
        E = PlaneProjector(vol).project(make_geom([[5, 0, 5], [0, 10, 5], [15, 0, 5], [20, 10, 5]]))
        assert E.shape == (round(math.hypot(5, 10)), 20)

    def test_UT_Logic_03_out_of_bounds_padding(self):
        """UT-Logic-03: no IndexError beyond the (max-size) volume; padding is exactly 0."""
        n = N_MAX
        require_ram(n ** 3 * 2)
        M = np.full((n, n, n), 200, dtype=np.uint8)
        # plane straddling the +x/+y faces and the -z face of the volume
        E = PlaneProjector(Volume(M)).project(
            make_geom([[n - 10, n - 10, 0], [n - 10, n + 20, 0], [n + 20, n + 20, 0]]))
        assert E.shape == (30, 30)
        i, j = np.mgrid[0:30, 0:30]
        inside = (n - 10 + i <= n - 1) & (n - 10 + j <= n - 1)
        assert np.all(E[inside] == 200)
        assert np.all(E[~inside] == 0)                       # exactly 0, nothing else
        assert inside.sum() == 10 * 10

    def test_UT_Logic_03_out_of_bounds_padding_negative_side(self):
        M = np.full((20, 20, 20), 9, np.uint8)
        E = PlaneProjector(Volume(M)).project(make_geom([[-10, -10, 5], [-10, 10, 5], [10, 10, 5]]))
        assert np.all(E[:10, :] == 0) and np.all(E[:, :10] == 0)
        assert np.all(E[10:, 10:] == 9)


@unit
class TestDataClasses:
    def test_volume_shape_xyz_is_reversed_numpy_shape(self):
        assert Volume(np.zeros((5, 6, 7), np.uint8)).shape_xyz == (7, 6, 5)

    def test_volume_is_frozen(self):
        v = Volume(np.zeros((2, 2, 2), np.uint8))
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.data = np.ones((2, 2, 2), np.uint8)

    def test_plane_stores_points_and_line_number(self):
        p = Plane(np.zeros((3, 3)), 4)
        assert p.points.shape == (3, 3) and p.line_no == 4

    def test_plane_default_line_number(self):
        assert Plane(np.zeros((3, 3))).line_no == 0

    def test_plane_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Plane(np.zeros((3, 3))).line_no = 9


@unit
class TestPlaneGeometry:
    def test_triangle_basic(self):
        g = make_geom([[0, 0, 0], [0, 10, 0], [20, 10, 0]])
        assert (g.rows, g.cols) == (10, 20)
        np.testing.assert_allclose(g.origin, [0, 0, 0])
        np.testing.assert_allclose(g.u, [0, 1, 0])
        np.testing.assert_allclose(g.w, [1, 0, 0])

    def test_trapezoid_uses_first_second_and_last_points(self):
        g = make_geom([[5, 0, 0], [0, 10, 0], [15, 0, 0], [20, 10, 0]])
        np.testing.assert_allclose(g.origin, [5, 0, 0])
        assert g.rows == round(math.hypot(5, 10)) and g.cols == 20

    def test_vectors_are_unit_length(self):
        g = make_geom([[1, 2, 3], [4, 6, 3], [4, 6, 15]])
        assert np.linalg.norm(g.u) == pytest.approx(1, abs=1e-6)
        assert np.linalg.norm(g.w) == pytest.approx(1, abs=1e-6)

    @pytest.mark.parametrize("dab,expected", [(0.4, 1), (0.5, 1), (3.4, 3), (3.6, 4)])
    def test_rows_rounded_and_at_least_one(self, dab, expected):
        assert make_geom([[0, 0, 0], [dab, 0, 0], [dab, 5, 0]]).rows == expected

    def test_degenerate_zero_length_edges_do_not_crash(self):
        g = make_geom([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
        assert g.rows == 1 and g.cols == 1
        assert np.all(np.isfinite(g.u)) and np.all(np.isfinite(g.w))

    def test_oblique_plane_length(self):
        g = make_geom([[0, 0, 0], [3, 4, 0], [3, 4, 12]])
        assert g.rows == 5 and g.cols == 12

    def test_dtypes_are_float32(self):
        g = make_geom([[0, 0, 0], [0, 4, 0], [4, 4, 0]])
        assert g.origin.dtype == g.u.dtype == g.w.dtype == np.float32

    def test_is_frozen(self):
        g = make_geom([[0, 0, 0], [0, 4, 0], [4, 4, 0]])
        with pytest.raises(dataclasses.FrozenInstanceError):
            g.rows = 3


@unit
class TestProjectorSanity:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int16])
    def test_dtype_preserved(self, dtype):
        vol = Volume(ramp_volume(8, 8, 8, dtype))
        assert PlaneProjector(vol).project(make_geom([[0, 0, 0], [0, 5, 0], [5, 5, 0]])).dtype == vol.data.dtype

    def test_uint32_large_values_not_truncated(self):
        data = np.full((6, 6, 6), 4_000_000_000, dtype=np.uint32)
        E = PlaneProjector(Volume(data)).project(make_geom([[0, 0, 2], [0, 4, 2], [4, 4, 2]]))
        assert np.all(E == 4_000_000_000)

    def test_axial_slice_equals_numpy_slice(self, small_volume):
        M = small_volume.data
        nz, ny, nx = M.shape
        E = PlaneProjector(small_volume).project(make_geom([[0, 0, 5], [0, ny - 1, 5], [nx - 1, ny - 1, 5]]))
        np.testing.assert_array_equal(E, M[5, :ny - 1, :nx - 1])

    def test_coronal_slice_equals_numpy_slice(self, small_volume):
        M = small_volume.data
        nz, ny, nx = M.shape
        E = PlaneProjector(small_volume).project(make_geom([[0, 4, 0], [0, 4, nz - 1], [nx - 1, 4, nz - 1]]))
        np.testing.assert_array_equal(E, M[:nz - 1, 4, :nx - 1])

    def test_sagittal_slice_equals_numpy_slice(self, small_volume):
        M = small_volume.data
        nz, ny, nx = M.shape
        E = PlaneProjector(small_volume).project(make_geom([[7, 0, 0], [7, 0, nz - 1], [7, ny - 1, nz - 1]]))
        np.testing.assert_array_equal(E, M[:nz - 1, :ny - 1, 7])

    def test_plane_fully_outside_is_all_zero(self, small_volume):
        E = PlaneProjector(small_volume).project(make_geom([[100, 100, 100], [100, 110, 100], [110, 110, 100]]))
        assert E.shape == (10, 10) and not E.any()

    def test_projection_does_not_modify_volume(self, small_volume):
        before = small_volume.data.copy()
        PlaneProjector(small_volume).project(make_geom([[-5, -5, 3], [-5, 20, 3], [30, 20, 3]]))
        np.testing.assert_array_equal(small_volume.data, before)

    def test_result_is_independent_copy(self, small_volume):
        proj = PlaneProjector(small_volume)
        g = make_geom([[0, 0, 3], [0, 8, 3], [8, 8, 3]])
        E1 = proj.project(g)
        E1[:] = 0
        E2 = proj.project(g)
        assert E2.any() and not np.shares_memory(E2, small_volume.data)

    def test_same_plane_twice_is_deterministic(self, small_volume):
        proj = PlaneProjector(small_volume)
        g = make_geom([[1, 1, 1], [3, 9, 6], [14, 9, 8]])
        np.testing.assert_array_equal(proj.project(g), proj.project(g))

    def test_grid_cache_reuses_and_is_bounded(self, small_volume):
        proj = PlaneProjector(small_volume)
        assert proj._grid(4, 5)[0] is proj._grid(4, 5)[0]
        for n in range(1, 120):
            proj._grid(n, n)
        assert len(proj._grid_cache) <= 66


@unit
class TestProjectorAlgorithm:
    """Extensive checks of the sampling maths, mostly against a naive reference."""

    def test_partial_overlap_zero_fills_only_outside(self):
        M = np.full((10, 10, 10), 9, np.uint8)
        E = PlaneProjector(Volume(M)).project(make_geom([[-5, 0, 5], [-5, 8, 5], [4, 8, 5]]))
        assert E.shape == (8, 9) and not E[:, :5].any() and np.all(E[:, 5:] == 9)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_exact_boundaries(self, axis):
        """Voxel n-1 is valid; n and -1 are not."""
        n = 6
        proj = PlaneProjector(Volume(np.full((n, n, n), 7, np.uint8)))

        def pt(a, b, c):
            return {"x": [a, b, c], "y": [b, a, c], "z": [b, c, a]}[axis]

        for start, valid in [(n - 1, True), (n, False), (-1, False), (0, True)]:
            E = proj.project(make_geom([pt(start, 0, 0), pt(start, 3, 0), pt(start, 3, 3)]))
            assert (E == 7).all() == valid and (E == 0).all() == (not valid)

    def test_negative_coordinates_do_not_wrap(self):
        M = np.ones((8, 8, 8), np.uint8) * 5
        assert not PlaneProjector(Volume(M)).project(make_geom([[-1, 0, 0], [-1, 4, 0], [-1, 4, 4]])).any()

    def test_huge_coordinates_out_of_bounds(self):
        E = PlaneProjector(Volume(np.ones((8, 8, 8), np.uint8))).project(
            make_geom([[1e6, 1e6, 1e6], [1e6, 1e6 + 5, 1e6], [1e6 + 5, 1e6 + 5, 1e6]]))
        assert not E.any()

    def test_rounding_to_nearest_voxel(self):
        M = np.arange(5 * 5 * 5, dtype=np.uint8).reshape(5, 5, 5)
        proj = PlaneProjector(Volume(M))
        assert proj.project(make_geom([[1.4, 2, 2], [1.4, 4, 2], [3.4, 4, 2]]))[0, 0] == M[2, 2, 1]
        assert proj.project(make_geom([[1.6, 2, 2], [1.6, 4, 2], [3.6, 4, 2]]))[0, 0] == M[2, 2, 2]

    def test_rotating_plane_90_degrees_transposes_axes(self):
        proj = PlaneProjector(Volume(ramp_volume(12, 12, 12, np.uint8)))
        E1 = proj.project(make_geom([[0, 0, 4], [0, 8, 4], [8, 8, 4]]))
        E2 = proj.project(make_geom([[0, 0, 4], [8, 0, 4], [8, 8, 4]]))
        np.testing.assert_array_equal(E2, E1.T)

    def test_reversed_direction_flips_image(self):
        proj = PlaneProjector(Volume(ramp_volume(12, 12, 12, np.uint8)))
        fwd = proj.project(make_geom([[0, 0, 4], [0, 8, 4], [8, 8, 4]]))
        back = proj.project(make_geom([[7, 7, 4], [7, -1, 4], [-1, -1, 4]]))
        np.testing.assert_array_equal(back, fwd[::-1, ::-1])

    def test_origin_translation_shifts_content(self):
        M = ramp_volume(16, 16, 16, np.uint8)
        proj = PlaneProjector(Volume(M))
        b = proj.project(make_geom([[3, 2, 5], [3, 10, 5], [11, 10, 5]]))
        np.testing.assert_array_equal(b, M[5, 2:10, 3:11])

    def test_trapezoid_matches_equivalent_triangle(self):
        proj = PlaneProjector(Volume(ramp_volume(24, 24, 24, np.uint8)))
        tri = proj.project(make_geom([[4, 0, 6], [0, 10, 6], [16, 10, 6]]))
        trap = proj.project(make_geom([[4, 0, 6], [0, 10, 6], [12, 0, 6], [16, 10, 6]]))
        np.testing.assert_array_equal(tri, trap)

    def test_single_pixel_plane(self):
        M = ramp_volume(6, 6, 6, np.uint8)
        E = PlaneProjector(Volume(M)).project(make_geom([[2, 2, 2], [2, 2, 2], [2, 2, 2]]))
        assert E.shape == (1, 1) and E[0, 0] == M[2, 2, 2]

    def test_non_cubic_volume_axes_not_mixed_up(self):
        nx, ny, nz = 11, 7, 5
        proj = PlaneProjector(Volume(ramp_volume(nx, ny, nz, np.uint32)))   # value = z*10000 + y*100 + x
        E = proj.project(make_geom([[0, 0, 3], [0, 6, 3], [10, 6, 3]]))
        ii, jj = np.mgrid[0:6, 0:10]
        np.testing.assert_array_equal(E, 3 * 10000 + ii * 100 + jj)
        E = proj.project(make_geom([[2, 0, 0], [2, 0, 4], [2, 6, 4]]))
        ii, jj = np.mgrid[0:4, 0:6]
        np.testing.assert_array_equal(E, ii * 10000 + jj * 100 + 2)

    @pytest.mark.parametrize("shape", [(1, 1, 1), (1, 9, 9), (9, 1, 9), (9, 9, 1), (2, 3, 4)])
    def test_degenerate_volume_shapes(self, shape):
        E = PlaneProjector(Volume(np.full(shape, 3, np.uint8))).project(make_geom([[0, 0, 0], [0, 5, 0], [5, 5, 0]]))
        assert E.shape == (5, 5) and set(np.unique(E)) <= {0, 3}

    @pytest.mark.parametrize("seed", range(25))
    def test_random_oblique_planes_match_reference(self, seed):
        rng = np.random.default_rng(seed)
        shape = tuple(rng.integers(8, 28, 3))
        M = rng.integers(1, 65535, shape, dtype=np.uint16)
        nz, ny, nx = shape
        pts = rng.uniform([-6, -6, -6], [nx + 6, ny + 6, nz + 6], (3, 3))
        geom = make_geom(pts)
        if geom.rows * geom.cols > 4000:
            pts[1] = pts[0] + (pts[1] - pts[0]) * 0.3
            pts[2] = pts[1] + (pts[2] - pts[1]) * 0.3
            geom = make_geom(pts)
        np.testing.assert_array_equal(PlaneProjector(Volume(M)).project(geom), reference_project(M, geom))

    @pytest.mark.parametrize("seed", range(8))
    def test_random_trapezoids_match_reference(self, seed):
        rng = np.random.default_rng(100 + seed)
        M = rng.integers(1, 255, (15, 15, 15), dtype=np.uint8)
        a = rng.uniform(0, 14, 3)
        b = a + rng.uniform(-6, 6, 3)
        c = b + rng.uniform(-6, 6, 3)
        geom = make_geom([a, b, a + (c - b), c])
        np.testing.assert_array_equal(PlaneProjector(Volume(M)).project(geom), reference_project(M, geom))

    @pytest.mark.parametrize("angle", range(0, 181, 15))
    def test_rotating_plane_through_centre_matches_reference(self, angle):
        M = ramp_volume(21, 21, 21, np.uint8)
        c = np.array([10.0, 10.0, 10.0])
        th = math.radians(angle)
        w = np.array([math.cos(th), math.sin(th), 0.0])
        a = c - 9 * np.array([0, 0, 1.0]) - 9 * w
        b = a + 18 * np.array([0, 0, 1.0])
        geom = make_geom([a, b, b + 18 * w])
        np.testing.assert_array_equal(PlaneProjector(Volume(M)).project(geom), reference_project(M, geom))

    def test_constant_volume_inside_is_constant(self):
        E = PlaneProjector(Volume(np.full((20, 20, 20), 42, np.uint8))).project(
            make_geom([[2, 2, 2], [2, 15, 9], [15, 15, 16]]))
        assert set(np.unique(E)) <= {0, 42} and (E == 42).sum() > 0

    def test_known_sphere_cross_section_is_disc(self):
        n = 41
        z, y, x = np.mgrid[0:n, 0:n, 0:n]
        M = (((x - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2) <= 15 ** 2).astype(np.uint8) * 200
        E = PlaneProjector(Volume(M)).project(make_geom([[0, 0, 20], [0, 40, 20], [40, 40, 20]]))
        assert (E > 0).sum() == pytest.approx(math.pi * 15 ** 2, rel=0.05)
        rr, cc = np.nonzero(E)
        assert abs(rr.mean() - 20) < 1 and abs(cc.mean() - 20) < 1

    def test_any_tilt_of_centred_sphere_gives_a_disc_of_the_same_area(self):
        n = 41
        z, y, x = np.mgrid[0:n, 0:n, 0:n]
        M = (((x - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2) <= 12 ** 2).astype(np.uint8)
        proj = PlaneProjector(Volume(M))
        ideal = math.pi * 12 ** 2
        for deg in range(0, 90, 15):
            th = math.radians(deg)
            u = np.array([0, math.cos(th), math.sin(th)])
            w = np.array([1.0, 0, 0])
            a = np.array([20.0, 20, 20]) - 18 * u - 18 * w
            area = int(proj.project(make_geom([a, a + 36 * u, a + 36 * u + 36 * w])).sum())
            assert area == pytest.approx(ideal, rel=0.08)

    def test_large_flat_index_no_int32_overflow(self):
        """Index maths stays correct at the end of a ~134M-voxel volume."""
        n = 512
        require_ram(n ** 3 * 3)
        M = np.zeros((n, n, n), np.uint8)
        M[n - 1, n - 1, n - 1] = 77
        M[n - 1, n - 1, n - 2] = 66
        E = PlaneProjector(Volume(M)).project(make_geom([[n - 2, n - 1, n - 1], [n - 2, n, n - 1], [n, n, n - 1]]))
        assert E.shape == (1, 2) and E[0, 0] == 66 and E[0, 1] == 77


@unit
class TestVideoRenderer:
    def _make(self, tmp_path, size=(20, 30), fps=25, vmin=0, vmax=255, dtype=np.uint8):
        return VideoRenderer(str(tmp_path / "o.mp4"), size, fps, vmin, vmax, dtype)

    def test_dimensions_made_even(self, tmp_path):
        v = self._make(tmp_path, (21, 31))
        assert (v.h, v.w) == (22, 32)
        v.close()

    def test_even_dimensions_unchanged(self, tmp_path):
        v = self._make(tmp_path, (20, 30))
        assert (v.h, v.w) == (20, 30)
        v.close()

    def test_uint8_passthrough(self, tmp_path):
        v = self._make(tmp_path)
        E = np.arange(12, dtype=np.uint8).reshape(3, 4)
        assert v._to_u8(E) is E
        v.close()

    def test_uint16_scaling_endpoints_and_midpoint(self, tmp_path):
        v = self._make(tmp_path, vmin=1000, vmax=3000, dtype=np.uint16)
        out = v._to_u8(np.array([[1000, 2000, 3000]], np.uint16))
        assert out.dtype == np.uint8 and out[0, 0] == 0 and out[0, 2] == 255 and abs(int(out[0, 1]) - 127) <= 1
        v.close()

    def test_scaling_clips_outside_range(self, tmp_path):
        v = self._make(tmp_path, vmin=100, vmax=200, dtype=np.uint16)
        assert list(v._to_u8(np.array([[0, 100, 200, 60000]], np.uint16))[0]) == [0, 0, 255, 255]
        v.close()

    def test_scaling_uint32_large_values(self, tmp_path):
        v = self._make(tmp_path, vmin=0, vmax=4_000_000_000, dtype=np.uint32)
        out = v._to_u8(np.array([[0, 2_000_000_000, 4_000_000_000]], np.uint32))
        assert out[0, 0] == 0 and out[0, 2] == 255 and abs(int(out[0, 1]) - 127) <= 1
        v.close()

    def test_scaling_signed_negative_values(self, tmp_path):
        v = self._make(tmp_path, vmin=-100, vmax=100, dtype=np.int16)
        out = v._to_u8(np.array([[-100, 0, 100]], np.int16))
        assert out[0, 0] == 0 and out[0, 2] == 255 and abs(int(out[0, 1]) - 127) <= 1
        v.close()

    def test_constant_volume_no_division_by_zero(self, tmp_path):
        v = self._make(tmp_path, vmin=500, vmax=500, dtype=np.uint16)
        out = v._to_u8(np.full((2, 2), 500, np.uint16))
        assert out.dtype == np.uint8 and np.all(out == 0)
        v.close()

    def test_frame_counter_and_file_written(self, tmp_path):
        with self._make(tmp_path, (20, 30)) as v:
            for k in range(7):
                v.add_frame(np.full((20, 30), k * 30, np.uint8))
            assert v.frames == 7
        frames, (w, h, fps) = read_frames(tmp_path / "o.mp4")
        assert len(frames) == 7 and (w, h) == (30, 20) and fps == pytest.approx(25, abs=0.5)

    def test_smaller_frames_are_centered_on_black_canvas(self, tmp_path):
        v = self._make(tmp_path, (20, 30))
        v.add_frame(np.full((10, 10), 200, np.uint8))
        c = v._canvas
        assert c.shape == (20, 30) and np.all(c[5:15, 10:20] == 200)
        assert c[:5].sum() == 0 and c[15:].sum() == 0 and c[:, :10].sum() == 0 and c[:, 20:].sum() == 0
        v.close()

    def test_canvas_cleared_between_frames(self, tmp_path):
        v = self._make(tmp_path, (20, 30))
        v.add_frame(np.full((20, 30), 255, np.uint8))
        v.add_frame(np.full((4, 4), 9, np.uint8))
        assert (v._canvas > 0).sum() == 16
        v.close()

    def test_video_brightness_roughly_preserved(self, tmp_path):
        with self._make(tmp_path, (64, 64)) as v:
            v.add_frame(np.full((64, 64), 40, np.uint8))
            v.add_frame(np.full((64, 64), 220, np.uint8))
        frames, _ = read_frames(tmp_path / "o.mp4")
        assert frames[0].mean() == pytest.approx(40, abs=12) and frames[1].mean() == pytest.approx(220, abs=12)

    def test_close_is_idempotent(self, tmp_path):
        v = self._make(tmp_path)
        v.close()
        v.close()

    def test_context_manager_closes_on_exception(self, tmp_path):
        with pytest.raises(RuntimeError, match="boom"):
            with self._make(tmp_path) as v:
                raise RuntimeError("boom")
        assert v._writer is None

    def test_unwritable_path_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="video writer"):
            VideoRenderer(str(tmp_path / "missing_dir" / "o.mp4"), (20, 30), 25, 0, 255, np.uint8)

    def test_requires_opencv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Ex0, "cv2", None)
        with pytest.raises(RuntimeError, match="OpenCV"):
            VideoRenderer(str(tmp_path / "o.mp4"), (20, 30), 25, 0, 255, np.uint8)


# =========================================================================== #
# SECTION 4 - END-TO-END TESTING (STD 3.3)
# =========================================================================== #

@e2e
class TestEndToEndSTD:
    def test_E2E_01_full_pipeline_triangle(self, fixture_vol_8bit, fixture_traj_triangle, tmp_path):
        """E2E-01: 8-bit NRRD + 100 triangles -> playable MP4 with exactly 100 frames."""
        out = tmp_path / "out.mp4"
        r = run_cli(fixture_vol_8bit, fixture_traj_triangle, out)
        assert r.returncode == 0, r.stderr
        info = probe_video(out)
        assert info["decoded"] == 100                       # every frame decodes -> playable
        if info["frames"] is not None:                      # ffprobe available
            assert info["frames"] == 100 and info["codec"] == "mpeg4"

    def test_E2E_02_full_pipeline_trapezoid(self, fixture_vol_32bit, fixture_traj_trapezoid, tmp_path):
        """E2E-02: 32-bit NRRD + 50 trapezoids -> MP4 with exactly 50 frames."""
        out = tmp_path / "out.mp4"
        r = run_cli(fixture_vol_32bit, fixture_traj_trapezoid, out)
        assert r.returncode == 0, r.stderr
        info = probe_video(out)
        assert info["decoded"] == 50
        if info["frames"] is not None:
            assert info["frames"] == 50

    def test_E2E_16bit_also_works(self, fixture_vol_16bit, fixture_traj_triangle, tmp_path):
        out = tmp_path / "out.mp4"
        assert run_cli(fixture_vol_16bit, fixture_traj_triangle, out).returncode == 0
        assert probe_video(out)["decoded"] == 100

    def test_E2E_frames_contain_the_phantom(self, fixture_vol_8bit, fixture_traj_triangle, tmp_path):
        out = tmp_path / "out.mp4"
        assert run_cli(fixture_vol_8bit, fixture_traj_triangle, out).returncode == 0
        means = [f.mean() for f in read_frames(out)[0]]
        assert max(means) > 20 and min(means) >= 0          # planes through the sphere are not blank


@e2e
class TestPipeline:
    def test_frame_count_matches_valid_csv_lines(self, sample_files):
        nrrd, csv, mp4 = sample_files
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 10
        assert len(read_frames(mp4)[0]) == 10

    def test_creates_missing_output_directories(self, sample_files, tmp_path):
        nrrd, csv, _ = sample_files
        deep = tmp_path / "a" / "b" / "c" / "r.mp4"
        run_pipeline(str(nrrd), str(csv), str(deep))
        assert deep.is_file() and deep.stat().st_size > 0

    def test_frame_size_is_largest_plane_padded_even(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mixed.csv"
        csv.write_text("0,0,5,0,10,5,20,10,5\n0,0,6,0,29,6,40,29,6\n")
        run_pipeline(str(nrrd), str(csv), str(mp4))
        _, (w, h, _) = read_frames(mp4)
        assert (w, h) == (40, 30)

    def test_frames_show_sphere_growing_then_shrinking(self, sample_files):
        nrrd, csv, mp4 = sample_files
        run_pipeline(str(nrrd), str(csv), str(mp4))
        bright = [f.mean() for f in read_frames(mp4)[0]]
        assert bright[0] < 5 and bright[-1] < 5 and max(bright) > 20
        assert bright.index(max(bright)) in (4, 5)

    def test_order_follows_csv(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "rev.csv"
        csv.write_text("0,0,15,0,39,15,49,39,15\n0,0,0,0,39,0,49,39,0\n")
        run_pipeline(str(nrrd), str(csv), str(mp4))
        frames, _ = read_frames(mp4)
        assert frames[0].mean() > frames[1].mean() + 10

    def test_fps_is_applied(self, sample_files):
        nrrd, csv, mp4 = sample_files
        run_pipeline(str(nrrd), str(csv), str(mp4), fps=50)
        assert read_frames(mp4)[1][2] == pytest.approx(50, abs=1)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_all_supported_voxel_sizes_end_to_end(self, tmp_path, dtype):
        rng = np.random.default_rng(0)
        vol = rng.integers(0, np.iinfo(dtype).max, (12, 14, 16), dtype=dtype, endpoint=True)
        nrrd = write_nrrd(tmp_path / "v.nrrd", vol)
        csv = tmp_path / "p.csv"
        csv.write_text("0,0,3,0,13,3,15,13,3\n0,0,6,0,13,6,15,13,6\n")
        out = tmp_path / "o.mp4"
        assert run_pipeline(str(nrrd), str(csv), str(out)) == 2
        assert len(read_frames(out)[0]) == 2

    def test_bad_csv_lines_do_not_stop_pipeline(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("header,line\n0,0,5,0,39,5,49,39,5\n1,2,3\n0,0,9,0,39,9,49,39,9\n")
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 2

    def test_bad_lines_warned_once_not_twice(self, sample_files, tmp_path, caplog):
        """The CSV is streamed twice; warnings must only come from the first pass."""
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("header,line\n0,0,5,0,39,5,49,39,5\n")
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            run_pipeline(str(nrrd), str(csv), str(mp4))
        assert caplog.text.count("CSV line 1 skipped") == 1

    def test_planes_outside_volume_still_produce_black_frames(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "out.csv"
        csv.write_text("500,500,500,500,520,500,520,520,500\n")
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 1
        assert read_frames(mp4)[0][0].mean() < 3

    def test_csv_with_no_valid_planes_raises(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "none.csv"
        csv.write_text("a,b,c\n")
        with pytest.raises(ValueError, match="no valid planes"):
            run_pipeline(str(nrrd), str(csv), str(mp4))


@e2e
class TestNonCoplanarInput:
    """Trapezoids whose 4 points are not in one plane are dropped, the rest still renders."""
    GOOD_TRI = "0,0,15,0,39,15,49,39,15"
    GOOD_TRAP = "10,0,5,5,20,5,30,0,5,35,20,5"
    BAD_TRAP = "10,0,5,5,20,5,30,0,15,35,20,5"            # a'' is 10 voxels above the plane

    def test_pipeline_skips_non_coplanar_trapezoid(self, sample_files, tmp_path, caplog):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("\n".join([self.GOOD_TRI, self.BAD_TRAP, self.GOOD_TRAP]) + "\n")
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 2
        assert len(read_frames(mp4)[0]) == 2
        assert caplog.text.count("not coplanar") == 1 and "line 2 skipped" in caplog.text

    def test_cli_exit_zero_with_warning_when_some_lines_are_good(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("\n".join([self.GOOD_TRAP, self.BAD_TRAP]) + "\n")
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 0 and "not coplanar" in r.stderr and "Traceback" not in r.stderr
        assert probe_video(mp4)["decoded"] == 1

    def test_cli_exit_one_when_every_line_is_non_coplanar(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "bad.csv"
        csv.write_text(self.BAD_TRAP + "\n")
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 1 and "no valid planes" in r.stderr and not mp4.exists()

    def test_skipped_line_does_not_shift_frame_order(self, sample_files, tmp_path):
        """Frames stay in CSV order: bright (z=15 through the sphere), skipped, dark (z=0)."""
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "order.csv"
        csv.write_text("\n".join([self.GOOD_TRI, self.BAD_TRAP, "0,0,0,0,39,0,49,39,0"]) + "\n")
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 2
        frames = read_frames(mp4)[0]
        assert frames[0].mean() > frames[1].mean() + 10


@e2e
class TestOversizedAndDegenerateInput:
    """One absurd or broken line must not take the whole run down."""
    GOOD_A = "0,0,15,0,39,15,49,39,15"
    GOOD_B = "0,0,5,0,39,5,49,39,5"
    HUGE = "0,0,0,0,1e12,0,20,1e12,0"                     # typo-sized plane
    IDENTICAL = "5,5,5,5,5,5,5,5,5"

    def test_max_plane_side_is_twice_the_volume_diagonal(self):
        assert Ex0.max_plane_side((30, 40, 50)) == 2 * math.ceil(math.sqrt(30 ** 2 + 40 ** 2 + 50 ** 2))
        assert Ex0.max_plane_side((1, 1, 1)) == 4
        assert Ex0.max_plane_side((500, 500, 500)) == 2 * 867

    def test_pipeline_skips_oversized_plane_and_renders_the_rest(self, sample_files, tmp_path, caplog):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("\n".join([self.GOOD_A, self.HUGE, self.GOOD_B]) + "\n")
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 2
        assert caplog.text.count("larger than") == 1 and "line 2 skipped" in caplog.text
        frames, (w, h, _) = read_frames(mp4)
        assert len(frames) == 2 and (w, h) == (50, 40)         # canvas sized by the GOOD planes, not the huge one

    @pytest.mark.parametrize("huge", ["0,0,0,0,1e6,0,20,1e6,0", "0,0,0,0,-1e9,0,20,-1e9,0", "0,0,0,1e12,0,0,1e12,20,0"])
    def test_various_oversized_lines(self, sample_files, tmp_path, huge):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text(self.GOOD_A + "\n" + huge + "\n")
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 1

    def test_size_limit_boundary(self, sample_files, tmp_path):
        """Volume 50x40x30: limit = 2 * ceil(70.7) = 142 voxels per side."""
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "edge.csv"
        csv.write_text("0,0,5,0,142,5,10,142,5\n0,0,5,0,143,5,10,143,5\n")
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 1        # 142 kept, 143 skipped

    def test_cli_exit_zero_and_warning_when_one_line_is_huge(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("\n".join([self.GOOD_A, self.HUGE, self.GOOD_B]) + "\n")
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 0 and "larger than" in r.stderr
        assert "Insufficient Memory" not in r.stderr and "Traceback" not in r.stderr
        assert probe_video(mp4)["decoded"] == 2

    def test_cli_exit_one_when_every_line_is_oversized(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "bad.csv"
        csv.write_text(self.HUGE + "\n")
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 1 and "no valid planes" in r.stderr and "Insufficient Memory" not in r.stderr
        assert not mp4.exists()

    def test_pipeline_skips_degenerate_planes_and_keeps_order(self, sample_files, tmp_path, caplog):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mix.csv"
        csv.write_text("\n".join([self.GOOD_A, self.IDENTICAL, "0,0,0,5,5,5,10,10,10", "0,0,0,0,39,0,49,39,0"]) + "\n")
        with caplog.at_level(logging.WARNING, logger="Ex0"):
            assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 2
        assert caplog.text.count("degenerate plane") == 2
        frames = read_frames(mp4)[0]
        assert frames[0].mean() > frames[1].mean() + 10        # bright z=15 plane first, dark z=0 plane second

    def test_cli_exit_one_when_every_line_is_degenerate(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "bad.csv"
        csv.write_text(self.IDENTICAL + "\n0,0,0,5,5,5,10,10,10\n")
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 1 and "no valid planes" in r.stderr

    def test_everything_bad_at_once_still_renders_the_good_lines(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "chaos.csv"
        csv.write_text("\n".join([
            "ax,ay,az,bx,by,bz,cx,cy,cz", self.GOOD_A, "1,2,3", self.HUGE, self.IDENTICAL,
            "10,0,5,5,20,5,30,0,15,35,20,5",                   # non-coplanar trapezoid
            "(0,0,0),(0,10,0),(20,10,0)", "0,0,0,0,10,,20,10,0", self.GOOD_B]) + "\n")
        r = run_cli(nrrd, csv, mp4)
        assert r.returncode == 0 and "Traceback" not in r.stderr
        assert probe_video(mp4)["decoded"] == 2


@e2e
class TestMain:
    def test_success_returns_zero(self, sample_files):
        nrrd, csv, mp4 = sample_files
        assert main([str(nrrd), str(csv), str(mp4)]) == 0 and mp4.is_file()

    def test_fps_option(self, sample_files):
        nrrd, csv, mp4 = sample_files
        assert main([str(nrrd), str(csv), str(mp4), "--fps", "100"]) == 0
        assert read_frames(mp4)[1][2] == pytest.approx(100, abs=2)

    def test_missing_nrrd_returns_one(self, sample_files):
        _, csv, mp4 = sample_files
        assert main(["nope.nrrd", str(csv), str(mp4)]) == 1 and not mp4.exists()

    def test_missing_csv_returns_one(self, sample_files):
        nrrd, _, mp4 = sample_files
        assert main([str(nrrd), "nope.csv", str(mp4)]) == 1

    def test_corrupt_nrrd_returns_one(self, sample_files, tmp_path):
        _, csv, mp4 = sample_files
        bad = tmp_path / "bad.nrrd"
        bad.write_bytes(b"not an nrrd file at all")
        assert main([str(bad), str(csv), str(mp4)]) == 1

    def test_empty_csv_returns_one(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        empty = tmp_path / "e.csv"
        empty.write_text("")
        assert main([str(nrrd), str(empty), str(mp4)]) == 1

    @pytest.mark.parametrize("argv", [[], ["a"], ["a", "b"]])
    def test_missing_arguments_exit_with_usage_error(self, argv):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2

    def test_too_many_arguments_rejected(self):
        with pytest.raises(SystemExit):
            main(["a", "b", "c", "d"])

    def test_invalid_fps_value_rejected(self, sample_files):
        nrrd, csv, mp4 = sample_files
        with pytest.raises(SystemExit):
            main([str(nrrd), str(csv), str(mp4), "--fps", "fast"])

    def test_memory_error_is_reported_cleanly(self, sample_files, monkeypatch, caplog):
        nrrd, csv, mp4 = sample_files
        monkeypatch.setattr(Ex0.NrrdReader, "read", staticmethod(lambda p: (_ for _ in ()).throw(MemoryError())))
        with caplog.at_level(logging.ERROR, logger="Ex0"):
            assert main([str(nrrd), str(csv), str(mp4)]) == 1
        assert "Insufficient Memory" in caplog.text


# =========================================================================== #
# SECTION 5 - REGRESSION TESTING (STD 3.4)
# =========================================================================== #

# SHA-256 over the projected matrices E of the reference CSV on the 500^3 phantom.
# It is a "gold standard" captured from a verified run of Ex0.py.  If a deliberate
# algorithm change alters the output, re-run `python test_Ex0.py --print-gold`
# and paste the new value (after reviewing why it changed!).
GOLD_REG01_SHA256 = "b436298b6cc740397cc1641b119f15b3d37722b2f89352e795c61c7258134cda"


def reference_frames_sha256(volume: Volume, csv_path) -> tuple[str, int]:
    h = hashlib.sha256()
    proj = PlaneProjector(volume)
    n = 0
    for plane in PlaneCsvParser.iter_planes(str(csv_path)):
        E = proj.project(PlaneGeometry.from_points(plane.points))
        h.update(str(E.shape).encode() + str(E.dtype).encode())
        h.update(np.ascontiguousarray(E).tobytes())
        n += 1
    return h.hexdigest(), n


@regression
class TestRegression:
    def test_REG_01_output_checksum_verification(self, phantom_500_u8, tmp_path):
        """REG-01: frames for the reference CSV on the 500^3 volume match the gold hash."""
        csv = gf.write_reference_trajectory(tmp_path / "reference.csv", 500)
        digest, n = reference_frames_sha256(phantom_500_u8, csv)
        assert n == 30
        assert digest == GOLD_REG01_SHA256, (
            "Rendered frames changed. If this is intentional, review the change and update "
            f"GOLD_REG01_SHA256 to {digest}")

    def test_REG_01_video_matches_expected_frames(self, phantom_500_u8, tmp_path):
        """The encoded MP4 (lossy codec, so compared by brightness) follows the same frames."""
        nrrd = write_nrrd(tmp_path / "v.nrrd", phantom_500_u8.data, encoding="gzip")
        csv = gf.write_reference_trajectory(tmp_path / "reference.csv", 500)
        out = tmp_path / "o.mp4"
        assert run_pipeline(str(nrrd), str(csv), str(out)) == 30
        frames, (w, h, _) = read_frames(out)
        assert len(frames) == 30
        proj = PlaneProjector(phantom_500_u8)
        for plane, frame in zip(PlaneCsvParser.iter_planes(str(csv)), frames):
            E = proj.project(PlaneGeometry.from_points(plane.points))
            expected = E.sum(dtype=np.float64) / (w * h)          # E is centred on a black w x h canvas
            assert frame[..., 0].mean() == pytest.approx(expected, abs=8)

    def test_REG_01_hash_detects_a_single_changed_voxel(self, tmp_path):
        """Guards the guard: the hashing scheme is sensitive to one-voxel differences."""
        M = gf.build_phantom(32, 32, 32, np.uint8)
        p = tmp_path / "reg01_sensitivity.csv"
        p.write_text("0,0,10,0,20,10,20,20,10\n")
        before, _ = reference_frames_sha256(Volume(M), p)
        M2 = M.copy()
        M2[10, 5, 5] ^= 1                                    # voxel (x=5, y=5, z=10) lies on the plane
        after, _ = reference_frames_sha256(Volume(M2), p)
        assert before != after

    def test_REG_02_zero_padding_corner_origin(self):
        """REG-02: planes starting at the absolute corner (0,0,0) have no off-by-one shift."""
        M = ramp_volume(12, 10, 8, np.uint32)                       # value = z*10000 + y*100 + x
        proj = PlaneProjector(Volume(M))
        E = proj.project(make_geom([[0, 0, 0], [0, 9, 0], [11, 9, 0]]))
        assert E.shape == (9, 11)
        assert E[0, 0] == M[0, 0, 0] == 0                           # the corner voxel itself
        assert E[0, 1] == 1 and E[1, 0] == 100                      # first step along each axis
        np.testing.assert_array_equal(E, M[0, :9, :11])
        for pts, expect in [
                ([[0, 0, 0], [0, 0, 7], [11, 0, 7]], M[:7, 0, :11]),    # corner, coronal
                ([[0, 0, 0], [0, 0, 7], [0, 9, 7]], M[:7, :9, 0])]:     # corner, sagittal
            np.testing.assert_array_equal(proj.project(make_geom(pts)), expect)

    def test_REG_02_corner_with_negative_start_is_padded_not_shifted(self):
        M = ramp_volume(12, 10, 8, np.uint32)
        E = PlaneProjector(Volume(M)).project(make_geom([[-1, -1, 0], [-1, 5, 0], [6, 5, 0]]))
        assert E.shape == (6, 7)
        assert not E[0, :].any() and not E[:, 0].any()              # row / column at -1 are padding
        np.testing.assert_array_equal(E[1:, 1:], M[0, 0:5, 0:6])    # content not shifted

    def test_REG_02_far_corner(self):
        n = 10
        M = ramp_volume(n, n, n, np.uint32)
        E = PlaneProjector(Volume(M)).project(make_geom([[n - 1, n - 1, n - 1], [n - 1, n, n - 1], [n, n, n - 1]]))
        assert E.shape == (1, 1) and E[0, 0] == M[n - 1, n - 1, n - 1]


# =========================================================================== #
# SECTION 6 - PERFORMANCE, LOAD AND STRESS TESTING (STD 3.5)
# =========================================================================== #

def measure_fps(volume: Volume, geoms, out_path) -> float:
    """Frames per second of project + render (volume loading excluded)."""
    proj = PlaneProjector(volume)
    h, w = max(g.rows for g in geoms), max(g.cols for g in geoms)
    vmax = int(np.iinfo(volume.data.dtype).max)
    with VideoRenderer(str(out_path), (h, w), 30, 0, vmax, volume.data.dtype) as video:
        proj.project(geoms[0])                                   # warm-up
        t = time.perf_counter()
        for g in geoms:
            video.add_frame(proj.project(g))
        dt = time.perf_counter() - t
    return len(geoms) / dt


def geoms_from_csv(path):
    return [PlaneGeometry.from_points(p.points) for p in PlaneCsvParser.iter_planes(str(path))]


@performance
class TestBenchmark:
    """STD 3.5.1"""

    def test_PF_01_baseline_throughput(self, phantom_500_u8, tmp_path):
        """PF-01: 500^3 volume, 300-row CSV -> >= 30 FPS."""
        csv = gf.write_trajectory(tmp_path / "t300.csv", 300, "triangle", 500, seed=3)
        geoms = geoms_from_csv(csv)
        assert len(geoms) == 300
        fps = measure_fps(phantom_500_u8, geoms, tmp_path / "o.mp4")
        print(f"\nPF-01: {fps:.0f} FPS")
        assert fps >= PERF_MIN_FPS, f"{fps:.1f} FPS < {PERF_MIN_FPS}"

    def test_PF_01_pytest_benchmark_plugin(self, request, phantom_500_u8, tmp_path):
        """Same measurement through pytest-benchmark (skipped when the plugin is not installed)."""
        pytest.importorskip("pytest_benchmark")
        benchmark = request.getfixturevalue("benchmark")
        csv = gf.write_trajectory(tmp_path / "t.csv", 20, "triangle", 500, seed=3)
        geoms = geoms_from_csv(csv)
        proj = PlaneProjector(phantom_500_u8)
        benchmark(lambda: [proj.project(g) for g in geoms])
        assert len(geoms) / benchmark.stats.stats.mean >= PERF_MIN_FPS

    @pytest.mark.skipif(os.environ.get("PERF_TARGET_HARDWARE") != "1",
                        reason="PF-02 targets optimised hardware: set PERF_TARGET_HARDWARE=1")
    def test_PF_02_optimized_hardware_target(self, phantom_500_u8, tmp_path):
        """PF-02: >= 100 FPS on the target (multi-core / AVX) machine."""
        csv = gf.write_trajectory(tmp_path / "t300.csv", 300, "triangle", 500, seed=3)
        fps = measure_fps(phantom_500_u8, geoms_from_csv(csv), tmp_path / "o.mp4")
        assert fps >= 100, f"{fps:.1f} FPS < 100"


@performance
class TestLoad:
    """STD 3.5.2"""

    def test_LD_01_sustained_max_volume_load(self, tmp_path):
        """LD-01: max-size 32-bit volume, thousands of frames -> >= 30 FPS throughout."""
        n, frames = (1023, 5000) if FULL_SCALE else (256, 1000)
        require_ram(n ** 3 * 4 + 600_000_000)
        vol = Volume(gf.build_phantom(n, n, n, np.uint32))
        csv = gf.write_trajectory(tmp_path / "heavy.csv", frames, "triangle", n, seed=4)
        geoms = geoms_from_csv(csv)
        assert len(geoms) == frames
        t = time.perf_counter()
        fps = measure_fps(vol, geoms, tmp_path / "o.mp4")
        print(f"\nLD-01 ({n}^3 uint32, {frames} frames): {fps:.0f} FPS in {time.perf_counter() - t:.1f}s")
        assert fps >= PERF_MIN_FPS
        # sustained: second half must not be dramatically slower than the first half
        half = frames // 2
        f1 = measure_fps(vol, geoms[:half], tmp_path / "a.mp4")
        f2 = measure_fps(vol, geoms[half:], tmp_path / "b.mp4")
        assert f2 > 0.5 * f1, f"throughput collapsed: {f1:.0f} -> {f2:.0f} FPS"

    def test_LD_02_memory_leak_monitor_python_heap(self):
        """LD-02 (portable): traced Python/NumPy memory is flat across the frame loop."""
        vol = Volume(gf.build_phantom(96, 96, 96, np.uint16))
        proj = PlaneProjector(vol)
        geoms = geoms_from_csv(_tmp_csv(2000, 96))
        tracemalloc.start()
        try:
            samples = []
            for i, g in enumerate(geoms):
                proj.project(g)
                if i in (300, 1000, 1999):
                    samples.append(tracemalloc.get_traced_memory()[0])
        finally:
            tracemalloc.stop()
        growth = samples[-1] - samples[0]
        assert growth < 1_000_000, f"heap grew by {growth} bytes over 1700 frames: {samples}"

    def test_LD_02_memory_leak_monitor_rss(self, tmp_path):
        """LD-02: process RSS is flat during the frame-generation loop (needs psutil)."""
        psutil = pytest.importorskip("psutil")
        proc = psutil.Process()
        vol = Volume(gf.build_phantom(96, 96, 96, np.uint16))
        proj = PlaneProjector(vol)
        geoms = geoms_from_csv(_tmp_csv(3000, 96))
        h, w = max(g.rows for g in geoms), max(g.cols for g in geoms)
        rss = []
        with VideoRenderer(str(tmp_path / "o.mp4"), (h, w), 30, 0, 65535, np.uint16) as video:
            for i, g in enumerate(geoms):
                video.add_frame(proj.project(g))
                if i in (500, 1500, 2999):
                    rss.append(proc.memory_info().rss)
        growth_mb = (rss[-1] - rss[0]) / 1e6
        assert growth_mb < 60, f"RSS grew {growth_mb:.1f} MB between frame 500 and 2999: {rss}"


_CSV_CACHE: dict = {}


def _tmp_csv(count: int, n: int) -> Path:
    """Small helper: a cached triangle CSV in the system temp dir."""
    import tempfile
    key = (count, n)
    if key not in _CSV_CACHE:
        p = Path(tempfile.mkdtemp(prefix="std_csv_")) / f"t_{count}_{n}.csv"
        gf.write_trajectory(p, count, "triangle", n, seed=11)
        _CSV_CACHE[key] = p
    return _CSV_CACHE[key]


@performance
class TestStress:
    """STD 3.5.3"""

    def test_STR_01_oversized_volume_fails_gracefully(self, tmp_path):
        """STR-01: a 2048^3 uint32 volume either loads or fails with a clear error - never a crash."""
        stub = gf.write_oversized_stub(tmp_path / "huge.nrrd", 2048)
        csv = gf.write_trajectory(tmp_path / "t.csv", 5, "triangle", 100, seed=1)
        t = time.perf_counter()
        r = run_cli(stub, csv, tmp_path / "o.mp4")
        assert r.returncode == 1                                    # handled error, not a crash signal
        assert "Traceback" not in r.stderr and "expected" in r.stderr
        assert time.perf_counter() - t < 20
        assert not (tmp_path / "o.mp4").exists()

    def test_STR_01_out_of_memory_message(self, sample_files, monkeypatch, caplog):
        nrrd, csv, mp4 = sample_files
        monkeypatch.setattr(np, "empty", lambda *a, **k: (_ for _ in ()).throw(MemoryError()))
        with caplog.at_level(logging.ERROR, logger="Ex0"):
            assert main([str(nrrd), str(csv), str(mp4)]) == 1       # gzip path allocates with np.empty
        assert "Insufficient Memory" in caplog.text

    def test_STR_02_massive_csv_is_streamed_lazily(self, tmp_path):
        """STR-02: planes are read one at a time; memory does not grow with file size."""
        path = tmp_path / "big.csv"
        path.write_text("0,0,5,0,6,5,6,6,5\n" * 500_000)            # 500,000 planar coordinates
        assert inspect.isgenerator(PlaneCsvParser.iter_planes(str(path)))
        first = next(PlaneCsvParser.iter_planes(str(path)))         # starts yielding immediately
        assert first.line_no == 1

        tracemalloc.start()
        try:
            count = 0
            for _ in PlaneCsvParser.iter_planes(str(path)):
                count += 1
                if count == 30_000:
                    break
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert count == 30_000
        assert peak < 2_000_000, f"parser held {peak} bytes - it is not streaming"   # a list would need ~20 MB

    def test_STR_02_massive_csv_renders(self, tmp_path):
        """500,000 planes render without loading the CSV into RAM (20,000 unless FULL_SCALE=1)."""
        count = 500_000 if FULL_SCALE else 20_000
        csv = tmp_path / "big.csv"
        csv.write_text("0,0,5,0,6,5,6,6,5\n" * count)
        nrrd = write_nrrd(tmp_path / "v.nrrd", ramp_volume(16, 16, 16, np.uint8))
        out = tmp_path / "o.mp4"
        before = None
        try:
            import psutil
            before = psutil.Process().memory_info().rss
        except ImportError:
            pass
        assert run_pipeline(str(nrrd), str(csv), str(out)) == count
        if before is not None:
            import psutil
            assert (psutil.Process().memory_info().rss - before) / 1e6 < 300
        cap = cv2.VideoCapture(str(out))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == count
        cap.release()

    def test_STR_03_rapid_sequential_execution_in_process(self, tmp_path):
        """STR-03: 50 back-to-back runs to the same output file leave one valid MP4."""
        nrrd = write_nrrd(tmp_path / "v.nrrd", ramp_volume(24, 24, 24, np.uint8))
        csv = tmp_path / "p.csv"
        csv.write_text("\n".join(f"0,0,{k},0,20,{k},20,20,{k}" for k in range(5)) + "\n")
        out = tmp_path / "out" / "same.mp4"
        for _ in range(50):
            assert main([str(nrrd), str(csv), str(out)]) == 0
        assert probe_video(out)["decoded"] == 5
        assert [p.name for p in out.parent.iterdir()] == ["same.mp4"]        # no stray temp files

    def test_STR_03_rapid_sequential_execution_subprocesses(self, tmp_path):
        """STR-03: real CLI processes in a loop (10 by default, 50 with FULL_SCALE=1)."""
        nrrd = write_nrrd(tmp_path / "v.nrrd", ramp_volume(24, 24, 24, np.uint8))
        csv = tmp_path / "p.csv"
        csv.write_text("\n".join(f"0,0,{k},0,20,{k},20,20,{k}" for k in range(5)) + "\n")
        out = tmp_path / "same.mp4"
        runs = 50 if FULL_SCALE else 10
        for _ in range(runs):
            r = run_cli(nrrd, csv, out)
            assert r.returncode == 0, r.stderr
        assert probe_video(out)["decoded"] == 5
        assert [p.name for p in tmp_path.glob("*.mp4*")] == ["same.mp4"]
        try:
            import psutil
            assert not [c for c in psutil.Process().children(recursive=True) if "Ex0.py" in " ".join(c.cmdline())]
        except ImportError:
            pass


# =========================================================================== #
# SECTION 7 - REQUIREMENTS TRACEABILITY (STD section 4)
# =========================================================================== #

RTM = {
    "REQ-CLI-1":  ["ST-01", "ST-02", "STR-03"],
    "REQ-DATA-1": ["UT-Parse-01", "UT-Parse-02", "UT-Parse-03", "STR-02"],
    "REQ-ALG-1":  ["UT-Logic-01", "UT-Logic-02", "UT-Logic-03", "REG-02"],
    "REQ-VID-1":  ["E2E-01", "E2E-02", "REG-01"],
    "REQ-PERF-1": ["PF-01", "PF-02", "LD-01"],
    "REQ-QUAL-1": ["LD-02", "STR-01", "STR-02"],
    "REQ-TEST-1": ["UT-Logic-01", "UT-Logic-02", "UT-Logic-03"],
    "REQ-TEST-2": ["UT-Parse-01", "UT-Parse-02", "UT-Parse-03", "UT-Parse-04"],
    "REQ-TEST-3": ["PF-01", "PF-02"],
    "REQ-TEST-4": ["E2E-01", "E2E-02", "REG-01"],
}


@unit
class TestTraceability:
    def test_every_requirement_is_covered_by_a_named_test(self):
        """Each STD test ID in the RTM exists as part of at least one test function name."""
        source = Path(__file__).read_text(encoding="utf-8")
        names = re.findall(r"^\s*def (test_\w+)", source, flags=re.M)
        missing = {}
        for req, ids in RTM.items():
            for tid in ids:
                token = tid.replace("-", "_")
                if not any(token in n for n in names):
                    missing.setdefault(req, []).append(tid)
        assert not missing, f"RTM entries without a test: {missing}"

    def test_all_std_test_ids_are_implemented(self):
        source = Path(__file__).read_text(encoding="utf-8")
        names = " ".join(re.findall(r"^\s*def (test_\w+)", source, flags=re.M))
        every_id = ["ST_01", "ST_02", "UT_Parse_01", "UT_Parse_02", "UT_Parse_03", "UT_Parse_04",
                    "UT_Logic_01", "UT_Logic_02", "UT_Logic_03", "E2E_01", "E2E_02", "REG_01", "REG_02",
                    "PF_01", "PF_02", "LD_01", "LD_02", "STR_01", "STR_02", "STR_03"]
        assert [i for i in every_id if i not in names] == []


if __name__ == "__main__":
    if "--print-gold" in sys.argv:
        import tempfile
        vol = Volume(gf.build_phantom(500, 500, 500, np.uint8))
        with tempfile.TemporaryDirectory() as d:
            csv = gf.write_reference_trajectory(Path(d) / "reference.csv", 500)
            print(reference_frames_sha256(vol, csv)[0])
    else:
        sys.exit(pytest.main([__file__, *sys.argv[1:]]))
