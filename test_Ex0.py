"""
test_Ex0.py - pytest suite for Ex0.py

Run from the folder that contains Ex0.py:
    python -m pytest test_Ex0.py -v
    python -m pytest test_Ex0.py -v -k "not performance"   # skip the speed test

Sections
    1. Data classes            (Volume, Plane, PlaneGeometry)
    2. NrrdReader              (formats, endianness, errors, axis order)
    3. PlaneCsvParser          (valid / invalid lines, file handling)
    4. PlaneProjector          (sanity + extensive algorithm tests vs a naive reference)
    5. VideoRenderer           (scaling, padding, file output)
    6. run_pipeline / main     (end-to-end and CLI)
"""
from __future__ import annotations

import dataclasses
import gzip
import logging
import math
import time
from pathlib import Path

import numpy as np
import pytest

import Ex0
from Ex0 import (NrrdReader, Plane, PlaneCsvParser, PlaneGeometry,
                 PlaneProjector, VideoRenderer, Volume, main, run_pipeline)

cv2 = pytest.importorskip("cv2")


# =========================================================================== #
# Helpers
# =========================================================================== #

NRRD_NAMES = {np.dtype(np.uint8): "uchar", np.dtype(np.uint16): "ushort",
              np.dtype(np.uint32): "uint", np.dtype(np.int8): "int8",
              np.dtype(np.int16): "short", np.dtype(np.int32): "int"}


def write_nrrd(path: Path, arr: np.ndarray, *, encoding="raw", endian="little",
               type_name=None, extra_header="", byte_skip=0, magic="NRRD0004",
               dimension=None, sizes=None) -> Path:
    """Write `arr` (shape z,y,x) as an NRRD file."""
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
    path.write_bytes(hdr.encode() + b"\n".join([b""]) * 0 + raw)
    return path


def ramp_volume(nx, ny, nz, dtype=np.uint16) -> np.ndarray:
    """Volume whose value uniquely encodes (x, y, z): easy to verify lookups."""
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    return (z * 10_000 + y * 100 + x).astype(dtype) if np.dtype(dtype).itemsize >= 4 \
        else ((z * 7 + y * 3 + x) % 250 + 1).astype(dtype)


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


@pytest.fixture
def small_volume() -> Volume:
    return Volume(ramp_volume(20, 16, 12, np.uint8))


# =========================================================================== #
# 1. Data classes
# =========================================================================== #

class TestVolume:
    def test_shape_xyz_is_reversed_numpy_shape(self):
        v = Volume(np.zeros((5, 6, 7), np.uint8))   # (z, y, x)
        assert v.shape_xyz == (7, 6, 5)

    def test_is_frozen(self):
        v = Volume(np.zeros((2, 2, 2), np.uint8))
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.data = np.ones((2, 2, 2), np.uint8)


class TestPlane:
    def test_stores_points_and_line_number(self):
        p = Plane(np.zeros((3, 3)), 4)
        assert p.points.shape == (3, 3) and p.line_no == 4

    def test_default_line_number(self):
        assert Plane(np.zeros((3, 3))).line_no == 0

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Plane(np.zeros((3, 3))).line_no = 9


class TestPlaneGeometry:
    def test_triangle_basic(self):
        g = make_geom([[0, 0, 0], [0, 10, 0], [20, 10, 0]])
        assert (g.rows, g.cols) == (10, 20)
        np.testing.assert_allclose(g.origin, [0, 0, 0])
        np.testing.assert_allclose(g.u, [0, 1, 0])      # a -> b
        np.testing.assert_allclose(g.w, [1, 0, 0])      # b -> c

    def test_trapezoid_uses_first_second_and_last_points(self):
        # a', b, a'', c
        g = make_geom([[5, 0, 0], [0, 10, 0], [15, 0, 0], [20, 10, 0]])
        np.testing.assert_allclose(g.origin, [5, 0, 0])
        assert g.rows == round(math.hypot(5, 10))
        assert g.cols == 20

    def test_vectors_are_unit_length(self):
        g = make_geom([[1, 2, 3], [4, 6, 3], [4, 6, 15]])
        assert np.linalg.norm(g.u) == pytest.approx(1, abs=1e-6)
        assert np.linalg.norm(g.w) == pytest.approx(1, abs=1e-6)

    @pytest.mark.parametrize("dab,expected", [(0.4, 1), (0.5, 0 or 1), (3.4, 3), (3.6, 4)])
    def test_rows_rounded_and_at_least_one(self, dab, expected):
        g = make_geom([[0, 0, 0], [dab, 0, 0], [dab, 5, 0]])
        assert g.rows == max(1, round(dab)) == expected or g.rows == 1

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


# =========================================================================== #
# 2. NrrdReader
# =========================================================================== #

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
        arr = (np.arange(24, dtype=np.uint16).reshape(2, 3, 4) * 1000)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, endian=endian)))
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
        """sizes = 'x y z' -> numpy shape (z, y, x); first voxels walk along x."""
        arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)   # z=2,y=3,x=4
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr)))
        assert vol.shape_xyz == (4, 3, 2)
        np.testing.assert_array_equal(vol.data.reshape(-1)[:4], [0, 1, 2, 3])
        assert vol.data[1, 2, 3] == 23

    def test_result_is_c_contiguous(self, tmp_path):
        arr = ramp_volume(6, 5, 4, np.uint8)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr)))
        assert vol.data.flags["C_CONTIGUOUS"]

    def test_comments_and_key_value_lines_ignored(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        extra = "# a comment\nmyKey:=myValue\nspace: left-posterior-superior\n"
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, extra_header=extra)))
        np.testing.assert_array_equal(vol.data, arr)

    def test_header_keys_case_insensitive_type_names(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        vol = NrrdReader.read(str(write_nrrd(tmp_path / "v.nrrd", arr, type_name="UChar")))
        np.testing.assert_array_equal(vol.data, arr)

    def test_byte_skip_raw(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        p = write_nrrd(tmp_path / "v.nrrd", arr, extra_header="byte skip: 10\n", byte_skip=10)
        np.testing.assert_array_equal(NrrdReader.read(str(p)).data, arr)

    def test_extra_trailing_bytes_ignored_for_raw(self, tmp_path):
        arr = ramp_volume(4, 4, 4, np.uint8)
        p = write_nrrd(tmp_path / "v.nrrd", arr)
        with open(p, "ab") as f:
            f.write(b"\xff" * 50)
        np.testing.assert_array_equal(NrrdReader.read(str(p)).data, arr)

    # ---- error handling ---------------------------------------------------- #
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

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            NrrdReader.read(str(tmp_path / "nope.nrrd"))

    def test_dtype_helper_default_little_endian(self):
        dt = NrrdReader._dtype({"type": "ushort"})
        assert dt.itemsize == 2 and dt.byteorder in ("<", "=")

    def test_dtype_helper_big_endian(self):
        dt = NrrdReader._dtype({"type": "ushort", "endian": "big"})
        assert dt.byteorder == ">"


# =========================================================================== #
# 3. PlaneCsvParser
# =========================================================================== #

class TestPlaneCsvParser:
    TRI = "0,0,0,0,10,0,20,10,0"
    TRAP = "5,0,0,0,10,0,15,0,0,20,10,0"

    def test_parse_line_triangle(self):
        p = PlaneCsvParser.parse_line(self.TRI, 1)
        assert p.points.shape == (3, 3) and p.line_no == 1
        np.testing.assert_array_equal(p.points[2], [20, 10, 0])

    def test_parse_line_trapezoid(self):
        p = PlaneCsvParser.parse_line(self.TRAP)
        assert p.points.shape == (4, 3)

    def test_floats_and_negatives_and_exponent(self):
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
        line = ",".join(["1"] * n)
        with caplog.at_level(logging.WARNING):
            assert PlaneCsvParser.parse_line(line, 3) is None
        if n:
            assert "line 3" in caplog.text

    def test_header_line_skipped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert PlaneCsvParser.parse_line("ax,ay,az,bx,by,bz,cx,cy,cz", 1) is None
        assert "non-numeric" in caplog.text

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_non_finite_values_skipped(self, bad):
        assert PlaneCsvParser.parse_line(f"{bad},0,0,0,10,0,20,10,0") is None

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

    def test_large_file(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text((self.TRI + "\n") * 5000)
        assert len(PlaneCsvParser.parse(str(f))) == 5000


# =========================================================================== #
# 4. PlaneProjector - sanity
# =========================================================================== #

class TestProjectorSanity:
    def test_output_shape_and_dtype(self, small_volume):
        E = PlaneProjector(small_volume).project(make_geom([[0, 0, 3], [0, 10, 3], [15, 10, 3]]))
        assert E.shape == (10, 15) and E.dtype == np.uint8

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int16])
    def test_dtype_preserved(self, dtype):
        vol = Volume(ramp_volume(8, 8, 8, dtype))
        E = PlaneProjector(vol).project(make_geom([[0, 0, 0], [0, 5, 0], [5, 5, 0]]))
        assert E.dtype == vol.data.dtype

    def test_uint32_large_values_not_truncated(self):
        data = np.full((6, 6, 6), 4_000_000_000, dtype=np.uint32)
        E = PlaneProjector(Volume(data)).project(make_geom([[0, 0, 2], [0, 4, 2], [4, 4, 2]]))
        assert np.all(E == 4_000_000_000)

    def test_axial_slice_equals_numpy_slice(self, small_volume):
        M = small_volume.data
        nz, ny, nx = M.shape
        k = 5
        E = PlaneProjector(small_volume).project(
            make_geom([[0, 0, k], [0, ny - 1, k], [nx - 1, ny - 1, k]]))
        np.testing.assert_array_equal(E, M[k, :ny - 1, :nx - 1])

    def test_coronal_slice_equals_numpy_slice(self, small_volume):
        M = small_volume.data
        nz, ny, nx = M.shape
        j = 4
        E = PlaneProjector(small_volume).project(
            make_geom([[0, j, 0], [0, j, nz - 1], [nx - 1, j, nz - 1]]))
        np.testing.assert_array_equal(E, M[:nz - 1, j, :nx - 1])

    def test_sagittal_slice_equals_numpy_slice(self, small_volume):
        M = small_volume.data
        nz, ny, nx = M.shape
        i = 7
        E = PlaneProjector(small_volume).project(
            make_geom([[i, 0, 0], [i, 0, nz - 1], [i, ny - 1, nz - 1]]))
        np.testing.assert_array_equal(E, M[:nz - 1, :ny - 1, i])

    def test_plane_fully_outside_is_all_zero(self, small_volume):
        E = PlaneProjector(small_volume).project(
            make_geom([[100, 100, 100], [100, 110, 100], [110, 110, 100]]))
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
        assert E2.any()
        assert not np.shares_memory(E2, small_volume.data)

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


# =========================================================================== #
# 4b. PlaneProjector - extensive algorithm tests
# =========================================================================== #

class TestProjectorAlgorithm:
    # ---- bounds ----------------------------------------------------------- #
    def test_partial_overlap_zero_fills_only_outside(self):
        M = np.full((10, 10, 10), 9, np.uint8)
        # plane z=5, x from -5..: columns with x<0 are outside
        E = PlaneProjector(Volume(M)).project(make_geom([[-5, 0, 5], [-5, 8, 5], [4, 8, 5]]))
        assert E.shape == (8, 9)
        assert not E[:, :5].any()          # x = -5..-1 -> 0
        assert np.all(E[:, 5:] == 9)       # x = 0..3 -> inside

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_exact_boundaries(self, axis):
        """Voxel n-1 is valid, voxel n is not (and -1 is not)."""
        n = 6
        M = np.full((n, n, n), 7, np.uint8)
        proj = PlaneProjector(Volume(M))

        def pt(a, b, c):
            return {"x": [a, b, c], "y": [b, a, c], "z": [b, c, a]}[axis]

        # 1-pixel-thick line along axis-a: use a long plane whose first row sweeps outside
        for start, valid in [(n - 1, True), (n, False), (-1, False), (0, True)]:
            g = make_geom([pt(start, 0, 0), pt(start, 3, 0), pt(start, 3, 3)])
            E = proj.project(g)
            assert (E == 7).all() == valid and (E == 0).all() == (not valid)

    def test_negative_coordinates_do_not_wrap(self):
        """-1 must not wrap around to the last voxel (unsigned-compare trick)."""
        M = np.ones((8, 8, 8), np.uint8) * 5
        E = PlaneProjector(Volume(M)).project(make_geom([[-1, 0, 0], [-1, 4, 0], [-1, 4, 4]]))
        assert not E.any()

    def test_huge_coordinates_out_of_bounds(self):
        M = np.ones((8, 8, 8), np.uint8)
        E = PlaneProjector(Volume(M)).project(
            make_geom([[1e6, 1e6, 1e6], [1e6, 1e6 + 5, 1e6], [1e6 + 5, 1e6 + 5, 1e6]]))
        assert not E.any()

    def test_rounding_to_nearest_voxel(self):
        M = np.arange(5 * 5 * 5, dtype=np.uint8).reshape(5, 5, 5)
        proj = PlaneProjector(Volume(M))
        g = make_geom([[1.4, 2, 2], [1.4, 4, 2], [3.4, 4, 2]])
        E = proj.project(g)
        assert E[0, 0] == M[2, 2, 1]       # 1.4 -> 1
        g = make_geom([[1.6, 2, 2], [1.6, 4, 2], [3.6, 4, 2]])
        E = proj.project(g)
        assert E[0, 0] == M[2, 2, 2]       # 1.6 -> 2

    # ---- geometry --------------------------------------------------------- #
    def test_rotating_plane_90_degrees_transposes_axes(self):
        M = ramp_volume(12, 12, 12, np.uint8)
        proj = PlaneProjector(Volume(M))
        E1 = proj.project(make_geom([[0, 0, 4], [0, 8, 4], [8, 8, 4]]))   # rows=y, cols=x
        E2 = proj.project(make_geom([[0, 0, 4], [8, 0, 4], [8, 8, 4]]))   # rows=x, cols=y
        np.testing.assert_array_equal(E2, E1.T)

    def test_reversed_direction_flips_image(self):
        M = ramp_volume(12, 12, 12, np.uint8)
        proj = PlaneProjector(Volume(M))
        fwd = proj.project(make_geom([[0, 0, 4], [0, 8, 4], [8, 8, 4]]))
        back = proj.project(make_geom([[7, 7, 4], [7, -1, 4], [-1, -1, 4]]))  # both axes reversed
        np.testing.assert_array_equal(back, fwd[::-1, ::-1])

    def test_origin_translation_shifts_content(self):
        M = ramp_volume(16, 16, 16, np.uint8)
        proj = PlaneProjector(Volume(M))
        a = proj.project(make_geom([[0, 0, 5], [0, 8, 5], [8, 8, 5]]))
        b = proj.project(make_geom([[3, 2, 5], [3, 10, 5], [11, 10, 5]]))
        np.testing.assert_array_equal(b, M[5, 2:10, 3:11])
        assert a.shape == b.shape

    def test_trapezoid_matches_equivalent_triangle(self):
        """Trapezoid sampling depends only on (a', b, c), as documented."""
        M = ramp_volume(24, 24, 24, np.uint8)
        proj = PlaneProjector(Volume(M))
        tri = proj.project(make_geom([[4, 0, 6], [0, 10, 6], [16, 10, 6]]))
        trap = proj.project(make_geom([[4, 0, 6], [0, 10, 6], [12, 0, 6], [16, 10, 6]]))
        np.testing.assert_array_equal(tri, trap)

    def test_single_pixel_plane(self):
        M = ramp_volume(6, 6, 6, np.uint8)
        E = PlaneProjector(Volume(M)).project(make_geom([[2, 2, 2], [2, 2, 2], [2, 2, 2]]))
        assert E.shape == (1, 1) and E[0, 0] == M[2, 2, 2]

    def test_non_cubic_volume_axes_not_mixed_up(self):
        """Distinct nx, ny, nz expose any stride bug in the flat-index maths."""
        nx, ny, nz = 11, 7, 5
        M = ramp_volume(nx, ny, nz, np.uint32)       # value = z*10000 + y*100 + x
        proj = PlaneProjector(Volume(M))
        E = proj.project(make_geom([[0, 0, 3], [0, 6, 3], [10, 6, 3]]))
        ii, jj = np.mgrid[0:6, 0:10]
        np.testing.assert_array_equal(E, 3 * 10000 + ii * 100 + jj)
        E = proj.project(make_geom([[2, 0, 0], [2, 0, 4], [2, 6, 4]]))   # sagittal x=2
        ii, jj = np.mgrid[0:4, 0:6]
        np.testing.assert_array_equal(E, ii * 10000 + jj * 100 + 2)

    @pytest.mark.parametrize("shape", [(1, 1, 1), (1, 9, 9), (9, 1, 9), (9, 9, 1), (2, 3, 4)])
    def test_degenerate_volume_shapes(self, shape):
        nz, ny, nx = shape
        M = np.full(shape, 3, np.uint8)
        E = PlaneProjector(Volume(M)).project(make_geom([[0, 0, 0], [0, 5, 0], [5, 5, 0]]))
        assert E.shape == (5, 5)
        assert set(np.unique(E)) <= {0, 3}

    # ---- vs. naive reference (randomised) --------------------------------- #
    @pytest.mark.parametrize("seed", range(25))
    def test_random_oblique_planes_match_reference(self, seed):
        rng = np.random.default_rng(seed)
        shape = tuple(rng.integers(8, 28, 3))              # (z, y, x)
        M = rng.integers(1, 65535, shape, dtype=np.uint16)
        nz, ny, nx = shape
        pts = rng.uniform([-6, -6, -6], [nx + 6, ny + 6, nz + 6], (3, 3))
        geom = make_geom(pts)
        # keep test cheap
        if geom.rows * geom.cols > 4000:
            pts[1] = pts[0] + (pts[1] - pts[0]) * 0.3
            pts[2] = pts[1] + (pts[2] - pts[1]) * 0.3
            geom = make_geom(pts)
        np.testing.assert_array_equal(PlaneProjector(Volume(M)).project(geom),
                                      reference_project(M, geom))

    @pytest.mark.parametrize("seed", range(8))
    def test_random_trapezoids_match_reference(self, seed):
        rng = np.random.default_rng(100 + seed)
        M = rng.integers(1, 255, (15, 15, 15), dtype=np.uint8)
        a = rng.uniform(0, 14, 3)
        b = a + rng.uniform(-6, 6, 3)
        c = b + rng.uniform(-6, 6, 3)
        a2 = a + (c - b)
        geom = make_geom([a, b, a2, c])
        np.testing.assert_array_equal(PlaneProjector(Volume(M)).project(geom),
                                      reference_project(M, geom))

    @pytest.mark.parametrize("angle", range(0, 181, 15))
    def test_rotating_plane_through_centre_matches_reference(self, angle):
        M = ramp_volume(21, 21, 21, np.uint8)
        c = np.array([10.0, 10.0, 10.0])
        th = math.radians(angle)
        w = np.array([math.cos(th), math.sin(th), 0.0])
        a = c - 9 * np.array([0, 0, 1.0]) - 9 * w
        b = a + 18 * np.array([0, 0, 1.0])
        cc = b + 18 * w
        geom = make_geom([a, b, cc])
        np.testing.assert_array_equal(PlaneProjector(Volume(M)).project(geom),
                                      reference_project(M, geom))

    def test_constant_volume_inside_is_constant(self):
        M = np.full((20, 20, 20), 42, np.uint8)
        E = PlaneProjector(Volume(M)).project(make_geom([[2, 2, 2], [2, 15, 9], [15, 15, 16]]))
        assert set(np.unique(E)) <= {0, 42} and (E == 42).sum() > 0

    def test_known_sphere_cross_section_is_disc(self):
        n = 41
        z, y, x = np.mgrid[0:n, 0:n, 0:n]
        M = (((x - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2) <= 15 ** 2).astype(np.uint8) * 200
        E = PlaneProjector(Volume(M)).project(make_geom([[0, 0, 20], [0, 40, 20], [40, 40, 20]]))
        area = (E > 0).sum()
        assert area == pytest.approx(math.pi * 15 ** 2, rel=0.05)
        # centroid near the centre
        rr, cc = np.nonzero(E)
        assert abs(rr.mean() - 20) < 1 and abs(cc.mean() - 20) < 1

    def test_any_tilt_of_centred_sphere_gives_same_area(self):
        n = 41
        z, y, x = np.mgrid[0:n, 0:n, 0:n]
        M = (((x - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2) <= 12 ** 2).astype(np.uint8)
        proj = PlaneProjector(Volume(M))
        areas = []
        for deg in range(0, 90, 15):
            th = math.radians(deg)
            u = np.array([0, math.cos(th), math.sin(th)])
            w = np.array([1.0, 0, 0])
            a = np.array([20.0, 20, 20]) - 18 * u - 18 * w
            geom = make_geom([a, a + 36 * u, a + 36 * u + 36 * w])
            areas.append(int(proj.project(geom).sum()))
        ideal = math.pi * 12 ** 2        # a centred cut through a sphere is always a disc
        for a in areas:                  # nearest-neighbour sampling allows a few % noise
            assert a == pytest.approx(ideal, rel=0.08)

    # ---- scale / speed ---------------------------------------------------- #
    def test_large_flat_index_no_int32_overflow(self):
        """Index arithmetic near the end of a ~134M-voxel volume stays correct."""
        n = 512
        M = np.zeros((n, n, n), np.uint8)
        M[n - 1, n - 1, n - 1] = 77
        M[n - 1, n - 1, n - 2] = 66
        E = PlaneProjector(Volume(M)).project(
            make_geom([[n - 2, n - 1, n - 1], [n - 2, n, n - 1], [n, n, n - 1]]))
        assert E.shape == (1, 2) and E[0, 0] == 66 and E[0, 1] == 77

    def test_performance_500_cubed_target(self):
        """Spec target: high FPS on ~500^3 byte volumes (generous CI-safe bound)."""
        n = 500
        M = np.random.default_rng(0).integers(0, 255, (n, n, n), dtype=np.uint8)
        proj = PlaneProjector(Volume(M))
        geoms = [make_geom([[0, 0, z], [0, n - 1, z], [n - 1, n - 1, z]]) for z in range(0, n, 5)]
        proj.project(geoms[0])                      # warm-up
        t = time.perf_counter()
        for g in geoms:
            proj.project(g)
        fps = len(geoms) / (time.perf_counter() - t)
        assert fps > 15, f"too slow: {fps:.1f} FPS"


# =========================================================================== #
# 5. VideoRenderer
# =========================================================================== #

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
        assert out.dtype == np.uint8
        assert out[0, 0] == 0 and out[0, 2] == 255 and abs(int(out[0, 1]) - 127) <= 1
        v.close()

    def test_scaling_clips_outside_range(self, tmp_path):
        v = self._make(tmp_path, vmin=100, vmax=200, dtype=np.uint16)
        out = v._to_u8(np.array([[0, 100, 200, 60000]], np.uint16))
        assert list(out[0]) == [0, 0, 255, 255]
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
        assert c.shape == (20, 30)
        assert np.all(c[5:15, 10:20] == 200)
        assert c[:5].sum() == 0 and c[15:].sum() == 0
        assert c[:, :10].sum() == 0 and c[:, 20:].sum() == 0
        v.close()

    def test_canvas_cleared_between_frames(self, tmp_path):
        v = self._make(tmp_path, (20, 30))
        v.add_frame(np.full((20, 30), 255, np.uint8))
        v.add_frame(np.full((4, 4), 9, np.uint8))
        assert (v._canvas > 0).sum() == 16        # only the small frame remains
        v.close()

    def test_video_brightness_roughly_preserved(self, tmp_path):
        with self._make(tmp_path, (64, 64)) as v:
            v.add_frame(np.full((64, 64), 40, np.uint8))
            v.add_frame(np.full((64, 64), 220, np.uint8))
        frames, _ = read_frames(tmp_path / "o.mp4")
        assert frames[0].mean() == pytest.approx(40, abs=12)
        assert frames[1].mean() == pytest.approx(220, abs=12)

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
# 6. Pipeline & CLI
# =========================================================================== #

@pytest.fixture
def sample_files(tmp_path):
    z, y, x = np.mgrid[0:30, 0:40, 0:50]
    vol = (((x - 25) ** 2 + (y - 20) ** 2 + (z - 15) ** 2) < 12 ** 2).astype(np.uint16) * 50000
    nrrd = write_nrrd(tmp_path / "vol.nrrd", vol, encoding="gzip")
    csv = tmp_path / "planes.csv"
    rows = [f"0,0,{k},0,39,{k},49,39,{k}" for k in range(0, 30, 3)]      # 10 axial planes
    csv.write_text("\n".join(rows) + "\n")
    return nrrd, csv, tmp_path / "out" / "result.mp4"


class TestPipeline:
    def test_frame_count_matches_valid_csv_lines(self, sample_files):
        nrrd, csv, mp4 = sample_files
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 10
        frames, _ = read_frames(mp4)
        assert len(frames) == 10

    def test_creates_missing_output_directories(self, sample_files, tmp_path):
        nrrd, csv, _ = sample_files
        deep = tmp_path / "a" / "b" / "c" / "r.mp4"
        run_pipeline(str(nrrd), str(csv), str(deep))
        assert deep.is_file() and deep.stat().st_size > 0

    def test_frame_size_is_largest_plane_padded_even(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "mixed.csv"
        csv.write_text("0,0,5,0,10,5,20,10,5\n0,0,6,0,29,6,40,29,6\n")   # 10x20 and 29x40
        run_pipeline(str(nrrd), str(csv), str(mp4))
        _, (w, h, _) = read_frames(mp4)
        assert (w, h) == (40, 30)

    def test_frames_show_sphere_growing_then_shrinking(self, sample_files):
        nrrd, csv, mp4 = sample_files
        run_pipeline(str(nrrd), str(csv), str(mp4))
        frames, _ = read_frames(mp4)
        bright = [f.mean() for f in frames]            # z = 0,3,...,27; sphere centred z=15
        assert bright[0] < 5 and bright[-1] < 5        # outside the sphere
        assert max(bright) > 20
        assert bright.index(max(bright)) in (4, 5)     # z=12 or 15

    def test_order_follows_csv(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "rev.csv"
        csv.write_text("0,0,15,0,39,15,49,39,15\n0,0,0,0,39,0,49,39,0\n")  # bright then dark
        run_pipeline(str(nrrd), str(csv), str(mp4))
        frames, _ = read_frames(mp4)
        assert frames[0].mean() > frames[1].mean() + 10

    def test_fps_is_applied(self, sample_files):
        nrrd, csv, mp4 = sample_files
        run_pipeline(str(nrrd), str(csv), str(mp4), fps=50)
        _, (_, _, fps) = read_frames(mp4)
        assert fps == pytest.approx(50, abs=1)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_all_supported_voxel_sizes_end_to_end(self, tmp_path, dtype):
        info = np.iinfo(dtype)
        rng = np.random.default_rng(0)
        vol = rng.integers(0, info.max, (12, 14, 16), dtype=dtype, endpoint=True)
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

    def test_planes_outside_volume_still_produce_black_frames(self, sample_files, tmp_path):
        nrrd, _, mp4 = sample_files
        csv = tmp_path / "out.csv"
        csv.write_text("500,500,500,500,520,500,520,520,500\n")
        assert run_pipeline(str(nrrd), str(csv), str(mp4)) == 1
        assert read_frames(mp4)[0][0].mean() < 3


class TestMain:
    def test_success_returns_zero(self, sample_files):
        nrrd, csv, mp4 = sample_files
        assert main([str(nrrd), str(csv), str(mp4)]) == 0
        assert mp4.is_file()

    def test_fps_option(self, sample_files):
        nrrd, csv, mp4 = sample_files
        assert main([str(nrrd), str(csv), str(mp4), "--fps", "100"]) == 0
        assert read_frames(mp4)[1][2] == pytest.approx(100, abs=2)

    def test_missing_nrrd_returns_one(self, sample_files, capsys, caplog):
        _, csv, mp4 = sample_files
        assert main(["nope.nrrd", str(csv), str(mp4)]) == 1
        assert not mp4.exists()

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

    def test_cli_as_subprocess(self, sample_files):
        import subprocess, sys
        nrrd, csv, mp4 = sample_files
        r = subprocess.run([sys.executable, str(Path(Ex0.__file__)), str(nrrd), str(csv), str(mp4)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert mp4.is_file()
