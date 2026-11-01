#!/usr/bin/env python3
"""
generate_fixtures.py - creates the NRRD / CSV test data named in the Software Test
Documentation (STD).  Also imported by test_Ex0.py, so there is one source of truth.

Usage:
    python generate_fixtures.py [--out fixtures] [--sizes 500 1023] [--skip-large]

Volumes  (gzip-compressed NRRD, deterministic phantom, x-fastest as per NRRD):
    fixture_vol_8bit_<N>.nrrd     uint8
    fixture_vol_16bit_<N>.nrrd    uint16
    fixture_vol_32bit_<N>.nrrd    uint32
    fixture_vol_oversized_2048_stub.nrrd   header claims 2048^3 uint32, almost no data (STR-01)

Trajectories (CSV, x,y,z per point):
    fixture_traj_triangle.csv      100 triangles  (a, b, c)               E2E-01
    fixture_traj_trapezoid.csv      50 trapezoids (a', b, a'', c)         E2E-02
    fixture_malformed_traj.csv      missing commas / coords, strings, blank lines, null bytes (UT-Parse-04)
    fixture_traj_reference.csv      30 fixed planes for the regression hash (REG-01)
    fixture_traj_300.csv           300 planes (PF-01)
    fixture_traj_heavy_5000.csv   5000 planes (LD-01)
    fixture_traj_stress_500000.csv  500,000 planes (STR-02)

Every plane is built so that a->b is perpendicular to b->c (right isosceles triangle /
rectangle-based trapezoid) and its centre lies inside the volume, matching the sampling
convention of Ex0.py (rows along a->b, columns along b->c).
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

NRRD_NAME = {np.dtype(np.uint8): "uchar", np.dtype(np.uint16): "ushort", np.dtype(np.uint32): "uint"}
MAX_VALUE = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535, np.dtype(np.uint32): 4_000_000_000}


# --------------------------------------------------------------------------- #
# Volumes
# --------------------------------------------------------------------------- #

def phantom_slice(z: int, nx: int, ny: int, nz: int, dtype) -> np.ndarray:
    """One z-slice (shape ny x nx) of a deterministic phantom.

    Nested spheres + a bar + a small periodic texture. Pure integer arithmetic, so
    the result is bit-identical on every platform (needed for the REG-01 hash).
    """
    dtype = np.dtype(dtype)
    step = MAX_VALUE[dtype] // 5
    tex_unit = max(1, step // 32)

    y = np.arange(ny, dtype=np.int64)[:, None]
    x = np.arange(nx, dtype=np.int64)[None, :]
    X, Y, Z = 2 * x - (nx - 1), 2 * y - (ny - 1), 2 * z - (nz - 1)      # centred, doubled
    r2 = X * X + Y * Y + Z * Z
    half = (min(nx, ny, nz) - 1)                                        # doubled radius scale

    level = np.zeros((ny, nx), dtype=np.int64)
    for frac, lv in ((0.90, 1), (0.70, 2), (0.40, 3)):
        R = int(frac * half)
        level = np.where(r2 < R * R, lv, level)
    bar = (np.abs(X) < max(2, half // 8)) & (abs(Z) < max(2, half // 8))
    level = np.where(bar, 4, level)

    tex = ((x + 2 * y + 3 * z) % 16) * tex_unit
    out = np.where(level > 0, level * step + tex, 0)
    return out.astype(dtype)


def build_phantom(nx: int, ny: int, nz: int, dtype) -> np.ndarray:
    """Whole phantom in memory, shape (nz, ny, nx)."""
    vol = np.empty((nz, ny, nx), dtype=dtype)
    for z in range(nz):
        vol[z] = phantom_slice(z, nx, ny, nz, dtype)
    return vol


def nrrd_header(nx: int, ny: int, nz: int, dtype, encoding: str) -> bytes:
    dtype = np.dtype(dtype)
    lines = ["NRRD0004", f"type: {NRRD_NAME[dtype]}", "dimension: 3", f"sizes: {nx} {ny} {nz}"]
    if dtype.itemsize > 1:
        lines.append("endian: little")
    lines += [f"encoding: {encoding}", "", ""]
    return "\n".join(lines).encode("ascii")


def write_nrrd_array(path, arr: np.ndarray, encoding: str = "raw") -> Path:
    """Write an in-memory (z, y, x) array."""
    path = Path(path)
    nz, ny, nx = arr.shape
    little = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False)
    payload = little.tobytes()
    with open(path, "wb") as f:
        f.write(nrrd_header(nx, ny, nz, arr.dtype, encoding))
        f.write(gzip.compress(payload, 1) if encoding == "gzip" else payload)
    return path


def write_phantom_nrrd(path, n, dtype, encoding: str = "gzip") -> Path:
    """Stream a cubic n^3 phantom to disk slice by slice (low memory, any size)."""
    path = Path(path)
    nx = ny = nz = int(n)
    dtype = np.dtype(dtype)
    with open(path, "wb") as f:
        f.write(nrrd_header(nx, ny, nz, dtype, encoding))
        sink = gzip.GzipFile(fileobj=f, mode="wb", compresslevel=1) if encoding == "gzip" else f
        for z in range(nz):
            sink.write(phantom_slice(z, nx, ny, nz, dtype).astype(dtype.newbyteorder("<")).tobytes())
        if encoding == "gzip":
            sink.close()
    return path


def write_oversized_stub(path, n: int = 2048) -> Path:
    """NRRD whose header claims n^3 uint32 voxels but holds almost no data (STR-01)."""
    path = Path(path)
    with open(path, "wb") as f:
        f.write(nrrd_header(n, n, n, np.uint32, "raw"))
        f.write(bytes(1024))
    return path


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #

def _fmt(points) -> str:
    return ",".join(f"{v:.3f}".rstrip("0").rstrip(".") for p in points for v in p)


def _random_frame(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Two random orthonormal vectors (u, w)."""
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    w = rng.normal(size=3)
    w -= u * (w @ u)
    w /= np.linalg.norm(w)
    return u, w


def make_triangle(rng, n: int):
    """Right isosceles triangle (legs |ab| = |bc|) centred inside an n^3 volume."""
    leg = rng.uniform(0.15, 0.5) * n
    u, w = _random_frame(rng)
    centre = rng.uniform(0.3 * n, 0.7 * n, 3)
    a = centre - (u + w) * leg / 2
    b = a + u * leg
    c = b + w * leg
    return [a, b, c]


def make_trapezoid(rng, n: int):
    """Isosceles trapezoid (a', b, a'', c): a'a'' parallel to bc, legs a'b = a''c."""
    base = rng.uniform(0.25, 0.55) * n
    top = base * rng.uniform(0.4, 0.8)
    height = rng.uniform(0.15, 0.4) * n
    w, nrm = _random_frame(rng)
    centre = rng.uniform(0.3 * n, 0.7 * n, 3)
    b = centre - w * base / 2 - nrm * height / 2
    c = b + w * base
    a1 = b + w * (base - top) / 2 + nrm * height
    a2 = a1 + w * top
    return [a1, b, a2, c]


def write_trajectory(path, count: int, kind: str, n: int, seed: int) -> Path:
    """kind: 'triangle' | 'trapezoid'.  Streams to disk (fine for 500,000 lines)."""
    rng = np.random.default_rng(seed)
    make = make_triangle if kind == "triangle" else make_trapezoid
    with open(path, "w", newline="\n") as f:
        for _ in range(count):
            f.write(_fmt(make(rng, n)) + "\n")
    return Path(path)


def write_reference_trajectory(path, n: int = 500) -> Path:
    """Fixed, hand-structured planes for the regression hash (REG-01)."""
    lines = []
    c = (n - 1) / 2
    for k in range(10):                                     # axial sweep
        z = round(k * (n - 1) / 9)
        lines.append(_fmt([(0, 0, z), (0, n - 1, z), (n - 1, n - 1, z)]))
    for k in range(5):                                      # coronal sweep
        y = round(k * (n - 1) / 4)
        lines.append(_fmt([(0, y, 0), (0, y, n - 1), (n - 1, y, n - 1)]))
    for k in range(5):                                      # sagittal sweep
        x = round(k * (n - 1) / 4)
        lines.append(_fmt([(x, 0, 0), (x, 0, n - 1), (x, n - 1, n - 1)]))
    for deg in (15, 30, 45, 60, 75, 90, 120, 150, 165, 180)[:10]:   # rotating vertical plane
        th = np.radians(deg)
        w = np.array([np.cos(th), np.sin(th), 0.0])
        a = np.array([c, c, 0.0]) - w * (n - 1) / 2
        b = a + np.array([0, 0, n - 1.0])
        lines.append(_fmt([a, b, b + w * (n - 1)]))
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def write_malformed_trajectory(path) -> Path:
    """Missing commas, missing coordinates, strings, blank lines, null bytes, a
    non-coplanar trapezoid, degenerate planes, parentheses, quotes, trailing text.

    Exactly 3 lines are valid: line 2 (triangle), line 10 (trapezoid), line 11 (triangle).
    """
    rows = [
        b"ax,ay,az,bx,by,bz,cx,cy,cz",                    # 1  header text      -> skipped
        b"10,10,10,10,60,10,60,60,10",                    # 2  valid triangle
        b"10,10,10 10,60,10,60,60,10",                    # 3  missing comma    -> skipped
        b"10,10,10,10,60,,60,60,10",                      # 4  missing coord    -> skipped
        b"abc,def,ghi,jkl,mno,pqr,stu,vwx,yz",            # 5  strings          -> skipped
        b"",                                              # 6  blank
        b"10,10,\x00,10,60,10,60,60,10",                  # 7  null byte        -> skipped
        b"1,2,3",                                         # 8  too few values   -> skipped
        b"nan,0,0,0,50,0,50,50,0",                        # 9  non-finite       -> skipped
        b"20,10,10,10,60,10,40,10,10,60,60,10",           # 10 valid trapezoid
        b"20,20,20,20,70,20,70,70,20",                    # 11 valid triangle
        b"\x00\x00\x00",                                  # 12 only null bytes  -> skipped
        b"20,10,10,10,60,10,40,10,25,60,60,10",           # 13 trapezoid, a'' 15 voxels off the plane -> skipped
        b"5,5,5,5,5,5,5,5,5",                             # 14 all points identical   -> skipped
        b"0,0,0,5,5,5,10,10,10",                          # 15 three collinear points -> skipped
        b"(0,0,0),(0,10,0),(20,10,0)",                    # 16 parentheses            -> skipped
        b'"0","0","0","0","10","0","20","10","0"',        # 17 quoted numbers         -> skipped
        b"0,0,0,0,10,0,20,10,0 # note",                   # 18 trailing text          -> skipped
        b",,,,,,,,,",                                     # 19 only commas            -> skipped
    ]
    Path(path).write_bytes(b"\n".join(rows) + b"\n")
    return Path(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def generate_all(out: Path, sizes=(500, 1023), include_large_traj: bool = True, log=print) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        for bits, dt in ((8, np.uint8), (16, np.uint16), (32, np.uint32)):
            p = out / f"fixture_vol_{bits}bit_{n}.nrrd"
            write_phantom_nrrd(p, n, dt)
            log(f"  {p.name:36s} {p.stat().st_size / 1e6:8.1f} MB")
    write_oversized_stub(out / "fixture_vol_oversized_2048_stub.nrrd")

    n = min(sizes)
    write_trajectory(out / "fixture_traj_triangle.csv", 100, "triangle", n, seed=1)
    write_trajectory(out / "fixture_traj_trapezoid.csv", 50, "trapezoid", n, seed=2)
    write_malformed_trajectory(out / "fixture_malformed_traj.csv")
    write_reference_trajectory(out / "fixture_traj_reference.csv", n)
    write_trajectory(out / "fixture_traj_300.csv", 300, "triangle", n, seed=3)
    write_trajectory(out / "fixture_traj_heavy_5000.csv", 5000, "triangle", n, seed=4)
    if include_large_traj:
        write_trajectory(out / "fixture_traj_stress_500000.csv", 500_000, "triangle", n, seed=5)
    for p in sorted(out.glob("*.csv")) + [out / "fixture_vol_oversized_2048_stub.nrrd"]:
        log(f"  {p.name:36s} {p.stat().st_size / 1e6:8.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate STD test fixtures.")
    ap.add_argument("--out", default="fixtures", help="output folder (default: fixtures)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1023],
                    help="cubic volume edge lengths (default: 500 1023)")
    ap.add_argument("--skip-large", action="store_true", help="skip the 500,000-line CSV")
    args = ap.parse_args()
    print(f"Writing fixtures to {Path(args.out).resolve()}")
    generate_all(Path(args.out), tuple(args.sizes), not args.skip_large)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
