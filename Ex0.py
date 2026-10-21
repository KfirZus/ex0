#!/usr/bin/env python3
"""
Ex0.py - 3D Volume Plane Projection & Visualization

Usage:
    python Ex0.py <file.nrrd> <file.csv> <file.mp4> [--fps N]

Pipeline:
    1. Load a 3D NRRD volume M (uint8 / uint16 / uint32 voxels).
    2. Parse a CSV where each line holds 3 points (isosceles triangle a,b,c)
       or 4 points (isosceles trapezoid a',b,a'',c), as x,y,z voxel coordinates.
    3. For each plane, sample M on a regular grid spanning the plane and build
       a 2D matrix E of size Dab x Dbc (out-of-bounds samples are 0).
    4. Write every E as one frame of an MP4 video, in CSV order.

Geometry convention (single place to change: `PlaneGeometry.from_points`):
    origin = first point (a, or a' for trapezoids)
    u      = unit vector a -> b          (rows of E,    length Dab = |ab|)
    w      = unit vector b -> c          (columns of E, length Dbc = |bc|)
    E[i, j] = M[origin + i*u + j*w]      (nearest-neighbour sampling)
  For a trapezoid the second top vertex a'' is validated but not needed for
  sampling, since the (a', b, c) corner already fixes the plane and its extent.

Coordinates in the CSV are interpreted as (x, y, z) where x is the fastest
varying NRRD axis (the first entry of the NRRD `sizes` field).
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

LOG = logging.getLogger("Ex0")

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

_NRRD_TYPES = {
    "uchar": np.uint8, "unsigned char": np.uint8, "uint8": np.uint8, "uint8_t": np.uint8,
    "ushort": np.uint16, "unsigned short": np.uint16, "unsigned short int": np.uint16,
    "uint16": np.uint16, "uint16_t": np.uint16,
    "uint": np.uint32, "unsigned int": np.uint32, "uint32": np.uint32, "uint32_t": np.uint32,
    # Signed types are tolerated; display normalisation handles them.
    "signed char": np.int8, "int8": np.int8, "int8_t": np.int8, "char": np.int8,
    "short": np.int16, "short int": np.int16, "signed short": np.int16,
    "int16": np.int16, "int16_t": np.int16,
    "int": np.int32, "signed int": np.int32, "int32": np.int32, "int32_t": np.int32,
}


@dataclass(frozen=True)
class Volume:
    """3D scalar volume. `data` is shaped (z, y, x) (C order, x fastest)."""
    data: np.ndarray

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        z, y, x = self.data.shape
        return x, y, z


@dataclass(frozen=True)
class Plane:
    """A plane given by 3 (triangle) or 4 (trapezoid) 3D points."""
    points: np.ndarray  # shape (3,3) or (4,3), float64
    line_no: int = 0


@dataclass(frozen=True)
class PlaneGeometry:
    """Sampling frame of a plane: origin, two unit vectors and grid size."""
    origin: np.ndarray
    u: np.ndarray       # unit vector a -> b (varies along rows)
    w: np.ndarray       # unit vector b -> c (varies along columns)
    rows: int           # Dab
    cols: int           # Dbc

    @staticmethod
    def from_points(points: np.ndarray) -> "PlaneGeometry":
        a, b = points[0], points[1]
        c = points[2] if len(points) == 3 else points[3]
        ab, bc = b - a, c - b
        dab, dbc = float(np.linalg.norm(ab)), float(np.linalg.norm(bc))
        u = ab / dab if dab > 0 else np.zeros(3)
        w = bc / dbc if dbc > 0 else np.zeros(3)
        return PlaneGeometry(
            origin=a.astype(np.float32),
            u=u.astype(np.float32),
            w=w.astype(np.float32),
            rows=max(1, int(round(dab))),
            cols=max(1, int(round(dbc))),
        )


# --------------------------------------------------------------------------- #
# Loading / parsing
# --------------------------------------------------------------------------- #

class NrrdReader:
    """Minimal, dependency-free NRRD reader for 3D raw/gzip volumes."""

    @staticmethod
    def read(path: str) -> Volume:
        with open(path, "rb") as f:
            magic = f.readline().decode("ascii", "replace").strip()
            if not magic.startswith("NRRD"):
                raise ValueError(f"{path}: not an NRRD file (bad magic '{magic}')")
            header = NrrdReader._read_header(f)
            offset = f.tell()

        dtype = NrrdReader._dtype(header)
        sizes = [int(s) for s in header["sizes"].split()]
        if int(header.get("dimension", len(sizes))) != 3 or len(sizes) != 3:
            raise ValueError(f"{path}: only 3D volumes are supported (sizes={sizes})")
        if "data file" in header or "datafile" in header:
            raise ValueError("Detached NRRD data files are not supported")

        encoding = header.get("encoding", "raw").lower()
        count = sizes[0] * sizes[1] * sizes[2]
        skip = int(header.get("byte skip", 0))

        if encoding == "raw":
            arr = np.fromfile(path, dtype=dtype, count=count, offset=offset + max(skip, 0))
        elif encoding in ("gzip", "gz"):
            with open(path, "rb") as f:
                f.seek(offset)
                raw = gzip.decompress(f.read())
            arr = np.frombuffer(raw, dtype=dtype, count=count, offset=max(skip, 0))
        else:
            raise ValueError(f"Unsupported NRRD encoding '{encoding}'")

        if arr.size != count:
            raise ValueError(f"{path}: expected {count} voxels, found {arr.size}")
        # NRRD sizes[0] is the fastest axis -> numpy shape is reversed.
        data = arr.reshape(sizes[::-1])
        return Volume(data=np.ascontiguousarray(data))

    @staticmethod
    def _read_header(f) -> dict:
        header = {}
        while True:
            line = f.readline()
            if not line or line.strip() == b"":
                break
            text = line.decode("ascii", "replace").rstrip("\r\n")
            if text.startswith("#"):
                continue
            if ":=" in text:  # key/value pair, ignored
                continue
            if ":" in text:
                k, v = text.split(":", 1)
                header[k.strip().lower()] = v.strip()
        if "sizes" not in header or "type" not in header:
            raise ValueError("NRRD header missing 'sizes' or 'type'")
        return header

    @staticmethod
    def _dtype(header: dict) -> np.dtype:
        key = header["type"].lower()
        if key not in _NRRD_TYPES:
            raise ValueError(f"Unsupported NRRD type '{header['type']}'")
        dt = np.dtype(_NRRD_TYPES[key])
        if dt.itemsize > 1:
            endian = header.get("endian", "little").lower()
            dt = dt.newbyteorder("<" if endian == "little" else ">")
        return dt


class PlaneCsvParser:
    """Parses a CSV of planes: 9 values (triangle) or 12 values (trapezoid) per line."""

    @staticmethod
    def parse(path: str) -> List[Plane]:
        planes: List[Plane] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                plane = PlaneCsvParser.parse_line(line, line_no)
                if plane is not None:
                    planes.append(plane)
        if not planes:
            raise ValueError(f"{path}: no valid planes found")
        return planes

    @staticmethod
    def parse_line(line: str, line_no: int = 0) -> Optional[Plane]:
        text = line.strip()
        if not text or text.startswith("#"):
            return None
        try:
            vals = [float(t) for t in text.replace(";", ",").replace("\t", ",").split(",") if t.strip()]
        except ValueError:
            LOG.warning("CSV line %d skipped (non-numeric, e.g. header): %r", line_no, text[:60])
            return None
        if len(vals) not in (9, 12) or not np.all(np.isfinite(vals)):
            LOG.warning("CSV line %d skipped (expected 9 or 12 finite values, got %d)", line_no, len(vals))
            return None
        return Plane(np.asarray(vals, dtype=np.float64).reshape(-1, 3), line_no)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

class PlaneProjector:
    """Samples a volume along planes (nearest neighbour, zero outside bounds)."""

    def __init__(self, volume: Volume):
        self._flat = volume.data.reshape(-1)          # view, no copy
        self.nx, self.ny, self.nz = volume.shape_xyz
        self._sy = self.nx
        self._sz = self.nx * self.ny
        self._grid_cache: dict = {}

    def _grid(self, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (rows, cols)
        g = self._grid_cache.get(key)
        if g is None:
            if len(self._grid_cache) > 64:
                self._grid_cache.clear()
            g = (np.arange(rows, dtype=np.float32)[:, None],
                 np.arange(cols, dtype=np.float32)[None, :])
            self._grid_cache[key] = g
        return g

    def project(self, geom: PlaneGeometry) -> np.ndarray:
        """Return E (rows x cols) with the volume's dtype."""
        I, J = self._grid(geom.rows, geom.cols)
        o, u, w = geom.origin, geom.u, geom.w

        # p = origin + i*u + j*w for each axis, rounded to the nearest voxel.
        ix = np.rint(o[0] + I * u[0] + J * w[0]).astype(np.int32)
        iy = np.rint(o[1] + I * u[1] + J * w[1]).astype(np.int32)
        iz = np.rint(o[2] + I * u[2] + J * w[2]).astype(np.int32)

        # Unsigned reinterpretation turns "0 <= i < n" into a single compare.
        valid = ((ix.view(np.uint32) < self.nx)
                 & (iy.view(np.uint32) < self.ny)
                 & (iz.view(np.uint32) < self.nz))

        flat_idx = iz * self._sz + iy * self._sy + ix
        flat_idx[~valid] = 0                         # keep take() in range
        E = self._flat.take(flat_idx)
        E[~valid] = 0                                # out of bounds -> v = 0
        return E


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

class VideoRenderer:
    """Writes 2D matrices as grayscale frames of a fixed-size MP4 video."""

    def __init__(self, path: str, frame_size: Tuple[int, int], fps: float,
                 vmin: float, vmax: float, dtype: np.dtype):
        if cv2 is None:
            raise RuntimeError("OpenCV is required: pip install opencv-python")
        # Frame size = (height, width); force even dimensions for codec compatibility.
        self.h = frame_size[0] + (frame_size[0] & 1)
        self.w = frame_size[1] + (frame_size[1] & 1)
        self._canvas = np.zeros((self.h, self.w), dtype=np.uint8)
        self._is_u8 = np.dtype(dtype) == np.uint8
        self._scale = 255.0 / (vmax - vmin) if vmax > vmin else 0.0
        self._vmin = float(vmin)
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{path}'")
        self.frames = 0

    def _to_u8(self, E: np.ndarray) -> np.ndarray:
        if self._is_u8:
            return E
        f = (E.astype(np.float32) - self._vmin) * np.float32(self._scale)
        np.clip(f, 0, 255, out=f)
        return f.astype(np.uint8)

    def add_frame(self, E: np.ndarray) -> None:
        img = self._to_u8(E)
        h, w = img.shape
        self._canvas.fill(0)
        r0, c0 = (self.h - h) // 2, (self.w - w) // 2
        self._canvas[r0:r0 + h, c0:c0 + w] = img
        self._writer.write(cv2.cvtColor(self._canvas, cv2.COLOR_GRAY2BGR))
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Pipeline / CLI
# --------------------------------------------------------------------------- #

def run_pipeline(nrrd_path: str, csv_path: str, mp4_path: str, fps: float = 30.0) -> int:
    t0 = time.perf_counter()
    volume = NrrdReader.read(nrrd_path)
    planes = PlaneCsvParser.parse(csv_path)
    LOG.info("Loaded volume %s (%s) and %d planes in %.2fs",
             volume.shape_xyz, volume.data.dtype, len(planes), time.perf_counter() - t0)

    geoms = [PlaneGeometry.from_points(p.points) for p in planes]
    frame_h = max(g.rows for g in geoms)       # common canvas: pad smaller planes
    frame_w = max(g.cols for g in geoms)

    vmin, vmax = float(volume.data.min()), float(volume.data.max())
    projector = PlaneProjector(volume)

    # Create the output folder (e.g. "out/") if it doesn't exist yet.
    Path(mp4_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    t1 = time.perf_counter()
    with VideoRenderer(mp4_path, (frame_h, frame_w), fps, vmin, vmax, volume.data.dtype) as video:
        for geom in geoms:
            video.add_frame(projector.project(geom))
        n = video.frames
    dt = time.perf_counter() - t1
    LOG.info("Rendered %d frames in %.2fs (%.1f FPS) -> %s", n, dt, n / dt if dt else float("inf"), mp4_path)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render plane cross-sections of a 3D NRRD volume to MP4.")
    ap.add_argument("nrrd", help="input 3D NRRD volume")
    ap.add_argument("csv", help="input CSV of planes (3 or 4 xyz points per line)")
    ap.add_argument("mp4", help="output MP4 video")
    ap.add_argument("--fps", type=float, default=30.0, help="playback frame rate (default 30)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        for p in (args.nrrd, args.csv):
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        run_pipeline(args.nrrd, args.csv, args.mp4, args.fps)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())