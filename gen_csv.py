#!/usr/bin/env python3
"""
gen_csv.py - generate a planes CSV for Ex0.py from an NRRD file's size.

Usage:
    python gen_csv.py <file.nrrd> <planes.csv> [--frames N]

Creates N planes sweeping through the volume from the first to the last z-slice.
Each line is one triangle: a, b, c  (x,y,z each), matching Ex0.py's convention:
    a = (0, 0, z)           origin
    b = (0, ny-1, z)        rows run a -> b  (along y)
    c = (nx-1, ny-1, z)     columns run b -> c (along x)
so each frame is a full ny x nx slice.
"""
import argparse
from pathlib import Path


def read_sizes(path):
    with open(path, "rb") as f:
        if not f.readline().startswith(b"NRRD"):
            raise ValueError("not an NRRD file")
        while True:
            line = f.readline()
            if not line or not line.strip():
                break
            text = line.decode("ascii", "replace")
            if text.lower().startswith("sizes:"):
                return [int(s) for s in text.split(":", 1)[1].split()]
    raise ValueError("no 'sizes' field in NRRD header")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nrrd")
    ap.add_argument("csv")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    nx, ny, nz = read_sizes(args.nrrd)
    n = max(1, args.frames)
    Path(args.csv).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w") as f:
        for i in range(n):
            z = round(i * (nz - 1) / (n - 1)) if n > 1 else nz // 2
            f.write(f"0,0,{z},0,{ny - 1},{z},{nx - 1},{ny - 1},{z}\n")
    print(f"Volume {nx}x{ny}x{nz}: wrote {n} planes to {args.csv}")


if __name__ == "__main__":
    main()
