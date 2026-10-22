#!/usr/bin/env python3
"""
gen_csv.py - generate plane CSVs for Ex0.py from any 3D NRRD file(s).

For every NRRD volume, three CSVs are produced - one per sweep direction:

    left_to_right   planes of constant z  (sweep along the 3rd NRRD axis)
    top_to_down     planes of constant y  (sweep along the 2nd NRRD axis)
    front_to_back   planes of constant x  (sweep along the 1st NRRD axis)

(The names match the supplied MRHead_1.nrrd, whose header says axis 3 runs
left->right, axis 2 top->down and axis 1 front->back. For other files they are
simply labels for the z / y / x sweeps.)

Usage:
    python gen_csv.py data                         # every *.nrrd in ./data
    python gen_csv.py a.nrrd b.nrrd -o out_csv     # specific files
    python gen_csv.py a.nrrd --direction top_to_down --frames 100
    python gen_csv.py a.nrrd planes.csv            # legacy: one CSV, left_to_right

Output files are named <stem>_<direction>.csv (default folder: ./csv).

Each CSV line is one triangle a,b,c (x,y,z each), Ex0.py's convention:
rows run a -> b, columns run b -> c, e.g. for left_to_right:
    a = (0, 0, z)   b = (0, ny-1, z)   c = (nx-1, ny-1, z)
"""
import argparse
import sys
from pathlib import Path

# direction -> index of the swept axis in (nx, ny, nz)
DIRECTIONS = {
    "left_to_right": 2,   # z
    "top_to_down": 1,     # y
    "front_to_back": 0,   # x
}


def read_sizes(path):
    """Return [nx, ny, nz] from the NRRD header (works for any 3D NRRD)."""
    with open(path, "rb") as f:
        if not f.readline().startswith(b"NRRD"):
            raise ValueError("not an NRRD file")
        while True:
            line = f.readline()
            if not line or not line.strip():
                break
            text = line.decode("ascii", "replace")
            if text.lower().startswith("sizes:"):
                sizes = [int(s) for s in text.split(":", 1)[1].split()]
                if len(sizes) != 3:
                    raise ValueError(f"only 3D volumes are supported (sizes={sizes})")
                return sizes
    raise ValueError("no 'sizes' field in NRRD header")


def plane_points(direction, k, nx, ny, nz):
    """The three points (a, b, c) of the plane at index k for a given sweep."""
    if direction == "left_to_right":
        return (0, 0, k), (0, ny - 1, k), (nx - 1, ny - 1, k)
    if direction == "top_to_down":
        return (0, k, 0), (0, k, nz - 1), (nx - 1, k, nz - 1)
    if direction == "front_to_back":
        return (k, 0, 0), (k, 0, nz - 1), (k, ny - 1, nz - 1)
    raise ValueError(f"unknown direction '{direction}'")


def write_csv(sizes, direction, frames, csv_path):
    """Write `frames` evenly spaced planes sweeping the whole volume."""
    nx, ny, nz = sizes
    length = sizes[DIRECTIONS[direction]]
    n = max(1, frames)
    Path(csv_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        for i in range(n):
            k = round(i * (length - 1) / (n - 1)) if n > 1 else length // 2
            pts = plane_points(direction, k, nx, ny, nz)
            f.write(",".join(str(v) for p in pts for v in p) + "\n")
    return n


def collect_inputs(paths):
    """Expand files / folders into a list of .nrrd files."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(q for q in p.iterdir() if q.suffix.lower() == ".nrrd")
            if not found:
                print(f"warning: no .nrrd files in {p}", file=sys.stderr)
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(str(p))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate left_to_right / top_to_down / front_to_back plane CSVs "
                    "for one or more NRRD volumes.")
    ap.add_argument("inputs", nargs="+",
                    help=".nrrd files and/or folders containing .nrrd files")
    ap.add_argument("-o", "--out-dir", default="csv",
                    help="output folder (default: ./csv)")
    ap.add_argument("--frames", type=int, default=200,
                    help="planes per CSV (default 200)")
    ap.add_argument("--direction", choices=["all", *DIRECTIONS], default="all",
                    help="which sweep to generate (default: all three)")
    args = ap.parse_args(argv)

    # Legacy form: gen_csv.py <file.nrrd> <planes.csv>  -> one CSV (left_to_right)
    legacy_csv = None
    inputs = args.inputs
    if len(inputs) == 2 and inputs[1].lower().endswith(".csv"):
        inputs, legacy_csv = inputs[:1], inputs[1]

    directions = list(DIRECTIONS) if args.direction == "all" else [args.direction]
    if legacy_csv and args.direction == "all":
        directions = ["left_to_right"]

    try:
        files = collect_inputs(inputs)
    except FileNotFoundError as exc:
        print(f"error: not found: {exc}", file=sys.stderr)
        return 1

    failed = 0
    for nrrd in files:
        try:
            sizes = read_sizes(nrrd)
        except (ValueError, OSError) as exc:
            print(f"error: {nrrd}: {exc}", file=sys.stderr)
            failed += 1
            continue
        for d in directions:
            out = legacy_csv or Path(args.out_dir) / f"{nrrd.stem}_{d}.csv"
            n = write_csv(sizes, d, args.frames, out)
            print(f"{nrrd.name} {sizes[0]}x{sizes[1]}x{sizes[2]} [{d}]: wrote {n} planes to {out}")
    return 1 if failed or not files else 0


if __name__ == "__main__":
    sys.exit(main())