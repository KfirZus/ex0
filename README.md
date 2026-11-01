[README.md](https://github.com/user-attachments/files/34246665/README.md)
# 3D Volume Plane Projection

Renders cross-sections of a 3D volume (NRRD) along planes given in a CSV file and writes them
as frames of an MP4 video.

**Authors:** Kfir Zusman Artyom Radomishelsky

## Quick start

```
python -m pip install -r requirements.txt
python Ex0.py <file.nrrd> <file.csv> <file.mp4> [--fps N]
```

On Windows, use `py` instead of `python` if `python` is not recognised.

| Argument | Meaning |
|---|---|
| `file.nrrd` | 3D volume with 8, 16 or 32-bit unsigned integer voxels (raw or gzip encoded) |
| `file.csv` | one plane per line: 3 points (triangle `a,b,c`) or 4 points (trapezoid `a',b,a'',c`), each as `x,y,z` |
| `file.mp4` | output video, one frame per plane, in CSV order (the folder is created if missing) |
| `--fps N` | optional playback speed (default 30) |

Example CSV line (a triangle): `0,0,10, 0,255,10, 255,255,10`

## How it works

1. `NrrdReader` loads the volume `M`; `PlaneCsvParser` streams the planes from the CSV.
2. `PlaneGeometry` turns the points into an origin and two unit vectors (`a→b` and `b→c`).
3. `PlaneProjector` builds the image `E` (size `|ab| × |bc|`) by sampling `M` at
   `origin + i·u + j·w` (nearest voxel). Positions outside the volume give `0`.
4. `VideoRenderer` scales values to 8 bits and writes each `E` as an MP4 frame.

The CSV is streamed, so memory does not grow with the number of planes.

## Bad input

Invalid CSV lines are **skipped with a warning** and the rest still renders. This covers:
wrong number of values, non-numeric text, headers, NaN/inf, a trapezoid whose 4th point is not
in the plane of the other three, degenerate planes (identical or collinear points) and planes
far larger than the volume. If no valid line remains, the program exits with code 1.

## Tests

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests follow the Software Test Documentation: smoke, unit, end-to-end, regression and
performance/load/stress. See [TESTING.md](TESTING.md) for markers, full-scale runs and
how to update the regression hash. `python generate_fixtures.py` writes sample NRRD/CSV files.

Tests run automatically before every commit (`python install_hooks.py` once per clone) and on
every push through GitHub Actions (`.github/workflows/tests.yml`).

## Files

```
Ex0.py                 the program
test_Ex0.py            test suite
generate_fixtures.py   sample data generator (also used by the tests)
install_hooks.py       enables the pre-commit test hook
.githooks/pre-commit   runs the fast tests before each commit
.github/workflows/     CI on GitHub
pytest.ini, requirements*.txt, TESTING.md
```

## AI assistance

This project was developed with the help of Claude (Anthropic's AI assistant), which was used to
draft the code, tests and documentation. All of it was reviewed, tested, verified and approved
by the authors, who take responsibility for the final submission.
