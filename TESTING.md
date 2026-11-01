# Testing guide

Tests follow the Software Test Documentation (STD v1.1.0). Test IDs from the STD are in the
test names, e.g. `test_UT_Parse_04_...` is **UT-Parse-04**.

## One-time setup
```
py -m pip install -r requirements-dev.txt     # use `python` instead of `py` if that is what works for you
py install_hooks.py                           # run tests automatically before every commit
```
(`git init` first if the folder is not a git repository yet.)

## Running tests
| Command | What it runs |
|---|---|
| `py -m pytest` | everything (spec sizes scaled down so it finishes in seconds) |
| `py -m pytest -m smoke` | ST-01..02 |
| `py -m pytest -m unit` | UT-Parse / UT-Logic + class-level tests |
| `py -m pytest -m e2e` | E2E-01..02 + pipeline / CLI |
| `py -m pytest -m regression` | REG-01..02 |
| `py -m pytest -m performance` | PF / LD / STR |
| `py -m pytest -k UT_Parse_04` | a single STD test case |
| `py -m pytest -m "not performance" -n auto` | what CI runs (parallel, needs pytest-xdist) |

Full STD sizes (500^3, 1023^3, 5,000 frames, 500,000 planes): set `FULL_SCALE=1`
(PowerShell: `$env:FULL_SCALE=1`; cmd: `set FULL_SCALE=1`). Needs ~5 GB free RAM.
`PERF_TARGET_HARDWARE=1` also runs PF-02 (>= 100 FPS).

## What runs automatically
* **On every `git commit`** (`.githooks/pre-commit`): smoke + unit + e2e + regression.
  A failing test blocks the commit. Bypass once with `git commit --no-verify`.
* **On every push / pull request** (`.github/workflows/tests.yml`): the same suites on
  Windows, Linux and macOS, with coverage, then the performance/load/stress job.
  Trigger it by hand with a full-scale run from the Actions tab ("Run workflow").

## Test data
`py generate_fixtures.py --out fixtures` writes the NRRD/CSV files named in the STD
(8/16/32-bit volumes at 500^3 and 1023^3, triangle / trapezoid / malformed / reference /
heavy / stress CSVs). The tests build their own smaller copies, so this is optional.

## Updating the REG-01 gold hash
If you change the rendering algorithm on purpose, review the change, then run
`py test_Ex0.py --print-gold` and paste the printed value into `GOLD_REG01_SHA256`.
