#!/usr/bin/env python3
"""Enable the repository's git hooks (run once per clone):  python install_hooks.py

Points git at the versioned .githooks/ folder, so tests run automatically on every commit.
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
hook = root / ".githooks" / "pre-commit"
if not hook.is_file():
    sys.exit("Missing .githooks/pre-commit")

try:
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=root, check=True)
except (OSError, subprocess.CalledProcessError) as exc:
    sys.exit(f"Could not configure git (is this a git repository? run `git init` first): {exc}")

if os.name != "nt":                                       # make the hook executable on Linux / macOS
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(["git", "update-index", "--add", "--chmod=+x", ".githooks/pre-commit"], cwd=root, check=False)

print("Git hooks enabled: tests will now run before every commit.")
print("Make sure the requirements are installed:  python -m pip install -r requirements-dev.txt")
