#!/usr/bin/env python3
"""Run Kairos inference with local compatibility shims preloaded."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    third_party_root = repo_root / "kairos" / "third_party"

    os.chdir(repo_root)

    for path in (repo_root, third_party_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    injected = f"{repo_root}:{third_party_root}"
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = f"{injected}:{existing_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = injected

    import kairos_ext._apex_shim  # noqa: F401

    inference_py = repo_root / "examples" / "inference.py"
    sys.argv[0] = str(inference_py)
    runpy.run_path(str(inference_py), run_name="__main__")


if __name__ == "__main__":
    main()
