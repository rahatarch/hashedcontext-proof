"""Real open-source repositories used as the agents' working trees."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List

ROOT = os.environ.get("HC_CORPUS", os.path.join(os.path.dirname(__file__), "..", "corpus"))

# repo -> (pinned commit, source globs)
REPOS = {
    "flask": ("https://github.com/pallets/flask", "d086db856be187255b8ec61ef409357393020f32",
              ["src/**/*.py", "tests/**/*.py"]),
    "express": ("https://github.com/expressjs/express", "1c6f1b0fe4000977b9a7b4f34097e4d527dd9824",
                ["lib/**/*.js", "test/**/*.js", "examples/**/*.js"]),
    "zod": ("https://github.com/colinhacks/zod", "b1f6ba06c5de0e600640917615e5df70fa73f060",
            ["packages/zod/src/**/*.ts"]),
    "axios": ("https://github.com/axios/axios", "9d11464a75ee5c644c0308d742f58755c0482594",
              ["lib/**/*.js", "test/**/*.js"]),
}


@dataclass
class Repo:
    name: str
    files: Dict[str, str]          # path -> content (base snapshot)
    candidates: List[str]          # paths eligible for a working set
    by_dir: Dict[str, List[str]]


def load_repos(tok, min_tok=400, max_tok=12000) -> List[Repo]:
    repos = []
    for name, (_, _, globs) in REPOS.items():
        base = os.path.join(ROOT, name)
        if not os.path.isdir(base):
            raise SystemExit(f"corpus missing: {base}. Run: python -m sim.fetch_corpus")
        files = {}
        for g in globs:
            for fp in glob.glob(os.path.join(base, g), recursive=True):
                if fp.endswith(".d.ts") or not os.path.isfile(fp):
                    continue
                rel = f"{name}/" + os.path.relpath(fp, base)
                try:
                    files[rel] = open(fp, encoding="utf-8").read()
                except UnicodeDecodeError:
                    continue
        cands = [p for p, c in files.items() if min_tok <= tok.count(c) <= max_tok]
        by_dir: Dict[str, List[str]] = {}
        for p in cands:
            by_dir.setdefault(os.path.dirname(p), []).append(p)
        repos.append(Repo(name, files, sorted(cands), by_dir))
    return repos
