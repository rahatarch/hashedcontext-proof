"""Clone the pinned corpus repositories:  python -m sim.fetch_corpus"""
import os
import subprocess

from .corpus import REPOS, ROOT

os.makedirs(ROOT, exist_ok=True)
for name, (url, sha, _) in REPOS.items():
    d = os.path.join(ROOT, name)
    if os.path.isdir(d):
        print("ok  ", name)
        continue
    subprocess.check_call(["git", "init", "-q", d])
    subprocess.check_call(["git", "-C", d, "fetch", "-q", "--depth", "1", url, sha])
    subprocess.check_call(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"])
    print("got ", name, sha)
