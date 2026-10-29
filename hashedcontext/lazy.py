"""
Lazy Tail + line-diff variant of HashedContext.

History is never rewritten, so every prompt is a strict prefix of the next one
(prefix-cache friendly, like append-only). Each file observation is rendered as:

  first sight      -> full body once                      [FILE] ... [/FILE]
  unchanged since  -> stub: "turn X content is still current"   [HASH_STUB]
  last shown
  changed since    -> only the changed lines vs the last   [FILE_DIFF]
  last shown          version the model saw

"Changed lines" is a line-level diff (difflib opcodes, no context lines), not a
syntax-tree diff. Every diff is verified by applying it back to the previous
version and checking the SHA-256 of the result.

Trade-off vs core HashedContext: there is no single authoritative copy at the
tail. The current state of a file is its last full body plus every diff since.
`max_diffs_before_rebase` (default 10) bounds that chain: once a file has had
that many diffs since its last full body, its next change is sent as a full
body again (a "rebase"). None disables rebasing.
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

from hashedcontext.core import sha256_hex

FULL_OPEN = "[FILE]\npath: {path}\nsha256: {h}\n"
FULL_CLOSE = "\n[/FILE]\n"
STUB_LAZY = ("[HASH_STUB]\npath: {path}\nsha256: {h}\n"
             "state: Turn {t} content is still current\n[/HASH_STUB]\n")
DIFF_OPEN = "[FILE_DIFF]\npath: {path}\nbase: turn {t} sha256 {base}\nsha256: {h}\n"
DIFF_CLOSE = "[/FILE_DIFF]\n"
HUNK = "@@ {i},{n} @@\n"  # replace n old lines starting at 1-based line i (n=0: insert before i)

HUNK_RE = re.compile(r"@@ (\d+),(\d+) @@")


def line_diff(old: str, new: str) -> str:
    """Hunks that turn `old` into `new`: '-' old lines, '+' new lines, no context."""
    a, b = old.split("\n"), new.split("\n")
    out: List[str] = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append(HUNK.format(i=i1 + 1, n=i2 - i1))
        out.extend("-" + x + "\n" for x in a[i1:i2])
        out.extend("+" + x + "\n" for x in b[j1:j2])
    return "".join(out)


def apply_line_diff(old: str, hunks: str) -> str:
    """Inverse of line_diff. Raises ValueError if the old lines do not match."""
    a = old.split("\n")
    res: List[str] = []
    pos = 0
    lines = hunks.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        m = HUNK_RE.fullmatch(lines[i])
        if not m:
            raise ValueError(f"bad hunk header: {lines[i][:40]!r}")
        start, n = int(m.group(1)) - 1, int(m.group(2))
        i += 1
        dels, adds = [], []
        while i < len(lines) and lines[i][:1] in ("-", "+"):
            (dels if lines[i][0] == "-" else adds).append(lines[i][1:])
            i += 1
        if a[start:start + n] != dels or start < pos:
            raise ValueError("diff does not apply")
        res.extend(a[pos:start])
        res.extend(adds)
        pos = start + n
    res.extend(a[pos:])
    return "\n".join(res)


DEFAULT_MAX_DIFFS = 10


class LazyContext:
    """Tracks what the model has already seen and renders each new observation."""

    def __init__(self, max_diffs_before_rebase: Optional[int] = DEFAULT_MAX_DIFFS):
        self.max_diffs = max_diffs_before_rebase
        # path -> (content as the model can reconstruct it, sha, turn it was last shown)
        self.seen: Dict[str, Tuple[str, str, int]] = {}
        self.chain: Dict[str, int] = {}  # diffs since last full body, per path
        self.max_chain = 0
        self.n_full = self.n_stub = self.n_diff = self.n_rebase = 0
        self.diff_verified = 0

    def render(self, k: int, path: str, content: str) -> Tuple[str, str]:
        """Return (kind, segment) for observing `content` of `path` at turn k."""
        h = sha256_hex(content)
        prev = self.seen.get(path)
        if prev is None:
            self.seen[path] = (content, h, k)
            self.chain[path] = 0
            self.n_full += 1
            return "full", FULL_OPEN.format(path=path, h=h) + content + FULL_CLOSE
        old, oh, t = prev
        if oh == h:
            self.n_stub += 1
            return "stub", STUB_LAZY.format(path=path, h=h, t=t)
        if self.max_diffs is not None and self.chain[path] >= self.max_diffs:
            # rebase: resend the full body so the model never has to apply a long chain
            self.seen[path] = (content, h, k)
            self.chain[path] = 0
            self.n_rebase += 1
            return "rebase", FULL_OPEN.format(path=path, h=h) + content + FULL_CLOSE
        hunks = line_diff(old, content)
        rebuilt = apply_line_diff(old, hunks)
        if sha256_hex(rebuilt) != h:  # every diff must reproduce the exact new bytes
            raise AssertionError(f"lazy diff not lossless for {path} at turn {k}")
        self.diff_verified += 1
        self.seen[path] = (content, h, k)
        self.chain[path] += 1
        self.max_chain = max(self.max_chain, self.chain[path])
        self.n_diff += 1
        return "diff", DIFF_OPEN.format(path=path, t=t, base=oh[:16], h=h) + hunks + DIFF_CLOSE


# ----------------------------------------------------------------------------
# Reader: what can be reconstructed from the lazy prompt text alone
# ----------------------------------------------------------------------------
BLOCK_RE = re.compile(
    r"\[FILE\]\npath: (?P<fp>[^\n]*)\nsha256: (?P<fh>[0-9a-f]{64})\n(?P<body>.*?)\n\[/FILE\]\n"
    r"|\[FILE_DIFF\]\npath: (?P<dp>[^\n]*)\nbase: turn \d+ sha256 (?P<db>[0-9a-f]{16})\n"
    r"sha256: (?P<dh>[0-9a-f]{64})\n(?P<hunks>(?:(?:@@ \d+,\d+ @@|[-+][^\n]*)\n)*)\[/FILE_DIFF\]\n"
    r"|\[HASH_STUB\]\npath: (?P<sp>[^\n]*)\nsha256: (?P<sh>[0-9a-f]{64})\nstate: Turn \d+ content is still current\n\[/HASH_STUB\]\n",
    re.S,
)


def replay(prompt: str) -> Dict[str, str]:
    """Rebuild every file's latest content from the prompt text, with no vault.
    Checks each block's digest along the way."""
    state: Dict[str, str] = {}
    for m in BLOCK_RE.finditer(prompt):
        if m.group("fp") is not None:
            body = m.group("body")
            if sha256_hex(body) != m.group("fh"):
                raise ValueError("full block digest mismatch")
            state[m.group("fp")] = body
        elif m.group("dp") is not None:
            p = m.group("dp")
            new = apply_line_diff(state[p], m.group("hunks"))
            if sha256_hex(new) != m.group("dh"):
                raise ValueError("diff block digest mismatch")
            state[p] = new
        else:
            if sha256_hex(state[m.group("sp")]) != m.group("sh"):
                raise ValueError("stub digest mismatch")
    return state


def assemble_lazy(system: str, turns, max_diffs_before_rebase: Optional[int] = DEFAULT_MAX_DIFFS) -> str:
    """Full lazy prompt for the last turn (used for sampled verification)."""
    ctx = LazyContext(max_diffs_before_rebase)
    segs = [system]
    for k, t in enumerate(turns, 1):
        segs.append(t.text)
        for f in t.files:
            segs.append(ctx.render(k, f.path, f.content)[1])
    return "".join(segs)
