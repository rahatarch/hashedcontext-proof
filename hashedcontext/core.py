"""
HashedContext reference implementation (paper Section 3, Algorithm 1).

Components
----------
HashGate      SHA-256 content addressing of every file observation.
HashVault     content-addressed store  V[h] -> exact bytes  (in-memory + optional disk).
Stub          [HASH_STUB] reference that replaces historical file bodies.
assemble_*    conventional (append-only) and HashedContext prompt assembly.
expand        inverse transform: HC prompt + vault -> byte-exact conventional prompt.
resolve       R(h): on-demand historical state resolution.

Conversation model (one agent session)
--------------------------------------
turn k = Turn(text=U_k, files=[FileObs(path, content), ...])
    U_k   : action narrative (reasoning, tool calls, diffs, test output) -- never stubbed
    files : file observations made at turn k (read_file tool results)

Conventional prompt at turn k:
    S . U_1 . F_1 . U_2 . F_2 ... U_k . F_k
HashedContext prompt at turn k (Eq. assembly):
    S . U_1 . stub(F_1) ... U_{k-1} . stub(F_{k-1}) . U_k . LIVE(F_k)
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Formats
# ----------------------------------------------------------------------------
FILE_OPEN = '<file path="{path}">\n'
FILE_CLOSE = "\n</file>\n"

# Stub exactly as printed in the paper (Listing 1): full 64-hex SHA-256.
STUB_PAPER = "[HASH_STUB]\npath: {path}\nsha256: {h}\nstate: CACHED\n[/HASH_STUB]\n"
# Compact variant: 16-hex prefix (64-bit) in-context; the vault is still keyed by
# the full digest and prefix collisions are detected & escalated to full digest.
STUB_COMPACT = "[#{h16} {path}]\n"

LIVE_OPEN = "[LIVE_FILE]\npath: {path}\nsha256: {h}\n"
LIVE_CLOSE = "\n[/LIVE_FILE]\n"

STUB_RE_PAPER = re.compile(
    r"\[HASH_STUB\]\npath: (?P<path>[^\n]*)\nsha256: (?P<h>[0-9a-f]{64})\nstate: CACHED\n\[/HASH_STUB\]\n"
)
STUB_RE_COMPACT = re.compile(r"\[#(?P<h>[0-9a-f]{16}) (?P<path>[^\]\n]*)\]\n")
LIVE_RE = re.compile(
    r"\[LIVE_FILE\]\npath: (?P<path>[^\n]*)\nsha256: (?P<h>[0-9a-f]{64})\n(?P<body>.*?)\n\[/LIVE_FILE\]\n",
    re.S,
)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FileObs:
    path: str
    content: str

    @property
    def digest(self) -> str:
        return sha256_hex(self.content)


@dataclass
class Turn:
    text: str  # action narrative U_k (verbatim, never compressed)
    files: List[FileObs] = field(default_factory=list)


# ----------------------------------------------------------------------------
# HashVault: content-addressed, lossless store
# ----------------------------------------------------------------------------
class HashVault:
    """V[h] -> exact bytes. Optional on-disk persistence (git-style 2-char sharding)."""

    def __init__(self, root: Optional[str] = None):
        self._mem: Dict[str, str] = {}
        self._prefix: Dict[str, str] = {}  # 16-hex prefix -> full digest
        self.root = root
        self.puts = 0
        self.dedup_hits = 0
        self.bytes_offered = 0
        if root:
            os.makedirs(root, exist_ok=True)

    def put(self, content: str) -> str:
        h = sha256_hex(content)
        self.puts += 1
        self.bytes_offered += len(content.encode("utf-8"))
        if h in self._mem:
            self.dedup_hits += 1
            return h
        self._mem[h] = content
        p = h[:16]
        if p in self._prefix and self._prefix[p] != h:  # pragma: no cover - 2^-64
            raise RuntimeError(f"64-bit prefix collision {p}; escalate to full digest")
        self._prefix[p] = h
        if self.root:
            d = os.path.join(self.root, h[:2])
            os.makedirs(d, exist_ok=True)
            fp = os.path.join(d, h[2:])
            if not os.path.exists(fp):
                with open(fp, "wb") as f:
                    f.write(content.encode("utf-8"))
        return h

    def resolve(self, h: str) -> Optional[str]:
        """R(h): V[h] if h in dom(V) else None (cache miss)."""
        if len(h) == 16:
            h = self._prefix.get(h, h)
        c = self._mem.get(h)
        if c is None and self.root:
            fp = os.path.join(self.root, h[:2], h[2:])
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    c = f.read().decode("utf-8")
        if c is not None and sha256_hex(c) != h:  # integrity check on every resolve
            raise RuntimeError(f"vault corruption for {h}")
        return c

    def __contains__(self, h: str) -> bool:
        return h in self._mem

    def __len__(self) -> int:
        return len(self._mem)

    @property
    def unique_bytes(self) -> int:
        return sum(len(v.encode("utf-8")) for v in self._mem.values())


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render_file(f: FileObs) -> str:
    return FILE_OPEN.format(path=f.path) + f.content + FILE_CLOSE


def render_stub(path: str, h: str, compact: bool = False) -> str:
    if compact:
        return STUB_COMPACT.format(path=path, h16=h[:16])
    return STUB_PAPER.format(path=path, h=h)


def render_live(f: FileObs, h: Optional[str] = None) -> str:
    h = h or f.digest
    return LIVE_OPEN.format(path=f.path, h=h) + f.content + LIVE_CLOSE


# ----------------------------------------------------------------------------
# Assembly (Algorithm 1)
# ----------------------------------------------------------------------------
def assemble_conventional(system: str, turns: List[Turn]) -> List[str]:
    """Append-only (pass-by-value) prompt as a list of segments."""
    segs = [system]
    for t in turns:
        segs.append(t.text)
        for f in t.files:
            segs.append(render_file(f))
    return segs


def assemble_hashed(
    system: str,
    turns: List[Turn],
    vault: HashVault,
    compact: bool = False,
) -> List[str]:
    """HashedContext prompt:  S . H_stubbed . U_curr . F_authoritative.

    Every historical file block is vaulted and replaced by a stub; the files
    observed in the current turn are materialised once, at the tail, with a
    self-attesting sha256 header.
    """
    segs = [system]
    for t in turns[:-1]:
        segs.append(t.text)
        for f in t.files:
            h = vault.put(f.content)
            segs.append(render_stub(f.path, h, compact))
    last = turns[-1]
    segs.append(last.text)
    for f in last.files:  # P_tail
        h = vault.put(f.content)
        segs.append(render_live(f, h))
    return segs


def expand(hc_prompt: str, vault: HashVault, compact: bool = False) -> str:
    """Inverse transform: resolve every stub via R(h) and turn LIVE blocks back
    into ordinary file blocks. Must reproduce the conventional prompt byte-for-byte."""
    stub_re = STUB_RE_COMPACT if compact else STUB_RE_PAPER

    def _stub(m: re.Match) -> str:
        c = vault.resolve(m.group("h"))
        if c is None:
            raise KeyError(f"vault miss {m.group('h')}")
        return FILE_OPEN.format(path=m.group("path")) + c + FILE_CLOSE

    def _live(m: re.Match) -> str:
        body = m.group("body")
        if sha256_hex(body) != m.group("h"):
            raise ValueError("live block does not self-attest")
        return FILE_OPEN.format(path=m.group("path")) + body + FILE_CLOSE

    out = LIVE_RE.sub(_live, hc_prompt)
    return stub_re.sub(_stub, out)


# ----------------------------------------------------------------------------
# In-context oracle: what a reader can learn from the HC prompt ALONE (no vault)
# ----------------------------------------------------------------------------
def oracle_read(hc_prompt: str, compact: bool = False) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return (live_state, stub_history) extracted purely from prompt text.

    live_state[path]   = exact current bytes of every file in the tail
    stub_history[path] = ordered list of digests observed historically
    """
    live = {m.group("path"): m.group("body") for m in LIVE_RE.finditer(hc_prompt)}
    stub_re = STUB_RE_COMPACT if compact else STUB_RE_PAPER
    hist: Dict[str, List[str]] = {}
    for m in stub_re.finditer(hc_prompt):
        hist.setdefault(m.group("path"), []).append(m.group("h"))
    return live, hist


# ----------------------------------------------------------------------------
# Gateway mode: transform an unmodified OpenAI-style chat request
# ----------------------------------------------------------------------------
def gateway_transform(messages: List[dict], vault: HashVault, compact: bool = False) -> List[dict]:
    """Server-side HashedContext filter for OpenAI chat payloads.

    File observations are `tool` messages answering a `read_file` tool call.
    Every such observation becomes a stub; the observations made after the last
    user/assistant narrative boundary (the current turn) are materialised once
    in a final LIVE message at the tail. The gateway only knows what the client
    sent, so 'live' = latest observed bytes.
    """
    read_ids = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") == "read_file":
                import json as _json

                read_ids[tc["id"]] = _json.loads(fn.get("arguments") or "{}").get("path")
    # current turn = tool results after the last assistant message (agent loop:
    # assistant(tool_calls) -> tool results -> request)
    last_asst = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    out, tail = [], []
    for i, m in enumerate(messages):
        if m["role"] == "tool" and m.get("tool_call_id") in read_ids:
            path = read_ids[m["tool_call_id"]]
            h = vault.put(m["content"])
            if i > last_asst:
                tail.append(FileObs(path, m["content"]))
                out.append({**m, "content": f"[LIVE_REF {h[:16]} {path}]"})
            else:
                out.append({**m, "content": render_stub(path, h, compact)})
        else:
            out.append(m)
    if tail:
        out.append({"role": "user", "content": "".join(render_live(f) for f in tail)})
    return out


LIVE_REF_RE = re.compile(r"^\[LIVE_REF (?P<h>[0-9a-f]{16}) (?P<path>[^\]]*)\]$")


def gateway_expand(messages: List[dict], vault: HashVault, compact: bool = False) -> List[dict]:
    """Inverse of gateway_transform (used to verify every request is lossless)."""
    stub_re = STUB_RE_COMPACT if compact else STUB_RE_PAPER
    msgs = list(messages)
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith("[LIVE_FILE]"):
        msgs = msgs[:-1]
    out = []
    for m in msgs:
        if m["role"] == "tool":
            c = m["content"]
            s = stub_re.fullmatch(c)
            r = LIVE_REF_RE.match(c)
            if s:
                c = vault.resolve(s.group("h"))
            elif r:
                c = vault.resolve(r.group("h"))
            m = {**m, "content": c}
        out.append(m)
    return out
