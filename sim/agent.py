"""
Autonomous coding agents.

Each agent owns a copy-on-write workspace over a real repository and runs a
closed loop:   observe (read files) -> reason -> edit -> run checks -> repeat.

Edits are real source mutations; diffs are real `difflib` unified diffs; the
"test" step really compiles Python (``compile``) or bracket-checks JS/TS and
its output goes into the next turn's narrative.  Agent behaviour is identical
under both context architectures -- HashedContext only changes what is SENT
to the model, which is exactly what the paper claims (no change in fidelity).

Profiles
--------
paper      : re-reads all M files every turn (the paper's cost model)
realistic  : discovers files lazily, re-reads ~35% of known files per turn,
             always re-reads a file before editing it
runaway    : stuck in an edit -> test-fail -> re-read loop (Section 8)
"""
from __future__ import annotations

import difflib
import os
import random
import re
from typing import Dict, List, Optional

from hashedcontext.core import FileObs, Turn

SYSTEM_PROMPT = (
    "You are an autonomous software engineering agent. You can call tools: "
    "read_file(path), edit_file(path, diff), run_tests(target). Work step by step, "
    "keep edits minimal, verify each change with tests, and explain your reasoning. "
    "File contents are provided inside <file> blocks; when a [HASH_STUB] appears it "
    "refers to an earlier exact version identified by its SHA-256 digest, and the "
    "authoritative current version of every file you just read is in the [LIVE_FILE] "
    "blocks at the end of the prompt.\n"
)

TASKS = [
    "Fix the failing edge case reported in issue #{n}: {fn} returns the wrong value when the input is empty.",
    "Refactor {fn} for readability without changing behaviour; keep the public API stable.",
    "Add input validation to {fn} and make sure the existing tests still pass.",
    "Investigate a flaky test touching {fn} and make it deterministic.",
    "Rename internal helper usages around {fn} and update call sites consistently.",
    "Add debug logging around {fn} to trace issue #{n}, then clean up.",
]
THOUGHTS = [
    "Looking at `{fn}` in {path}, the control flow around line {line} is where the value is computed.",
    "The call site of `{fn}` passes `{ident}` straight through, so a guard belongs before it is used.",
    "I need to confirm how `{ident}` is initialised before changing `{fn}`.",
    "The previous edit compiled, but I want to double-check the branch that handles `{ident}`.",
    "Tests reference `{fn}` indirectly; I'll re-read {path} to make sure the change is consistent.",
    "Next I'll make a minimal change in {path} and re-run the checks.",
    "`{ident}` is shadowed in this scope, which explains the behaviour described in the task.",
    "This matches the hypothesis: the bug is local to `{fn}`; other modules only import it.",
]

IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")
DEF_RE_PY = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)", re.M)
DEF_RE_JS = re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)|^(\s*)(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(", re.M)


def _defs(path: str, src: str):
    out = []
    if path.endswith(".py"):
        for m in DEF_RE_PY.finditer(src):
            out.append((m.start(), m.group(1), m.group(2)))
    else:
        for m in DEF_RE_JS.finditer(src):
            ind = m.group(1) if m.group(1) is not None else (m.group(3) or "")
            out.append((m.start(), ind, m.group(2) or m.group(4)))
    return out


def run_checks(path: str, src: str) -> tuple[bool, str]:
    """Real static check: Python compile() or bracket balance for JS/TS."""
    if path.endswith(".py"):
        try:
            compile(src, path, "exec")
            return True, f"$ python -m py_compile {path}\nOK\n"
        except SyntaxError as e:
            return False, (f"$ python -m py_compile {path}\n  File \"{path}\", line {e.lineno}\n"
                           f"    {(e.text or '').rstrip()}\nSyntaxError: {e.msg}\n")
    stack, pairs = [], {")": "(", "]": "[", "}": "{"}
    for i, ch in enumerate(src):
        if ch in "([{":
            stack.append((ch, i))
        elif ch in pairs:
            if not stack or stack[-1][0] != pairs[ch]:
                line = src.count("\n", 0, i) + 1
                return False, f"$ tsc --noEmit {path}\n{path}({line}): error TS1005: unexpected '{ch}'.\n"
            stack.pop()
    if stack:
        line = src.count("\n", 0, stack[-1][1]) + 1
        return False, f"$ tsc --noEmit {path}\n{path}({line}): error TS1005: '{stack[-1][0]}' was never closed.\n"
    return True, f"$ tsc --noEmit {path}\nFound 0 errors.\n"


class CodingAgent:
    def __init__(self, agent_id: int, repo, profile: str, rng: random.Random,
                 M: int = 5, P: int = 30, p_edit: float = 0.5, min_file_tok: int = 0, tok=None):
        self.id, self.repo, self.profile, self.rng = agent_id, repo, profile, rng
        self.P = P
        self.p_edit = p_edit
        self.ws: Dict[str, str] = {}  # copy-on-write overlay
        cands = repo.candidates
        if min_file_tok and tok is not None:
            cands = [p for p in cands if tok.count(repo.files[p]) >= min_file_tok] or cands
        seed = rng.choice(cands)
        sib = [p for p in repo.by_dir.get(os.path.dirname(seed), []) if p != seed and p in cands]
        rng.shuffle(sib)
        others = [p for p in cands if p != seed and p not in sib]
        rng.shuffle(others)
        self.working_set = ([seed] + sib + others)[:M]
        self.known: List[str] = []
        self.pending_edit: Optional[str] = None
        self.broken: Dict[str, str] = {}   # path -> last good content (runaway)
        self.last_out = ""
        defs = _defs(seed, repo.files[seed])
        fn = defs[rng.randrange(len(defs))][2] if defs else "main"
        self.task = rng.choice(TASKS).format(fn=fn, n=rng.randint(100, 9999))
        self.edits = 0
        self.check_failures = 0
        self.generated_text = ""  # what the model "wrote" this turn (for G_k)

    # ---------------------------------------------------------------- workspace
    def file(self, p: str) -> str:
        return self.ws.get(p, self.repo.files[p])

    def _ident(self, src: str) -> str:
        ids = IDENT_RE.findall(src[: 20000])
        return self.rng.choice(ids) if ids else "value"

    # ---------------------------------------------------------------- editing
    def _mutate(self, path: str) -> str:
        src = self.file(path)
        r = self.rng
        lines = src.split("\n")
        defs = _defs(path, src)
        py = path.endswith(".py")
        if self.profile == "runaway":
            if path in self.broken:  # "fix" attempt: restore + a small different change
                good = self.broken.pop(path)
                gl = good.split("\n")
                i = r.randrange(len(gl))
                ind = re.match(r"\s*", gl[i]).group(0)
                gl.insert(i, f"{ind}{'#' if py else '//'} attempt {self.edits}: adjust handling of {self._ident(good)}")
                return "\n".join(gl)
            # introduce a defect (unbalanced bracket / bad indent) -> checks fail
            self.broken[path] = src
            i = r.randrange(len(lines))
            ind = re.match(r"\s*", lines[i]).group(0)
            bad = f"{ind}if ({self._ident(src)} == null {{" if not py else f"{ind}if {self._ident(src)} is None\n{ind}    return None"
            lines.insert(i, bad)
            return "\n".join(lines)
        op = r.choice(["comment", "guard", "literal", "rename", "comment"])
        if defs and op in ("comment", "guard"):
            pos, ind, name = defs[r.randrange(len(defs))]
            li = src.count("\n", 0, pos)
            if op == "comment":
                lines.insert(li, f"{ind}{'#' if py else '//'} NOTE({self.id}): {name} -- {self.rng.choice(['validated inputs', 'see issue', 'keep behaviour stable', 'hot path'])}")
            else:
                j = li + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    body_ind = re.match(r"\s*", lines[j]).group(0) or ind + "    "
                    v = self._ident(src)
                    guard = (f"{body_ind}if {v} is None:\n{body_ind}    raise ValueError('{v} is required')" if py
                             else f"{body_ind}if ({v} == null) {{ throw new TypeError('{v} is required'); }}")
                    lines.insert(j, guard)
            return "\n".join(lines)
        if op == "literal":
            idx = [i for i, l in enumerate(lines) if re.search(r"\b\d+\b", l)]
            if idx:
                i = r.choice(idx)
                lines[i] = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), lines[i], count=1)
                return "\n".join(lines)
        ident = self._ident(src)
        new = ident + "_" + r.choice(["v2", "safe", "checked", "norm"])
        idx = [i for i, l in enumerate(lines) if re.search(rf"\b{re.escape(ident)}\b", l)]
        for i in idx[:3]:
            lines[i] = re.sub(rf"\b{re.escape(ident)}\b", new, lines[i])
        return "\n".join(lines)

    # ---------------------------------------------------------------- one turn
    def step(self, k: int) -> Turn:
        """Produce turn k: narrative U_k (results of the previous action + new
        reasoning and tool calls) and file observations F_k (current bytes)."""
        r = self.rng
        parts: List[str] = []
        gen: List[str] = []  # model-authored portion
        if k == 1:
            parts.append(f"USER: {self.task}\n")
        # 1) execute the edit decided in the previous response
        if self.pending_edit:
            p = self.pending_edit
            old = self.file(p)
            new = self._mutate(p)
            self.ws[p] = new
            self.edits += 1
            diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                f"a/{p}", f"b/{p}", n=2))
            call = f"ASSISTANT: edit_file(path=\"{p}\")\n```diff\n{diff}```\n"
            gen.append(call)
            parts.append(call)
            ok, out = run_checks(p, new)
            if not ok:
                self.check_failures += 1
            parts.append(f"TOOL run_tests:\n{out}")
            self.pending_edit = None
        # 2) choose what to read now
        ws = self.working_set
        if self.profile == "paper":
            reads = list(ws)
        elif self.profile == "runaway":
            reads = ws[:2]
        else:  # realistic
            if not self.known:
                self.known = ws[: min(2, len(ws))]
            elif len(self.known) < len(ws) and r.random() < 0.4:
                self.known.append(ws[len(self.known)])
            reads = [p for p in self.known if r.random() < 0.35]
        # 3) decide next edit (it will be applied at the start of turn k+1)
        if self.profile == "runaway" or r.random() < self.p_edit:
            target = ws[0] if self.profile == "runaway" else r.choice(self.known or ws)
            self.pending_edit = target
            if target not in reads:
                reads.append(target)  # agents re-read before editing
        # 4) narrative: reasoning + tool calls
        nthought = r.randint(2, 5)
        for _ in range(nthought):
            p = r.choice(reads or ws)
            src = self.file(p)
            defs = _defs(p, src)
            fn = defs[r.randrange(len(defs))][2] if defs else "module"
            t = r.choice(THOUGHTS).format(fn=fn, path=p, line=r.randint(1, src.count("\n") + 1),
                                          ident=self._ident(src))
            gen.append(t + "\n")
            parts.append(f"ASSISTANT: {t}\n")
        for p in reads:
            call = f"ASSISTANT: read_file(path=\"{p}\")\n"
            gen.append(call)
            parts.append(call)
        self.generated_text = "".join(gen)
        files = [FileObs(p, self.file(p)) for p in reads]
        return Turn(text="".join(parts), files=files)
