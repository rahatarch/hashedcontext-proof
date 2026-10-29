"""Unit/property tests for the HashedContext core.   python -m pytest -q tests"""
import json
import random
import string

from hashedcontext.core import (FileObs, HashVault, Turn, assemble_conventional, assemble_hashed,
                                expand, gateway_expand, gateway_transform, oracle_read, sha256_hex)


def rand_text(r, n):
    alphabet = string.ascii_letters + string.digits + " \n\t{}()[];:'\"#/\\-_=+<>"
    return "".join(r.choice(alphabet) for _ in range(n))


def random_session(seed, P=12, M=4):
    r = random.Random(seed)
    files = {f"src/m{i}.py": rand_text(r, r.randint(50, 3000)) for i in range(M)}
    turns = []
    for _ in range(P):
        if r.random() < 0.5:  # edit
            p = r.choice(list(files))
            s = files[p]
            i = r.randrange(len(s) + 1)
            files[p] = s[:i] + rand_text(r, r.randint(1, 40)) + s[i:]
        reads = [p for p in files if r.random() < 0.6]
        turns.append(Turn(text=rand_text(r, r.randint(10, 400)) + "\n",
                          files=[FileObs(p, files[p]) for p in reads]))
    return turns, files


def test_lossless_roundtrip_property():
    for seed in range(300):
        turns, files = random_session(seed)
        for compact in (False, True):
            v = HashVault()
            conv = "".join(assemble_conventional("SYS\n", turns))
            hc = "".join(assemble_hashed("SYS\n", turns, v, compact=compact))
            assert expand(hc, v, compact=compact) == conv
            live, _ = oracle_read(hc, compact=compact)
            assert live == {f.path: f.content for f in turns[-1].files}
            assert len(hc) <= len(conv) or len(turns) == 1


def test_vault_dedup_and_integrity(tmp_path):
    v = HashVault(str(tmp_path))
    h1 = v.put("hello")
    h2 = v.put("hello")
    assert h1 == h2 == sha256_hex("hello") and len(v) == 1 and v.dedup_hits == 1
    assert v.resolve(h1) == "hello" and v.resolve(h1[:16]) == "hello"
    v2 = HashVault(str(tmp_path))  # reload from disk only
    assert v2.resolve(h1) == "hello"
    assert v.resolve("0" * 64) is None


def test_change_detection_from_prompt_alone():
    """The model-side O(1) check: is the live file identical to a historical stub?"""
    turns, _ = random_session(7, P=10, M=3)
    v = HashVault()
    hc = "".join(assemble_hashed("SYS\n", turns, v))
    live, hist = oracle_read(hc)
    for p, body in live.items():
        truth = [sha256_hex(f.content) for t in turns[:-1] for f in t.files if f.path == p]
        assert hist.get(p, []) == truth
        for h in hist.get(p, []):
            assert (h == sha256_hex(body)) == (v.resolve(h) == body)


def test_gateway_openai_roundtrip():
    for seed in range(50):
        turns, _ = random_session(seed, P=8)
        msgs = [{"role": "system", "content": "SYS"}]
        for k, t in enumerate(turns):
            calls = [{"id": f"c{k}_{j}", "type": "function",
                      "function": {"name": "read_file", "arguments": json.dumps({"path": f.path})}}
                     for j, f in enumerate(t.files)]
            msgs.append({"role": "assistant", "content": t.text, **({"tool_calls": calls} if calls else {})})
            for c, f in zip(calls, t.files):
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": f.content})
        v = HashVault()
        out = gateway_transform(msgs, v)
        assert gateway_expand(out, v) == msgs
        assert len(json.dumps(out)) <= len(json.dumps(msgs)) + 2000


def test_lazy_tail_roundtrip_and_prefix_stability():
    """Lazy Tail + line diff: every file rebuilds byte-exact from the prompt text
    alone, and each turn's prompt is a strict prefix of the next one."""
    from hashedcontext.lazy import apply_line_diff, assemble_lazy, line_diff, replay
    for seed in range(300):
        turns, files = random_session(seed)
        prev = None
        for k in range(1, len(turns) + 1):
            p = assemble_lazy("SYS\n", turns[:k])
            if prev is not None:
                assert p.startswith(prev)
            st = replay(p)
            for f in turns[k - 1].files:
                assert st[f.path] == f.content
            prev = p
    r = random.Random(1)
    for _ in range(500):
        a = rand_text(r, r.randint(0, 800))
        b = rand_text(r, r.randint(0, 800)) if r.random() < 0.3 else a[: r.randint(0, len(a))] + rand_text(r, 5)
        assert apply_line_diff(a, line_diff(a, b)) == b
