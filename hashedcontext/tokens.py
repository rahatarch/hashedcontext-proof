"""Real BPE token counting (tiktoken cl100k_base / o200k_base), loaded offline."""
from __future__ import annotations

import os
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(_HERE, "..", "tokenizer_cache"))

import tiktoken  # noqa: E402


class Tokenizer:
    """Token counter with a content-addressed cache (each unique string is
    tokenised exactly once -- the same trick HashedContext applies to prompts)."""

    def __init__(self, name: str = "cl100k_base", cache_size: int = 400_000):
        self.enc = tiktoken.get_encoding(name)
        self.name = name
        self._len: "OrderedDict[str,int]" = OrderedDict()
        self._ids: "OrderedDict[str,list]" = OrderedDict()
        self.cache_size = cache_size
        self.encoded_bytes = 0

    def count(self, s: str) -> int:
        n = self._len.get(s)
        if n is None:
            n = len(self.enc.encode_ordinary(s))
            self.encoded_bytes += len(s)
            self._len[s] = n
            if len(self._len) > self.cache_size:
                self._len.popitem(last=False)
        return n

    def ids(self, s: str) -> list:
        v = self._ids.get(s)
        if v is None:
            v = self.enc.encode_ordinary(s)
            self._ids[s] = v
            self._len[s] = len(v)
            if len(self._ids) > 20_000:
                self._ids.popitem(last=False)
        return v

    def count_many(self, segs) -> int:
        return sum(self.count(s) for s in segs)
