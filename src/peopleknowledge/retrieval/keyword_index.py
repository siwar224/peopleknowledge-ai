"""Keyword search (BM25) over the indexed chunks, and rank fusion with the vector search.

Why: a dense embedding captures meaning, but can miss an exact word that
matters ("SRE", "astreinte", "RW-ENG-008", "J+3"). BM25 is the classic keyword
ranking: a chunk scores high when it contains the rare words of the question.
The two rankings are merged with Reciprocal Rank Fusion (RRF): each list gives
1 / (k + rank) to a chunk, and the sums are sorted. RRF only uses RANKS, so the
scores of the two methods do not need to be comparable.

The index is built in memory from the Qdrant payloads (187 chunks: a few
milliseconds); nothing is stored on disk.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

# Frequent French / English words that carry no meaning for the search.
STOPWORDS = frozenset(
    """
    a au aux avec ce ces cette dans de des du elle en est et il ils je j la le les leur
    lui ma mais me mes mon ne nos notre nous on ou par pas pour qu que qui sa se ses son
    sur ta te tes ton tu un une vos votre vous y d l s c n m t qu est-ce ai as avez ont
    suis es sont etre avoir fait faire peux peut puis combien quel quelle quels quelles
    comment quand est-ce si ou bien plus moins tres the of to and in is for on
    """.split()
)
_WORD = re.compile(r"[a-z0-9]+(?:[+-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, accents removed, stopwords removed: "Congés" and "conges" match."""

    ascii_text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [word for word in _WORD.findall(ascii_text) if word not in STOPWORDS]


class KeywordIndex:
    """Okapi BM25 over chunk texts (k1 = 1.5, b = 0.75, the usual values)."""

    def __init__(self, payloads: Iterable[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.payloads = list(payloads)
        self.k1, self.b = k1, b
        self.documents = [Counter(tokenize(p.get("text", ""))) for p in self.payloads]
        self.lengths = [sum(counts.values()) for counts in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        frequency: Counter[str] = Counter()
        for counts in self.documents:
            frequency.update(counts.keys())
        total = len(self.documents)
        self.idf = {
            word: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for word, count in frequency.items()
        }

    def search(
        self, question: str, limit: int, include_archived: bool = False
    ) -> list[tuple[float, dict[str, Any]]]:
        """(score, payload) of the best chunks; archived documents left out by default."""

        words = tokenize(question)
        scored = []
        for counts, length, payload in zip(
            self.documents, self.lengths, self.payloads, strict=True
        ):
            if not include_archived and payload["document"]["lifecycle_status"] != "active":
                continue
            score = 0.0
            for word in words:
                tf = counts.get(word, 0)
                if tf:
                    norm = 1 - self.b + self.b * length / (self.average_length or 1)
                    score += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            if score > 0:
                scored.append((score, payload))
        scored.sort(key=lambda item: -item[0])
        return scored[:limit]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """id -> sum over the rankings of 1 / (k + rank). k = 60 is the value of the RRF paper."""

    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0.0) + 1 / (k + rank)
    return fused
