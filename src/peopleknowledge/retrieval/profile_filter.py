"""Use the employee's profile to stop table rows about OTHER people from crowding the results.

The eligibility matrix has 38 rows that look alike ("department: X | job_family:
Y | ..."). For "I am an SRE in Engineering", the rows of Sales or Support jobs
are about someone else, yet the vector search ranks many of them high and
pushes the policy sections down (baseline: expected section at rank 6-8).

Two generic rules, applied only to ``record`` chunks (table rows):

1. a row that names ANOTHER value of a profile field the user gave (another
   department, another job) and not the user's own value is moved to the end;
2. at most ``max_records`` rows stay among the first results (the others move
   to the end): a table row should not take every place of the context.

Nothing here knows the corpus: the values come from company_profile.yaml
(label + aliases of every department, job family and contract type).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

# user_context field -> list of company_profile.yaml that holds its values
PROFILE_LISTS = {
    "department": "departments",
    "job_family": "job_families",
    "employment_type": "employment_types",
}


class Scored(Protocol):
    score: float
    payload: dict[str, Any]


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


@dataclass(frozen=True)
class ProfileVocabulary:
    """field -> value id -> the words that name it (label + aliases, normalised)."""

    terms: dict[str, dict[str, tuple[str, ...]]]

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> ProfileVocabulary:
        terms: dict[str, dict[str, tuple[str, ...]]] = {}
        for field, list_name in PROFILE_LISTS.items():
            terms[field] = {
                item["id"]: tuple(
                    sorted(
                        {
                            _normalise(t)
                            for t in [item.get("label", ""), *item.get("aliases", [])]
                            if t
                        }
                    )
                )
                for item in profile.get(list_name) or []
            }
        return cls(terms)

    @classmethod
    def from_file(cls, path: Path) -> ProfileVocabulary:
        return cls.from_profile(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _mentions(text: str, words: Sequence[str]) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) for word in words)


def conflicts_with_profile(
    payload: dict[str, Any], user_context: dict[str, Any], vocabulary: ProfileVocabulary
) -> bool:
    """A table row about someone else: it names another value of a known field, not the user's."""

    if payload.get("kind") != "record":
        return False
    text = _normalise(payload.get("content", ""))
    for field, value in user_context.items():
        values = vocabulary.terms.get(field)
        if not values or value not in values:
            continue
        own = values[value]
        others = [word for other, words in values.items() if other != value for word in words]
        if _mentions(text, others) and not _mentions(text, own):
            return True
    return False


def apply_profile(
    hits: Sequence[Scored],
    user_context: dict[str, Any] | None,
    vocabulary: ProfileVocabulary | None,
    max_records: int | None = 3,
) -> list[Scored]:
    """Same hits, reordered: rows about other people, then extra rows, go to the end."""

    known = {k: v for k, v in (user_context or {}).items() if k in PROFILE_LISTS and v}
    first: list[Scored] = []
    demoted: list[Scored] = []
    records = 0
    for hit in hits:
        if vocabulary and known and conflicts_with_profile(hit.payload, known, vocabulary):
            demoted.append(hit)
            continue
        if hit.payload.get("kind") == "record":
            if max_records is not None and records >= max_records:
                demoted.append(hit)
                continue
            records += 1
        first.append(hit)
    return first + demoted
