"""Key facts of a gold question, derived from its ``rule_refs``.

A generated answer is free text: comparing it word for word with the gold
answer does not work ("sous 48 h" and "dans les 48 heures" say the same thing).
In an HR corpus, what makes an answer right or wrong is mostly its FIGURES
(2 days, 48 hours, 40 TND...). So each question's key facts are the numbers of
the rules it points to in ``canonical_policy_rules.yaml``, and an answer is
checked by looking for those numbers in it.

Three details make the derivation reliable:

- a rule path can address one list item: ``annual_leave.notice_periods[2]``
  (otherwise a question about one notice period would require all four);
- document references inside a rule ("voir HR-013", "lot 2") are not facts;
- numbers are normalised the same way in rules and answers: "1 500" = "1500",
  "08:00" = "8", "0,5" = "0.5".

Besides numbers, a rule often names WHO decides or WHERE to act ("accord du
manager ET validation RH (hr_admin)", "dans NovaHR", "le lundi"). These TERM
facts come from a vocabulary built from data, not code: the roles and systems
of company_profile.yaml (label + aliases) and the days of the week. A term fact
is present when the answer uses ANY of its names ("RH", "équipe RH", "People").

``scripts/validate_gold.py`` checks that every gold answer contains all of its
key facts: a too broad ``rule_refs`` is then caught at once.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# One path part: a mapping key, optionally followed by list indexes ("periods[2]").
_PART = re.compile(r"^(?P<key>[^\[\]]+)(?P<indexes>(?:\[\d+\])*)$")
_INDEX = re.compile(r"\[(\d+)\]")

# References to other documents or batches: identifiers, not facts.
_REFERENCES = re.compile(r"\b(?:HR-\d+|RW-[A-Z]+-\d+|lots?\s*\d+(?:\s*(?:et|,)\s*\d+)*)\b", re.I)
# A clock time with minutes ("17:00", "8h30"): one token, so "17:00" is not "17" and "00".
# "48 h" is a duration, not a time: it stays a plain number.
_TIME = re.compile(r"\b(\d{1,2})(?::|h)(\d{2})(?!\d)", re.I)
# A number with thousand separators ("1 500"), or a plain / decimal number ("0,5", "48").
_NUMBER = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?![\d,.])|\d+(?:[.,]\d+)?")


class RuleRefError(ValueError):
    """A ``rule_refs`` path does not exist in the rules file."""


@dataclass(frozen=True)
class KeyFact:
    value: str  # normalised number ("48", "1500", "0.5") or a term's display name ("Manager")
    rule_ref: str  # the rule it comes from
    kind: str = "number"  # "number" or "term"
    names: tuple[str, ...] = ()  # term: every accepted name, normalised
    entity: str = ""  # term: its id ("hr_admin"), the value to put in not_required_facts


WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def normalise_text(text: str) -> str:
    """Lowercase without accents, "_" as a space: "Équipe RH" -> "equipe rh"."""

    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return plain.replace("_", " ")


def mentions(text: str, names: tuple[str, ...]) -> bool:
    """Whether the (normalised) text contains one of the names as whole words."""

    return any(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text) for name in names)


@dataclass(frozen=True)
class TermVocabulary:
    """entity id -> (display name, accepted names). Roles, systems, weekdays."""

    entities: dict[str, tuple[str, tuple[str, ...]]]

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> TermVocabulary:
        entities: dict[str, tuple[str, tuple[str, ...]]] = {}
        for list_name in ("roles", "systems"):
            for item in profile.get(list_name) or []:
                names = {item["id"], item.get("label") or "", *(item.get("aliases") or [])}
                entities[item["id"]] = (
                    item.get("label") or item["id"],
                    tuple(sorted({normalise_text(n) for n in names if n}, key=len, reverse=True)),
                )
        for day in WEEKDAYS:
            entities.setdefault(day, (day, (day,)))
        return cls(entities)

    def find(self, text: str) -> list[str]:
        """Entity ids named in a text (by id, label or alias)."""

        plain = normalise_text(_REFERENCES.sub(" ", text))  # "HR-013" is not the HR team
        return [entity for entity, (_, names) in self.entities.items() if mentions(plain, names)]


def resolve_rule(rules: dict[str, Any], path: str) -> Any:
    """The value at ``path`` ("a.b", "a.list[2].c"); RuleRefError if it does not exist."""

    node: Any = rules
    for part in path.split("."):
        match = _PART.match(part)
        if not match or not isinstance(node, dict) or match["key"] not in node:
            raise RuleRefError(f"rule {path!r} does not exist (at {part!r}).")
        node = node[match["key"]]
        for index in _INDEX.findall(match["indexes"] or ""):
            if not isinstance(node, list) or int(index) >= len(node):
                raise RuleRefError(f"rule {path!r}: no item [{index}] in {part!r}.")
            node = node[int(index)]
    return node


def _normalise(token: str) -> str:
    token = re.sub(r"[   ]", "", token).replace(",", ".")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return str(int(token)) if token.isdigit() else token


def extract_numbers(text: str) -> set[str]:
    """Every number of a text, normalised (references to documents left out)."""

    text = _REFERENCES.sub(" ", text)
    numbers: set[str] = set()

    def time_token(match: re.Match[str]) -> str:
        hours, minutes = int(match[1]), match[2]
        numbers.add(str(hours) if minutes == "00" else f"{hours}:{minutes}")
        return " "

    text = _TIME.sub(time_token, text)
    numbers.update(_normalise(token) for token in _NUMBER.findall(text))
    return numbers


def _leaves(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def numbers_of_rule(value: Any) -> set[str]:
    """The numbers held by a rule value (a scalar, a string, a list or a mapping)."""

    numbers: set[str] = set()
    for leaf in _leaves(value):
        if isinstance(leaf, bool) or leaf is None:
            continue  # true / false are not figures
        if isinstance(leaf, int | float):
            numbers.add(_normalise(str(leaf)))
        elif isinstance(leaf, str):
            numbers.update(extract_numbers(leaf))
    return numbers


def derive_key_facts(
    rule_refs: list[str],
    rules: dict[str, Any],
    not_required: list[str] | None = None,
    vocabulary: TermVocabulary | None = None,
) -> list[KeyFact]:
    """The key facts of a question: the numbers (and, with a vocabulary, the
    roles / systems / days) of its rules, each listed once.

    ``not_required``: numbers or term ids of a rule that are not part of THIS
    answer (a rule string can hold several facts, e.g. "2 j/mois ; pas de report").
    """

    skipped = {_normalise(value) for value in not_required or []}
    numbers: dict[str, KeyFact] = {}
    terms: dict[str, KeyFact] = {}
    for path in rule_refs:
        value = resolve_rule(rules, path)
        for number in sorted(numbers_of_rule(value)):
            if number not in skipped:
                numbers.setdefault(number, KeyFact(value=number, rule_ref=path))
        if vocabulary is None:
            continue
        texts = [leaf for leaf in _leaves(value) if isinstance(leaf, str)]
        for entity in vocabulary.find(" ; ".join(texts)):
            if entity not in skipped:
                label, names = vocabulary.entities[entity]
                terms.setdefault(
                    entity,
                    KeyFact(value=label, rule_ref=path, kind="term", names=names, entity=entity),
                )
    ordered = sorted(numbers.values(), key=lambda fact: (len(fact.value), fact.value))
    return ordered + sorted(terms.values(), key=lambda fact: fact.value.lower())


def fact_present(fact: KeyFact, text: str) -> bool:
    """Whether an answer (or a gold answer) states this fact."""

    if fact.kind == "term":
        return mentions(normalise_text(text), fact.names)
    return fact.value in extract_numbers(text)
