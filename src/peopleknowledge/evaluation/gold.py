"""Gold questions: the measuring stick of the RAG.

Each question says which SOURCES must be found (a section number, a CSV row
id or ``card``), never which chunk: the chunking can change without breaking
the gold set. The file is ``datasets/evaluation/rag_questions_gold.jsonl``,
one question per line.

- ``expected_sources``: needed to answer; ALL of them should be retrieved.
- ``acceptable_sources``: correct but secondary (e.g. the Handbook summary).
- ``forbidden_sources``: must not be used to answer (e.g. the archived v1
  for a question about the current rule).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal[
    "direct",  # a clear rule in one section
    "exception_elsewhere",  # general rule in one place, exception in another
    "csv_only",  # the answer is only in the eligibility matrix
    "version_conflict",  # current question; the archived version says otherwise
    "history",  # explicit question about a former rule
    "vocabulary",  # informal wording, far from the document's words
    "multi_source",  # the answer needs several sections
    "unanswerable",  # not in the corpus: the assistant must say it does not know
]
Difficulty = Literal["easy", "medium", "hard"]


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)  # "HR-005"
    section: str = Field(min_length=1)  # "4.2", "RW-ENG-008" or "card"


def source_of_chunk(chunk: dict) -> Source:
    """The source a chunk belongs to, from its JSON form (chunk file line or Qdrant payload).

    A section chunk -> its section number; a card or CSV row -> its id suffix
    ("card", "RW-ENG-008"); the "-2" of a second part is dropped.
    """

    document_id = chunk["document"]["document_id"]
    if chunk["kind"] == "section" and chunk.get("section_number"):
        return Source(document_id=document_id, section=chunk["section_number"])
    section = chunk["chunk_id"].split("#", 1)[1]
    if chunk.get("part", 1) > 1:
        section = section.rsplit("-", 1)[0]
    return Source(document_id=document_id, section=section)


class UserContext(BaseModel):
    """Who asks: some answers depend on it. Values are company_profile.yaml ids."""

    model_config = ConfigDict(extra="forbid")

    department: str | None = None
    job_family: str | None = None
    employment_type: str | None = None
    in_probation: bool | None = None
    seniority_years: int | None = Field(default=None, ge=0)


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^Q\d{3}$")
    question: str = Field(min_length=5)
    user_context: UserContext = Field(default_factory=UserContext)
    expected_sources: list[Source]
    acceptable_sources: list[Source] = Field(default_factory=list)
    forbidden_sources: list[Source] = Field(default_factory=list)
    expected_answer: str = Field(min_length=1)
    # paths in canonical_policy_rules.yaml ("a.b", "a.list[2]"); their numbers are the key facts
    rule_refs: list[str] = Field(default_factory=list)
    # numbers of those rules that are NOT part of this answer (see evaluation/key_facts.py)
    not_required_facts: list[str] = Field(default_factory=list)
    category: Category
    difficulty: Difficulty

    @model_validator(mode="after")
    def check_sources(self) -> GoldQuestion:
        if self.category == "unanswerable" and self.expected_sources:
            raise ValueError("an unanswerable question has no expected source.")
        if self.category != "unanswerable" and not self.expected_sources:
            raise ValueError("an answerable question needs at least one expected source.")
        if self.category == "multi_source" and len(self.expected_sources) < 2:
            raise ValueError("a multi_source question needs at least two expected sources.")
        overlap = set(self.forbidden_sources) & {*self.expected_sources, *self.acceptable_sources}
        if overlap:
            names = sorted(f"{s.document_id} §{s.section}" for s in overlap)
            raise ValueError(f"a source cannot be both allowed and forbidden: {names}")
        return self


def load_gold(path: Path) -> list[GoldQuestion]:
    """Read and validate the gold file; raise ValueError naming the line on any problem."""

    questions: list[GoldQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append(GoldQuestion.model_validate(json.loads(line)))
        except ValueError as error:  # JSONDecodeError and ValidationError are ValueErrors
            raise ValueError(f"{path.name} line {number}: {error}") from error
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate question id(s) {duplicates}.")
    return questions
