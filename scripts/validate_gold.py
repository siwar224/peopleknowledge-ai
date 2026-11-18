"""Check the gold questions against the corpus before using them.

The format is checked by the GoldQuestion model; this script checks what a
model cannot see alone, across files:

- every cited source exists in the chunks (a typo like "4.3" for "4.2" would
  otherwise make a question impossible to pass);
- every ``rule_refs`` path exists in canonical_policy_rules.yaml;
- every ``user_context`` value exists in company_profile.yaml.

It also prints how the questions are spread over categories and documents.

Usage:
    python scripts/validate_gold.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from peopleknowledge.evaluation.gold import GoldQuestion, Source, load_gold, source_of_chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = PROJECT_ROOT / "datasets" / "evaluation" / "rag_questions_gold.jsonl"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "datasets" / "processed" / "chunks"
DEFAULT_RULES = PROJECT_ROOT / "datasets" / "metadata" / "canonical_policy_rules.yaml"
DEFAULT_PROFILE = PROJECT_ROOT / "datasets" / "metadata" / "company_profile.yaml"

# user_context field -> list of company_profile.yaml whose ids it must use
PROFILE_LISTS = {
    "department": "departments",
    "job_family": "job_families",
    "employment_type": "employment_types",
}


def source_index(chunks_dir: Path) -> set[Source]:
    """Every (document_id, section) a question may cite, from the chunk files."""

    sources: set[Source] = set()
    for path in sorted(chunks_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sources.add(source_of_chunk(json.loads(line)))
    return sources


def rule_exists(rules: dict[str, Any], path: str) -> bool:
    node: Any = rules
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def profile_ids(profile: dict[str, Any]) -> dict[str, set[str]]:
    return {
        field: {item["id"] for item in profile.get(list_name) or []}
        for field, list_name in PROFILE_LISTS.items()
    }


def check(
    questions: list[GoldQuestion],
    sources: set[Source],
    rules: dict[str, Any],
    ids: dict[str, set[str]],
) -> list[str]:
    """All cross-file problems, as readable messages (empty list = valid)."""

    errors: list[str] = []
    for question in questions:
        cited = [
            *(("expected", s) for s in question.expected_sources),
            *(("acceptable", s) for s in question.acceptable_sources),
            *(("forbidden", s) for s in question.forbidden_sources),
        ]
        for role, source in cited:
            if source not in sources:
                errors.append(
                    f"{question.id}: {role} source {source.document_id} §{source.section} "
                    "does not exist in the chunks."
                )
        for path in question.rule_refs:
            if not rule_exists(rules, path):
                errors.append(f"{question.id}: rule_refs {path!r} not in canonical_policy_rules.yaml.")
        context = question.user_context.model_dump(exclude_none=True)
        for field, allowed in ids.items():
            value = context.get(field)
            if value is not None and value not in allowed:
                errors.append(f"{question.id}: user_context.{field} {value!r} not in company_profile.yaml.")
    return errors


def print_summary(questions: list[GoldQuestion]) -> None:
    categories = Counter(q.category for q in questions)
    documents = Counter(s.document_id for q in questions for s in q.expected_sources)
    difficulties = Counter(q.difficulty for q in questions)
    print(f"{len(questions)} questions")
    print("  categories:  " + ", ".join(f"{k}={v}" for k, v in sorted(categories.items())))
    print("  documents:   " + ", ".join(f"{k}={v}" for k, v in sorted(documents.items())))
    print("  difficulty:  " + ", ".join(f"{k}={v}" for k, v in sorted(difficulties.items())))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    arguments = parser.parse_args(argv)

    try:
        questions = load_gold(arguments.gold)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    if not any(arguments.chunks_dir.glob("*.jsonl")):
        print(f"ERROR: no chunk in {arguments.chunks_dir} (run scripts/chunk_documents.py first).")
        return 1

    rules = yaml.safe_load(arguments.rules.read_text(encoding="utf-8"))
    profile = yaml.safe_load(arguments.profile.read_text(encoding="utf-8"))
    errors = check(questions, source_index(arguments.chunks_dir), rules, profile_ids(profile))

    print_summary(questions)
    for error in errors:
        print(f"  ERROR {error}")
    print("Gold questions: " + ("valid." if not errors else f"{len(errors)} error(s)."))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
