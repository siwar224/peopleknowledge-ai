"""Integration tests on the real batch 1 corpus: the intentional traps must
survive parsing. Skipped when the corpus is not present (it is not in git).

These tests check the corpus, not the design: the parsers must never contain
a rule written for these documents (that is what the synthetic tests check).
"""

import json
import re
from functools import cache

import pytest
from app.rag.parsers.models import ParsedDocument

from scripts.parse_documents import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    load_config,
    parse_record,
)

MANIFEST = PROJECT_ROOT / "datasets" / "metadata" / "document_manifest.jsonl"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="corpus not present")


def manifest_records() -> list[dict]:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@cache
def parsed(version_id: str) -> ParsedDocument:
    """Parse exactly as scripts/parse_documents.py does: options from the real
    parsing_config.yaml and from the catalog's ``parsing`` key (via the manifest)."""

    record = next(r for r in manifest_records() if r["version_id"] == version_id)
    return parse_record(record, load_config(DEFAULT_CONFIG), PROJECT_ROOT)


def all_version_ids() -> list[str]:
    return [r["version_id"] for r in manifest_records()] if MANIFEST.exists() else []


@pytest.mark.parametrize("version_id", all_version_ids())
def test_every_document_parses_cleanly(version_id: str) -> None:
    document = parsed(version_id)

    assert document.elements
    assert document.warnings == []


RUNNING_TEXT = re.compile(
    r"^NovaTech Solutions — HR-\d+ v[\d.]+ +(\| +)?Interne$"  # page header
    r"|^Document fictif — usage pédagogique +(\| +)?Page"  # page footer
)


@pytest.mark.parametrize("version_id", all_version_ids())
def test_no_running_header_or_footer_left(version_id: str) -> None:
    lines = [line for e in parsed(version_id).elements for line in e.text.splitlines()]

    assert not [line for line in lines if RUNNING_TEXT.search(line)]


@pytest.mark.parametrize("version_id", ["HR-001-v3.0", "HR-005-v1.2", "HR-006-v1.1", "HR-011-v2.0", "HR-029-v1.0"])
def test_each_text_document_has_one_disclaimer(version_id: str) -> None:
    assert [e.type for e in parsed(version_id).elements].count("disclaimer") == 1


def test_probation_exception_is_in_its_own_section() -> None:
    # Trap: the Handbook is vague, HR-005 §4.2 says 2 days max during probation.
    section = [e for e in parsed("HR-005-v1.2").elements if e.section_number == "4.2"]

    assert section[0].type == "heading"
    assert any(e.type == "paragraph" and "jusqu'à 2 jours" in e.text for e in section)
    assert any(e.type == "list_item" and "manager" in e.text for e in section)


def test_on_call_and_support_rules_only_exist_in_the_csv() -> None:
    records = {e.element_id: e for e in parsed("HR-013-v1.0").elements}

    sre = [e for e in records.values() if "SRE" in e.data["job_family"]]
    support = [e for e in records.values() if e.data["job_family"] == "Customer Support Agent"]
    assert {e.data["max_days_per_week"] for e in sre} == {"3"}
    assert {e.data["max_days_per_week"] for e in support} == {"1"}
    assert len(records) == 38
    assert records["RW-ENG-001"].section_path == ["Engineering"]


def test_v1_and_v2_contradictions_are_both_extracted() -> None:
    v1 = parsed("HR-029-v1.0").elements
    v2 = parsed("HR-011-v2.0").elements

    monday_rule = next(e for e in v1 if "Le lundi est exclu" in e.text)
    assert monday_rule.type == "paragraph"
    assert monday_rule.section_number == "3.2"
    anchor_table = next(e for e in v2 if e.type == "table" and e.section_number == "3.3")
    assert "Département: Sales | Jour d'ancrage: Lundi" in anchor_table.text


def test_table_split_over_two_pages_is_joined() -> None:
    history = [e for e in parsed("HR-005-v1.2").elements if e.type == "table" and e.section_number == "12"]

    assert len(history) == 1
    assert [row[0] for row in history[0].data["rows"]] == ["1.0", "1.1", "1.2"]


@pytest.mark.parametrize("version_id", all_version_ids())
def test_parsing_is_deterministic(version_id: str) -> None:
    first = parsed(version_id).to_json()
    parsed.cache_clear()

    assert parsed(version_id).to_json() == first
