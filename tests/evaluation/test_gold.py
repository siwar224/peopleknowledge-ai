"""Gold question format (GoldQuestion) and cross-file checks (scripts/validate_gold.py)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from peopleknowledge.evaluation.gold import GoldQuestion, Source, load_gold
from scripts.validate_gold import check, rule_exists, source_index


def question(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "Q001",
        "question": "Puis-je poser des congés pendant ma période d'essai ?",
        "user_context": {"employment_type": "cdi", "in_probation": True},
        "expected_sources": [{"document_id": "HR-005", "section": "4.2"}],
        "acceptable_sources": [{"document_id": "HR-001", "section": "6.1"}],
        "forbidden_sources": [],
        "expected_answer": "Oui, 2 jours maximum.",
        "rule_refs": ["annual_leave.probation.exception"],
        "category": "exception_elsewhere",
        "difficulty": "medium",
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "1"}, "pattern"),
        ({"category": "easy_one"}, "category"),
        ({"expected_sources": []}, "needs at least one expected source"),
        ({"category": "unanswerable"}, "has no expected source"),
        ({"category": "multi_source"}, "at least two expected sources"),
        ({"forbidden_sources": [{"document_id": "HR-001", "section": "6.1"}]}, "both allowed and forbidden"),
        ({"user_context": {"role": "manager"}}, "role"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_invalid_question_is_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        GoldQuestion.model_validate(question(**overrides))


def test_unanswerable_question_without_source_is_valid() -> None:
    gold = GoldQuestion.model_validate(
        question(category="unanswerable", expected_sources=[], rule_refs=[])
    )

    assert gold.expected_sources == []


def test_load_gold_names_the_bad_line_and_refuses_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(question()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_gold(path)

    path.write_text(json.dumps(question()) + "\n" + json.dumps(question()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question id"):
        load_gold(path)


def write_chunks(folder: Path) -> Path:
    """Three chunk lines: a section in two parts, a card, a CSV row (only the fields used)."""

    folder.mkdir()
    rows = [
        {"chunk_id": "HR-005-v1.2#4.2", "kind": "section", "section_number": "4.2", "part": 1},
        {"chunk_id": "HR-005-v1.2#4.2-2", "kind": "section", "section_number": "4.2", "part": 2},
        {"chunk_id": "HR-005-v1.2#card", "kind": "document_card", "section_number": None, "part": 1},
    ]
    lines = [json.dumps({**row, "document": {"document_id": "HR-005"}}) for row in rows]
    (folder / "HR-005-v1.2.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = {"chunk_id": "HR-013-v1.0#RW-ENG-008", "kind": "record", "section_number": None,
              "part": 1, "document": {"document_id": "HR-013"}}
    (folder / "HR-013-v1.0.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return folder


def test_source_index_knows_sections_cards_and_rows(tmp_path: Path) -> None:
    sources = source_index(write_chunks(tmp_path / "chunks"))

    assert sources == {
        Source(document_id="HR-005", section="4.2"),
        Source(document_id="HR-005", section="card"),
        Source(document_id="HR-013", section="RW-ENG-008"),
    }


def test_check_reports_missing_sources_rules_and_profile_values(tmp_path: Path) -> None:
    sources = source_index(write_chunks(tmp_path / "chunks"))
    rules = {"annual_leave": {"probation": {"exception": "..."}}}
    ids = {"department": {"sales"}, "job_family": set(), "employment_type": {"cdi"}}
    good = GoldQuestion.model_validate(question(acceptable_sources=[]))
    bad = GoldQuestion.model_validate(
        question(
            id="Q002",
            expected_sources=[{"document_id": "HR-005", "section": "4.3"}],  # typo
            acceptable_sources=[],
            rule_refs=["annual_leave.probation.typo"],
            user_context={"department": "marketing"},
        )
    )

    assert check([good], sources, rules, ids) == []
    errors = check([bad], sources, rules, ids)
    assert len(errors) == 3
    assert "HR-005 §4.3 does not exist" in errors[0]
    assert "annual_leave.probation.typo" in errors[1]
    assert "'marketing' not in company_profile.yaml" in errors[2]


def test_rule_exists_follows_dotted_paths() -> None:
    rules = {"a": {"b": {"c": 1}}, "list": [1, 2]}

    assert rule_exists(rules, "a.b.c")
    assert not rule_exists(rules, "a.x")
    assert not rule_exists(rules, "list.0")  # lists are not addressed by index


def test_rule_exists_addresses_list_items_by_index() -> None:
    rules = {"list": [{"x": 1}, {"x": 2}]}

    assert rule_exists(rules, "list[1].x")
    assert not rule_exists(rules, "list[2]")


def test_check_reports_a_gold_answer_without_its_key_facts(tmp_path: Path) -> None:
    sources = source_index(write_chunks(tmp_path / "chunks"))
    rules = {"leave": {"periods": [{"notice": 3}, {"notice": 20}]}}
    ids = {"department": set(), "job_family": set(), "employment_type": {"cdi"}}
    broad = GoldQuestion.model_validate(
        question(acceptable_sources=[], rule_refs=["leave.periods"], expected_answer="20 jours.")
    )
    narrow = broad.model_copy(update={"rule_refs": ["leave.periods[1].notice"]})

    errors = check([broad], sources, rules, ids)
    assert len(errors) == 1 and "lacks key fact(s) ['3']" in errors[0]
    assert check([narrow], sources, rules, ids) == []
