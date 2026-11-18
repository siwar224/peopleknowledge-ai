"""Tests of the key facts derivation, on a synthetic rules file (not the corpus)."""

import pytest

from peopleknowledge.evaluation.key_facts import (
    RuleRefError,
    derive_key_facts,
    extract_numbers,
    resolve_rule,
)

RULES = {
    "leave": {
        "days": 24,
        "enabled": True,
        "periods": [
            {"duration": "1 à 2 jours", "notice": 3},
            {"duration": "3 à 9 jours", "notice": 12},
        ],
        "sick": "certificat sous 48 h (détails : HR-007, lot 2)",
        "mixed": "acquisition 2 j/mois ; pas de report",
    },
    "office": {"Head Office": "jeudi", "hours": "07:30–20:00"},
    "budget": "1 500 TND",
}


def test_resolve_rule_follows_keys_and_list_indexes() -> None:
    assert resolve_rule(RULES, "leave.days") == 24
    assert resolve_rule(RULES, "leave.periods[1].notice") == 12
    assert resolve_rule(RULES, "office.Head Office") == "jeudi"


@pytest.mark.parametrize(
    "path", ["leave.x", "leave.periods[5]", "leave.days[0]", "leave.periods.0"]
)
def test_resolve_rule_rejects_missing_paths(path: str) -> None:
    with pytest.raises(RuleRefError):
        resolve_rule(RULES, path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sous 48 h", {"48"}),
        ("dans les 48 heures", {"48"}),
        ("1 500 TND", {"1500"}),
        ("1 500 TND", {"1500"}),
        ("0,5 jour", {"0.5"}),
        ("entre 20:00 et 08:00", {"20", "8"}),
        ("à 8h30", {"8:30"}),
        ("voir HR-013, lot 2 et 3, ligne RW-ENG-008", set()),
        ("J+3 puis J+5", {"3", "5"}),
    ],
)
def test_extract_numbers_normalises_rules_and_answers_alike(text: str, expected: set) -> None:
    assert extract_numbers(text) == expected


def test_derive_key_facts_keeps_numbers_only_once_and_skips_booleans() -> None:
    facts = derive_key_facts(["leave.days", "leave.enabled", "leave.sick"], RULES)

    assert [fact.value for fact in facts] == ["24", "48"]
    assert facts[0].rule_ref == "leave.days"


def test_a_list_index_avoids_requiring_every_item() -> None:
    whole = derive_key_facts(["leave.periods"], RULES)
    one = derive_key_facts(["leave.periods[1].notice"], RULES)

    assert {f.value for f in whole} == {"1", "2", "3", "9", "12"}
    assert [f.value for f in one] == ["12"]


def test_not_required_facts_are_left_out() -> None:
    assert derive_key_facts(["leave.mixed"], RULES, not_required=["2"]) == []
