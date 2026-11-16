from datetime import date

import pytest

from scripts.build_manifest import (
    apply_catalog,
    compare_with_previous,
    load_catalog,
    mark_duplicates,
    validate_catalog,
)


def make_record(
    source_path: str,
    sha256: str,
    document_id: str = "HR-EXAMPLE",
    version_id: str = "HR-EXAMPLE-v1",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "document_id": document_id,
        "version_id": version_id,
        "file_name": source_path.rsplit("/", maxsplit=1)[-1],
        "source_path": source_path,
        "sha256": sha256,
        "change_status": "new",
        "identity_resolution": "generated",
        "is_duplicate": False,
        "duplicate_of": None,
    }


def test_unchanged_preserves_identity() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [make_record("datasets/source/a.pdf", "hash-a", "HR-WRONG", "HR-WRONG-v9")]

    removed = compare_with_previous(current, previous)

    assert removed == []
    assert current[0]["change_status"] == "unchanged"
    assert current[0]["document_id"] == "HR-A"
    assert current[0]["version_id"] == "HR-A-v1"


def test_modified_preserves_identity_and_records_previous_hash() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [make_record("datasets/source/a.pdf", "hash-b", "HR-WRONG", "HR-WRONG-v9")]

    compare_with_previous(current, previous)

    assert current[0]["change_status"] == "modified"
    assert current[0]["document_id"] == "HR-A"
    assert current[0]["version_id"] == "HR-A-v1"
    assert current[0]["previous_sha256"] == "hash-a"


def test_renamed_file_preserves_identity_and_previous_path() -> None:
    previous = [
        make_record(
            "datasets/source/annual_leave_policy.pdf",
            "hash-a",
            "HR-ANNUAL-LEAVE",
            "HR-ANNUAL-LEAVE-v1",
        )
    ]
    current = [
        make_record(
            "datasets/source/annual_leave_policy_siwar.pdf",
            "hash-a",
            "HR-ANNUAL-LEAVE-POLICY-SIWAR",
            "HR-ANNUAL-LEAVE-POLICY-SIWAR-v1",
        )
    ]

    compare_with_previous(current, previous)

    assert current[0]["change_status"] == "renamed"
    assert current[0]["document_id"] == "HR-ANNUAL-LEAVE"
    assert current[0]["version_id"] == "HR-ANNUAL-LEAVE-v1"
    assert current[0]["previous_source_path"] == "datasets/source/annual_leave_policy.pdf"


def test_moved_file_preserves_identity_and_is_classified_as_moved() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [make_record("datasets/archive/a.pdf", "hash-a", "HR-WRONG", "HR-WRONG-v1")]

    compare_with_previous(current, previous)

    assert current[0]["change_status"] == "moved"
    assert current[0]["document_id"] == "HR-A"
    assert current[0]["version_id"] == "HR-A-v1"
    assert current[0]["previous_source_path"] == "datasets/source/a.pdf"


def test_copy_keeps_original_identity_without_being_called_a_rename() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [
        make_record("datasets/source/a.pdf", "hash-a", "HR-WRONG", "HR-WRONG-v9"),
        make_record("datasets/copies/a-copy.pdf", "hash-a", "HR-COPY", "HR-COPY-v1"),
    ]

    compare_with_previous(current, previous)

    assert current[1]["change_status"] == "copied"
    assert current[1]["document_id"] == "HR-A"
    assert current[1]["version_id"] == "HR-A-v1"
    assert current[1]["copied_from"] == "datasets/source/a.pdf"


def test_duplicate_files_are_marked_without_changing_their_new_status() -> None:
    current = [
        make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1"),
        make_record("datasets/source/a-copy.pdf", "hash-a", "HR-A-COPY", "HR-A-COPY-v1"),
    ]

    duplicate_count = mark_duplicates(current)
    records_by_path = {record["source_path"]: record for record in current}

    assert duplicate_count == 1
    assert sum(record["is_duplicate"] is True for record in current) == 1
    duplicate_record = next(record for record in current if record["is_duplicate"] is True)
    canonical_path = next(
        path for path, record in records_by_path.items() if record["is_duplicate"] is False
    )
    assert duplicate_record["duplicate_of"] == canonical_path


def test_new_file_keeps_generated_identity() -> None:
    current = [
        make_record(
            "datasets/source/remote_work_policy_v2.pdf",
            "hash-new",
            "HR-REMOTE-WORK",
            "HR-REMOTE-WORK-v2",
        )
    ]

    removed = compare_with_previous(current, [])

    assert removed == []
    assert current[0]["change_status"] == "new"
    assert current[0]["document_id"] == "HR-REMOTE-WORK"
    assert current[0]["version_id"] == "HR-REMOTE-WORK-v2"
    assert current[0]["identity_resolution"] == "generated"


def test_ambiguous_hash_match_does_not_guess_identity() -> None:
    previous = [
        make_record("datasets/source/a.pdf", "hash-same", "HR-A", "HR-A-v1"),
        make_record("datasets/source/b.pdf", "hash-same", "HR-B", "HR-B-v1"),
    ]
    current = [make_record("datasets/source/new-name.pdf", "hash-same")]

    compare_with_previous(current, previous)

    assert current[0]["change_status"] == "ambiguous"
    assert current[0]["document_id"] is None
    assert current[0]["version_id"] is None
    assert current[0]["identity_resolution"] == "ambiguous"
    assert len(current[0]["identity_candidates"]) == 2


def test_repeated_scan_is_idempotent() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    first_scan = [make_record("datasets/source/a-renamed.pdf", "hash-a", "HR-WRONG", "HR-WRONG-v9")]

    compare_with_previous(first_scan, previous)
    second_scan = [make_record("datasets/source/a-renamed.pdf", "hash-a", "HR-OTHER", "HR-OTHER-v4")]
    compare_with_previous(second_scan, first_scan)

    assert second_scan[0]["change_status"] == "unchanged"
    assert second_scan[0]["document_id"] == "HR-A"
    assert second_scan[0]["version_id"] == "HR-A-v1"
    assert "previous_source_path" not in second_scan[0]


def make_catalog_entry(file: str, document_id: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "file": file,
        "document_id": document_id,
        "family_id": "remote_work_policy",
        "title": "Politique de télétravail",
        "doc_type": "policy",
        "version": "2.0",
        "lifecycle_status": "active",
        "effective_from": date(2026, 9, 1),
        "effective_until": None,
        "supersedes": None,
        "superseded_by": None,
        "language": "fr",
        "category": "remote_work",
        "department_scope": "all",
        "owner": "people_finance",
    }
    entry.update(overrides)
    return entry


def make_catalog_pair() -> list[dict[str, object]]:
    return [
        make_catalog_entry("02/remote_work_policy_v2.pdf", "HR-011", supersedes="HR-029"),
        make_catalog_entry(
            "06/remote_work_policy_v1.pdf",
            "HR-029",
            version="1.0",
            lifecycle_status="archived",
            effective_from=date(2025, 1, 1),
            effective_until=date(2026, 8, 31),
            superseded_by="HR-011",
        ),
    ]


def test_valid_catalog_has_no_errors() -> None:
    assert validate_catalog(make_catalog_pair()) == []


def test_catalog_rejects_duplicate_ids_and_dangling_references() -> None:
    entries = make_catalog_pair()
    entries[1]["document_id"] = "HR-011"

    errors = validate_catalog(entries)

    assert any("Duplicate document_id: HR-011" in error for error in errors)
    assert any("HR-029, which is not in the catalog" in error for error in errors)


def test_archived_document_needs_end_date_and_successor() -> None:
    entries = [make_catalog_entry("a.pdf", "HR-001", lifecycle_status="archived")]

    errors = validate_catalog(entries)

    assert any("effective_until" in error for error in errors)
    assert any("superseded_by" in error for error in errors)


def test_catalog_rejects_unknown_status_and_inverted_dates() -> None:
    entries = [
        make_catalog_entry(
            "a.pdf",
            "HR-001",
            lifecycle_status="obsolete",
            effective_until=date(2026, 1, 1),
        )
    ]

    errors = validate_catalog(entries)

    assert any("lifecycle_status" in error for error in errors)
    assert any("earlier than" in error for error in errors)


def test_invalid_catalog_file_raises(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text("documents:\n  - file: a.pdf\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid catalog"):
        load_catalog(catalog_path)


def test_catalog_overrides_generated_identity() -> None:
    record = make_record("datasets/source_documents/02/remote_work_policy_v2.pdf", "hash-a",
                         "HR-REMOTE-WORK", "HR-REMOTE-WORK-v2")
    record["relative_path"] = "02/remote_work_policy_v2.pdf"
    catalog = {entry["file"]: entry for entry in make_catalog_pair()}

    absent = apply_catalog([record], catalog)

    assert record["document_id"] == "HR-011"
    assert record["version_id"] == "HR-011-v2.0"
    assert record["family_id"] == "remote_work_policy"
    assert record["lifecycle_status"] == "active"
    assert record["effective_from"] == "2026-09-01"
    assert record["effective_until"] is None
    assert record["identity_resolution"] == "catalog"
    assert record["catalog_status"] == "matched"
    assert absent == ["06/remote_work_policy_v1.pdf"]


def test_uncatalogued_file_is_flagged_and_keeps_its_identity() -> None:
    record = make_record("datasets/source_documents/new.pdf", "hash-n", "HR-NEW", "HR-NEW-v1")
    record["relative_path"] = "new.pdf"

    apply_catalog([record], {})

    assert record["catalog_status"] == "missing"
    assert record["lifecycle_status"] is None
    assert record["document_id"] == "HR-NEW"


def test_parsing_options_are_copied_to_the_manifest() -> None:
    record = make_record("datasets/source_documents/02/matrix.csv", "hash-m")
    record["relative_path"] = "02/matrix.csv"
    entry = make_catalog_entry("02/matrix.csv", "HR-013", parsing={"id_column": "row_id"})

    apply_catalog([record], {"02/matrix.csv": entry})

    assert record["parsing"] == {"id_column": "row_id"}


def test_parsing_options_default_to_empty() -> None:
    record = make_record("datasets/source_documents/02/remote_work_policy_v2.pdf", "hash-a")
    record["relative_path"] = "02/remote_work_policy_v2.pdf"

    apply_catalog([record], {entry["file"]: entry for entry in make_catalog_pair()})

    assert record["parsing"] == {}


def test_catalog_rejects_parsing_that_is_not_a_mapping() -> None:
    entries = [make_catalog_entry("a.csv", "HR-001", parsing=["row_id"])]

    assert any("parsing must be a mapping" in error for error in validate_catalog(entries))
