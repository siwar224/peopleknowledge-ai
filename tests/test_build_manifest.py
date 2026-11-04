from scripts.build_manifest import compare_with_previous, mark_duplicates


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
        "status": "new",
        "identity_resolution": "generated",
        "is_duplicate": False,
        "duplicate_of": None,
    }


def test_unchanged_preserves_identity() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [make_record("datasets/source/a.pdf", "hash-a", "HR-WRONG", "HR-WRONG-v9")]

    removed = compare_with_previous(current, previous)

    assert removed == []
    assert current[0]["status"] == "unchanged"
    assert current[0]["document_id"] == "HR-A"
    assert current[0]["version_id"] == "HR-A-v1"


def test_modified_preserves_identity_and_records_previous_hash() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [make_record("datasets/source/a.pdf", "hash-b", "HR-WRONG", "HR-WRONG-v9")]

    compare_with_previous(current, previous)

    assert current[0]["status"] == "modified"
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

    assert current[0]["status"] == "renamed"
    assert current[0]["document_id"] == "HR-ANNUAL-LEAVE"
    assert current[0]["version_id"] == "HR-ANNUAL-LEAVE-v1"
    assert current[0]["previous_source_path"] == "datasets/source/annual_leave_policy.pdf"


def test_moved_file_preserves_identity_and_is_classified_as_moved() -> None:
    previous = [make_record("datasets/source/a.pdf", "hash-a", "HR-A", "HR-A-v1")]
    current = [make_record("datasets/archive/a.pdf", "hash-a", "HR-WRONG", "HR-WRONG-v1")]

    compare_with_previous(current, previous)

    assert current[0]["status"] == "moved"
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

    assert current[1]["status"] == "copied"
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
    assert current[0]["status"] == "new"
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

    assert current[0]["status"] == "ambiguous"
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

    assert second_scan[0]["status"] == "unchanged"
    assert second_scan[0]["document_id"] == "HR-A"
    assert second_scan[0]["version_id"] == "HR-A-v1"
    assert "previous_source_path" not in second_scan[0]
