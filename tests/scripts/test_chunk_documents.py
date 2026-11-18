"""Tests of the chunking runner, on parsed files written to a temporary folder.

The token counter is fake (one token per word): no tokenizer is downloaded.
"""

import json
from pathlib import Path

import pytest

from peopleknowledge.ingestion.chunking.models import Chunk
from peopleknowledge.ingestion.parsers.models import ParsedDocument
from scripts.chunk_documents import main, run


def words(text: str) -> int:
    return len(text.split())


def parsed_document(document_id: str, paragraph: str = "Texte de la règle.") -> ParsedDocument:
    version_id = f"{document_id}-v1.0"
    return ParsedDocument(
        document={
            "document_id": document_id,
            "version_id": version_id,
            "family_id": "test",
            "title": "Politique de test",
            "doc_type": "policy",
            "version": "1.0",
            "lifecycle_status": "active",
            "effective_from": "2026-01-01",
            "language": "fr",
            "category": "test",
            "department_scope": "all",
            "owner": "hr",
        },
        source={"relative_path": "t.pdf", "extension": ".pdf", "sha256": "c" * 64},
        parsing={"parser": "pdf", "parser_version": "1"},
        elements=[
            {"element_id": f"{version_id}#0000", "index": 0, "type": "heading",
             "text": "1. Objet", "level": 1, "section_number": "1"},
            {"element_id": f"{version_id}#0001", "index": 1, "type": "paragraph",
             "text": paragraph, "section_path": ["1. Objet"], "section_number": "1"},
        ],
    )


class Folders:
    def __init__(self, root: Path) -> None:
        self.parsed = root / "parsed"
        self.chunks = root / "chunks"
        self.parsed.mkdir()

    def add(self, document: ParsedDocument) -> None:
        path = self.parsed / f"{document.document.version_id}.json"
        path.write_text(document.to_json(), encoding="utf-8")

    def run(self, **options):
        options.setdefault("max_tokens", 512)
        return run(self.parsed, self.chunks, words, "fake-words", **options)

    def status(self, report) -> dict[str, str]:
        return {outcome.version_id: outcome.status for outcome in report.outcomes}


@pytest.fixture
def folders(tmp_path: Path) -> Folders:
    return Folders(tmp_path)


def test_writes_one_jsonl_per_document_with_valid_chunks(folders: Folders) -> None:
    folders.add(parsed_document("HR-100"))

    assert folders.status(folders.run()) == {"HR-100-v1.0": "new"}
    lines = (folders.chunks / "HR-100-v1.0.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [Chunk.model_validate(json.loads(line)) for line in lines]
    assert [c.chunk_id for c in chunks] == ["HR-100-v1.0#1"]
    assert chunks[0].chunking.tokenizer == "fake-words"


def test_second_run_changes_nothing(folders: Folders) -> None:
    folders.add(parsed_document("HR-100"))
    folders.run()
    written = (folders.chunks / "HR-100-v1.0.jsonl").stat().st_mtime_ns

    assert folders.status(folders.run()) == {"HR-100-v1.0": "unchanged"}
    assert (folders.chunks / "HR-100-v1.0.jsonl").stat().st_mtime_ns == written


def test_new_settings_update_the_output(folders: Folders) -> None:
    folders.add(parsed_document("HR-100"))
    folders.run(max_tokens=512)

    assert folders.status(folders.run(max_tokens=300)) == {"HR-100-v1.0": "updated"}


def test_a_failure_does_not_stop_the_others(folders: Folders) -> None:
    folders.add(parsed_document("HR-100", paragraph="x " * 50))  # cannot fit in 10 tokens...
    folders.add(parsed_document("HR-101"))
    (folders.parsed / "broken.json").write_text("{}", encoding="utf-8")

    report = folders.run(max_tokens=10)

    status = folders.status(report)
    assert status["broken"] == "failed"
    assert status["HR-101-v1.0"] == "new"
    assert report.failed


def test_only_by_document_id(folders: Folders) -> None:
    folders.add(parsed_document("HR-100"))
    folders.add(parsed_document("HR-101"))

    assert folders.status(folders.run(only=["HR-101"])) == {"HR-101-v1.0": "new"}
    assert folders.status(folders.run(only=["HR-999"])) == {"HR-999": "failed"}


def test_obsolete_outputs_are_reported_then_pruned(folders: Folders) -> None:
    folders.add(parsed_document("HR-100"))
    folders.chunks.mkdir()
    old = folders.chunks / "HR-OLD-v1.0.jsonl"
    old.write_text("", encoding="utf-8")

    assert folders.run().obsolete == [old]
    assert old.exists()
    folders.run(prune=True)
    assert not old.exists()


def test_exit_codes(folders: Folders, capsys: pytest.CaptureFixture[str]) -> None:
    folders.add(parsed_document("HR-100"))
    arguments = ["--parsed-dir", str(folders.parsed), "--output-dir", str(folders.chunks)]

    assert main(arguments, count_tokens=words) == 0
    assert "Chunking: new=1" in capsys.readouterr().out
    assert main([*arguments, "--max-tokens", "0"], count_tokens=words) == 1
    empty = folders.parsed.parent / "empty"
    empty.mkdir()
    assert main(["--parsed-dir", str(empty)], count_tokens=words) == 1
    assert "no parsed file" in capsys.readouterr().out
