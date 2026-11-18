"""Tests of the parsing runner, on a tiny corpus built in a temporary folder."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.parse_documents import load_config, main, run

CSV = "row_id,team,days\nR-1,Ops,2\nR-2,Sales,1\n"


class Corpus:
    """A temporary project: source files, manifest, config and output folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: list[dict] = []
        self.manifest = root / "manifest.jsonl"
        self.config = root / "parsing_config.yaml"
        self.output = root / "parsed"

    def add(self, relative_path: str, content: str, document_id: str, **overrides) -> dict:
        path = self.root / "source" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        record = {
            "relative_path": relative_path,
            "source_path": f"source/{relative_path}",
            "extension": Path(relative_path).suffix,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "catalog_status": "matched",
            "is_duplicate": False,
            "duplicate_of": None,
            "document_id": document_id,
            "version_id": f"{document_id}-v1.0",
            "family_id": "test",
            "title": "Test",
            "doc_type": "matrix",
            "version": "1.0",
            "lifecycle_status": "active",
            "effective_from": "2026-01-01",
            "effective_until": None,
            "supersedes": None,
            "superseded_by": None,
            "language": "fr",
            "category": "test",
            "department_scope": "all",
            "owner": "test",
            "parsing": {},
        }
        record.update(overrides)
        self.records.append(record)
        return record

    def run(self, **options):
        lines = [json.dumps(record, ensure_ascii=False) for record in self.records]
        self.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return run(self.manifest, self.config, self.output, root=self.root, **options)

    def status(self, report) -> dict[str, str]:
        return {outcome.version_id: outcome.status for outcome in report.outcomes}

    def output_json(self, version_id: str) -> dict:
        return json.loads((self.output / f"{version_id}.json").read_text(encoding="utf-8"))


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    return Corpus(tmp_path)


def test_first_run_writes_then_second_run_changes_nothing(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100")

    assert corpus.status(corpus.run()) == {"HR-100-v1.0": "new"}
    written = (corpus.output / "HR-100-v1.0.json").stat().st_mtime_ns
    assert corpus.status(corpus.run()) == {"HR-100-v1.0": "unchanged"}
    assert (corpus.output / "HR-100-v1.0.json").stat().st_mtime_ns == written


def test_metadata_change_updates_the_output(corpus: Corpus) -> None:
    record = corpus.add("a.csv", CSV, "HR-100")
    corpus.run()

    record["lifecycle_status"] = "archived"
    report = corpus.run()

    assert corpus.status(report) == {"HR-100-v1.0": "updated"}
    assert corpus.output_json("HR-100-v1.0")["document"]["lifecycle_status"] == "archived"


def test_document_options_from_the_catalog_are_applied(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100", parsing={"id_column": "row_id", "group_by": "team"})

    corpus.run()
    elements = corpus.output_json("HR-100-v1.0")["elements"]

    assert [e["element_id"] for e in elements] == ["R-1", "R-2"]
    assert elements[0]["section_path"] == ["Ops"]


def test_corpus_options_from_the_config_are_applied(corpus: Corpus) -> None:
    corpus.config.write_text('disclaimer_patterns:\n  - "^avertissement"\n', encoding="utf-8")
    docx = pytest.importorskip("docx")
    path = corpus.root / "source" / "b.docx"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = docx.Document()
    document.add_paragraph("Avertissement : exemple.")
    document.add_paragraph("Contenu.")
    document.save(str(path))
    corpus.add("a.csv", CSV, "HR-100")  # creates the record skeleton...
    corpus.records[-1].update(  # ...then points it to the DOCX file
        relative_path="b.docx",
        source_path="source/b.docx",
        extension=".docx",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    corpus.run()

    types = [e["type"] for e in corpus.output_json("HR-100-v1.0")["elements"]]
    assert types == ["disclaimer", "paragraph"]


def test_one_failure_does_not_stop_the_others(corpus: Corpus) -> None:
    corpus.add("bad.csv", CSV, "HR-100", parsing={"no_such_option": True})
    corpus.add("good.csv", CSV, "HR-101")

    report = corpus.run()

    assert corpus.status(report) == {"HR-100-v1.0": "failed", "HR-101-v1.0": "new"}
    assert "invalid parsing options" in report.outcomes[0].detail
    assert report.failed


def test_uncatalogued_and_unsupported_files_fail(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100", catalog_status="missing")
    corpus.add("notes.txt", "texte", "HR-101")

    report = corpus.run()

    details = {outcome.version_id: outcome.detail for outcome in report.outcomes}
    assert "no catalog entry" in details["HR-100-v1.0"]
    assert "no parser for '.txt'" in details["HR-101-v1.0"]


def test_duplicates_are_skipped(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100")
    corpus.add("copy.csv", CSV, "HR-100", is_duplicate=True, duplicate_of="a.csv")

    report = corpus.run()

    assert [o.status for o in report.outcomes] == ["new", "skipped"]
    assert report.outcomes[1].detail == "duplicate of a.csv"


def test_file_changed_after_the_manifest_fails(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100")
    (corpus.root / "source" / "a.csv").write_text(CSV + "R-3,Ops,0\n", encoding="utf-8")

    report = corpus.run()

    assert report.outcomes[0].status == "failed"
    assert "rerun build_manifest.py" in report.outcomes[0].detail


def test_only_by_document_id(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100")
    corpus.add("b.csv", CSV, "HR-101")

    assert corpus.status(corpus.run(only=["HR-101"])) == {"HR-101-v1.0": "new"}
    assert corpus.status(corpus.run(only=["HR-999"])) == {"HR-999": "failed"}


def test_obsolete_outputs_are_reported_then_pruned(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100")
    corpus.output.mkdir()
    old = corpus.output / "HR-OLD-v1.0.json"
    old.write_text("{}", encoding="utf-8")

    assert corpus.run().obsolete == [old]
    assert old.exists()
    corpus.run(prune=True)
    assert not old.exists()


def test_obsolete_check_ignores_only(corpus: Corpus) -> None:
    corpus.add("a.csv", CSV, "HR-100")
    corpus.add("b.csv", CSV, "HR-101")
    corpus.run()

    # A run limited to HR-100 must not call HR-101's output obsolete.
    assert corpus.run(only=["HR-100"]).obsolete == []


def test_config_rejects_unknown_settings_and_bad_regex(tmp_path: Path) -> None:
    config = tmp_path / "parsing_config.yaml"
    config.write_text("disclaimer_pattern: []\n", encoding="utf-8")  # typo
    with pytest.raises(ValueError, match="unknown setting"):
        load_config(config)

    config.write_text('disclaimer_patterns:\n  - "(unclosed"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid regex"):
        load_config(config)


def test_missing_config_means_no_option(tmp_path: Path) -> None:
    assert load_config(tmp_path / "absent.yaml") == {}


def test_exit_code(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    corpus.add("a.csv", CSV, "HR-100")
    corpus.run()  # writes the manifest
    arguments = [
        "--manifest", str(corpus.manifest),
        "--config", str(corpus.config),
        "--output-dir", str(corpus.output),
        "--project-root", str(corpus.root),
    ]

    assert main(arguments) == 0
    assert "unchanged=1" in capsys.readouterr().out
    assert main([*arguments, "--only", "HR-999"]) == 1
    assert "failed" in capsys.readouterr().out
    assert main([*arguments[:2], "--manifest", str(corpus.root / "absent.jsonl")]) == 1
    assert "manifest not found" in capsys.readouterr().out
