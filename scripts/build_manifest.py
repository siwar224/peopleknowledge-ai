"""Build the manifest of the source files.

The script does not parse the files or extract their text. It combines:

- a technical inventory (path, size, SHA-256 fingerprint, comparison with
  the previous manifest);
- the descriptive entry of each document, written by hand in
  ``datasets/metadata/document_catalog.yaml`` (identity, version, validity,
  language, category).

It writes one JSON object per file to ``datasets/metadata/document_manifest.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets" / "source_documents"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "metadata" / "document_manifest.jsonl"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "datasets" / "metadata" / "document_catalog.yaml"
HASH_CHUNK_SIZE = 1024 * 1024
SCHEMA_VERSION = 2

# Catalog fields copied into the manifest, in this order.
CATALOG_FIELDS = (
    "document_id",
    "family_id",
    "title",
    "doc_type",
    "version",
    "lifecycle_status",
    "effective_from",
    "effective_until",
    "supersedes",
    "superseded_by",
    "language",
    "category",
    "department_scope",
    "owner",
)
# Fields that may be null in the catalog.
OPTIONAL_CATALOG_FIELDS = {"effective_until", "supersedes", "superseded_by"}
LIFECYCLE_STATUSES = {"active", "archived"}
DOC_TYPES = {"handbook", "policy", "procedure", "matrix"}
VERSION_SUFFIX_PATTERN = re.compile(r"(?:[_-](?:v|version)[_-]?)(\d+(?:[._-]\d+)?)$", re.IGNORECASE)
POLICY_SUFFIX_PATTERN = re.compile(r"(?:[_-]policy)$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 of a file without loading its whole content in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modified_at_utc(path: Path) -> str:
    """Return the modification date in ISO-8601 UTC format."""

    modified_timestamp = path.stat().st_mtime
    return datetime.fromtimestamp(modified_timestamp, tz=UTC).isoformat()


def logical_identity(path: Path) -> tuple[str, str]:
    """Derive the logical identity and version from the file name.

    Current convention: ``remote_work_policy_v2.pdf`` becomes
    ``HR-REMOTE-WORK`` and ``HR-REMOTE-WORK-v2``. A file without an explicit
    version suffix is treated as version ``v1``. This provisional identity is
    only kept for files missing from the catalog.
    """

    stem = path.stem
    version_match = VERSION_SUFFIX_PATTERN.search(stem)
    version = version_match.group(1).replace("_", ".").replace("-", ".") if version_match else "1"
    logical_stem = stem[: version_match.start()] if version_match else stem
    logical_stem = POLICY_SUFFIX_PATTERN.sub("", logical_stem)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", logical_stem).strip("-").upper()
    document_id = f"HR-{slug or 'DOCUMENT'}"
    version_id = f"{document_id}-v{version}"
    return document_id, version_id


def discover_documents(source_dir: Path) -> list[Path]:
    """Discover all source files, sorted by relative path.

    Hidden files, including the structural ``.gitkeep``, are ignored.
    The search is recursive so that any subfolders are kept.
    """

    if not source_dir.exists():
        source_dir.mkdir(parents=True)

    documents = (
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(source_dir).parts)
    )
    return sorted(documents, key=lambda path: path.relative_to(source_dir).as_posix().lower())


def build_record(path: Path, source_dir: Path) -> dict[str, object]:
    """Build the technical record of a document."""

    relative_source_path = path.relative_to(PROJECT_ROOT).as_posix()
    relative_path = path.relative_to(source_dir).as_posix()
    file_stat = path.stat()
    file_sha256 = sha256_file(path)
    document_id, version_id = logical_identity(path)

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "version_id": version_id,
        "file_name": path.name,
        "relative_path": relative_path,
        "source_path": relative_source_path,
        "extension": path.suffix.lower(),
        "size_bytes": file_stat.st_size,
        "modified_at_utc": modified_at_utc(path),
        "sha256": file_sha256,
        "change_status": "new",
        "identity_resolution": "generated",
        "is_duplicate": False,
        "duplicate_of": None,
    }


def load_previous_manifest(output_path: Path) -> list[dict[str, object]]:
    """Load the records of the previous manifest, if it exists."""

    if not output_path.exists():
        return []

    previous_records: list[dict[str, object]] = []
    with output_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid previous manifest at line {line_number}: {error}"
                ) from error
            if isinstance(record, dict):
                previous_records.append(record)

    return previous_records


def identity_pair(record: dict[str, object]) -> tuple[str, str] | None:
    """Return the full identity of a previous record, if available."""

    document_id = record.get("document_id")
    version_id = record.get("version_id")
    if not isinstance(document_id, str) or not document_id:
        return None
    if not isinstance(version_id, str) or not version_id:
        return None
    return document_id, version_id


def preserve_identity(record: dict[str, object], previous_record: dict[str, object]) -> bool:
    """Copy the identity of a previous record instead of regenerating it."""

    previous_identity = identity_pair(previous_record)
    if previous_identity is None:
        return False

    record["document_id"], record["version_id"] = previous_identity
    record["identity_resolution"] = "preserved"
    return True


def classify_path_change(previous_path: str, current_path: str) -> str:
    """Tell a file rename apart from a move to another directory."""

    previous_name = previous_path.rsplit("/", maxsplit=1)[-1]
    current_name = current_path.rsplit("/", maxsplit=1)[-1]
    if previous_name == current_name:
        return "moved"
    return "renamed"


def mark_ambiguous_identity(
    record: dict[str, object], candidates: list[dict[str, object]]
) -> None:
    """Prevent an arbitrary resolution when several identities differ."""

    record["document_id"] = None
    record["version_id"] = None
    record["identity_resolution"] = "ambiguous"
    record["identity_candidates"] = [
        {
            "source_path": candidate.get("source_path"),
            "document_id": candidate.get("document_id"),
            "version_id": candidate.get("version_id"),
        }
        for candidate in candidates
    ]


def compare_with_previous(
    records: list[dict[str, object]], previous_records: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Add a comparison status to each current document.

    The ``renamed`` and ``moved`` statuses are used when the content is
    identical but the source path changed. A new path whose previous path
    still exists is classified as ``copied``. Files present in the previous
    manifest but missing from the new one are reported separately as
    ``removed``.
    """

    previous_by_path = {
        str(record["source_path"]): record
        for record in previous_records
        if record.get("source_path")
    }
    previous_by_sha: dict[str, list[dict[str, object]]] = {}
    for record in previous_records:
        sha256 = record.get("sha256")
        if sha256:
            previous_by_sha.setdefault(str(sha256), []).append(record)

    current_paths = {str(record["source_path"]) for record in records}
    for record in records:
        source_path = str(record["source_path"])
        sha256 = str(record["sha256"])
        previous_record = previous_by_path.get(source_path)

        if previous_record is not None:
            previous_sha256 = str(previous_record.get("sha256", ""))
            if not preserve_identity(record, previous_record):
                mark_ambiguous_identity(record, [previous_record])
            if previous_sha256 == sha256:
                record["change_status"] = "unchanged"
            else:
                record["change_status"] = "modified"
                record["previous_sha256"] = previous_sha256
            continue

        same_content_records = previous_by_sha.get(sha256, [])
        if not same_content_records:
            continue

        candidate_identities = [
            identity_pair(previous_record) for previous_record in same_content_records
        ]
        all_candidates_have_identity = all(
            candidate_identity is not None for candidate_identity in candidate_identities
        )
        unique_identities = set(candidate_identities)
        unique_identity = (
            next(iter(unique_identities))
            if all_candidates_have_identity and len(unique_identities) == 1
            else None
        )

        current_candidate_paths = [
            str(previous_record["source_path"])
            for previous_record in same_content_records
            if str(previous_record.get("source_path")) in current_paths
        ]

        if unique_identity is not None:
            record["document_id"], record["version_id"] = unique_identity
            record["identity_resolution"] = "preserved"

        if len(same_content_records) != 1:
            if len(current_candidate_paths) == 1:
                record["change_status"] = "copied"
                record["copied_from"] = current_candidate_paths[0]
            else:
                record["change_status"] = "ambiguous"
                if unique_identity is None:
                    mark_ambiguous_identity(record, same_content_records)
                else:
                    record["identity_resolution"] = "preserved_ambiguous_source"
                    record["identity_candidates"] = [
                        candidate.get("source_path") for candidate in same_content_records
                    ]
            continue

        previous_source_path = str(same_content_records[0]["source_path"])
        if previous_source_path in current_paths:
            record["change_status"] = "copied"
            record["copied_from"] = previous_source_path
        else:
            record["change_status"] = classify_path_change(previous_source_path, source_path)
            record["previous_source_path"] = previous_source_path

        if unique_identity is None:
            mark_ambiguous_identity(record, same_content_records)

    return [
        record
        for record in previous_records
        if record.get("source_path") and str(record["source_path"]) not in current_paths
    ]


def mark_duplicates(records: list[dict[str, object]]) -> int:
    """Flag content duplicates and return their count."""

    first_record_by_sha: dict[str, dict[str, object]] = {}
    duplicate_count = 0

    # A copied file must not become the canonical representative ahead of
    # its original document just because its path sorts first.
    ordered_records = sorted(
        records,
        key=lambda record: (
            record.get("change_status") == "copied",
            str(record.get("source_path", "")),
        ),
    )
    for record in ordered_records:
        sha256 = str(record["sha256"])
        first_record = first_record_by_sha.get(sha256)
        if first_record is None:
            first_record_by_sha[sha256] = record
            continue

        record["is_duplicate"] = True
        record["duplicate_of"] = first_record["source_path"]
        duplicate_count += 1

    return duplicate_count


def iso_date(value: object) -> str | None:
    """Normalize a catalog date (YAML date or YYYY-MM-DD text)."""

    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)).isoformat()


def validate_catalog(entries: list[dict[str, object]]) -> list[str]:
    """Return the list of catalog errors (empty if the catalog is valid)."""

    errors: list[str] = []
    files: Counter[str] = Counter()
    document_ids: Counter[str] = Counter()

    for position, entry in enumerate(entries, start=1):
        label = f"documents[{position}] ({entry.get('file', '?')})"
        if not entry.get("file"):
            errors.append(f"{label}: missing field 'file'.")
        else:
            files[str(entry["file"])] += 1

        for field in CATALOG_FIELDS:
            if field not in entry:
                errors.append(f"{label}: missing field '{field}'.")
            elif entry[field] is None and field not in OPTIONAL_CATALOG_FIELDS:
                errors.append(f"{label}: empty field '{field}'.")

        if entry.get("document_id"):
            document_ids[str(entry["document_id"])] += 1
        if entry.get("lifecycle_status") not in LIFECYCLE_STATUSES:
            errors.append(f"{label}: lifecycle_status must be one of {sorted(LIFECYCLE_STATUSES)}.")
        if entry.get("doc_type") not in DOC_TYPES:
            errors.append(f"{label}: doc_type must be one of {sorted(DOC_TYPES)}.")

        try:
            start = iso_date(entry.get("effective_from"))
            end = iso_date(entry.get("effective_until"))
        except ValueError:
            errors.append(f"{label}: invalid date (expected format YYYY-MM-DD).")
        else:
            if start and end and end < start:
                errors.append(f"{label}: effective_until is earlier than effective_from.")
            if entry.get("lifecycle_status") == "archived":
                if end is None:
                    errors.append(f"{label}: an archived document must have effective_until.")
                if not entry.get("superseded_by"):
                    errors.append(f"{label}: an archived document must have superseded_by.")

    errors += [f"File catalogued more than once: {name}" for name, n in files.items() if n > 1]
    errors += [f"Duplicate document_id: {doc_id}" for doc_id, n in document_ids.items() if n > 1]

    for entry in entries:
        for field in ("supersedes", "superseded_by"):
            target = entry.get(field)
            if target and target not in document_ids:
                errors.append(
                    f"{entry.get('document_id')}: {field} points to {target}, which is not in the catalog."
                )

    return errors


def load_catalog(catalog_path: Path) -> dict[str, dict[str, object]]:
    """Load and validate the catalog, indexed by the file's relative path.

    A missing catalog is not an error: all documents are then reported as
    uncatalogued. An invalid catalog stops the script.
    """

    if not catalog_path.exists():
        return {}

    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        content = yaml.safe_load(catalog_file) or {}

    entries = content.get("documents") or []
    errors = validate_catalog(entries)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Invalid catalog ({catalog_path}):\n{details}")

    return {str(entry["file"]): entry for entry in entries}


def apply_catalog(
    records: list[dict[str, object]], catalog: dict[str, dict[str, object]]
) -> list[str]:
    """Complete each record with its catalog entry.

    The catalog is authoritative for identity: it replaces the identity that
    was generated or preserved from the previous manifest. Return the
    catalogued files that were not found on disk.
    """

    for record in records:
        entry = catalog.get(str(record["relative_path"]))
        if entry is None:
            record["catalog_status"] = "missing"
            record["lifecycle_status"] = None
            continue

        for field in CATALOG_FIELDS:
            record[field] = entry[field]
        record["version"] = str(entry["version"])
        record["effective_from"] = iso_date(entry["effective_from"])
        record["effective_until"] = iso_date(entry["effective_until"])
        record["version_id"] = f"{entry['document_id']}-v{record['version']}"
        record["identity_resolution"] = "catalog"
        record.pop("identity_candidates", None)
        record["catalog_status"] = "matched"

    found_paths = {str(record["relative_path"]) for record in records}
    return sorted(path for path in catalog if path not in found_paths)


def write_manifest(records: list[dict[str, object]], output_path: Path) -> None:
    """Write the JSONL atomically to avoid a partial file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f"{output_path.stem}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                temporary_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                temporary_file.write("\n")

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    """Parse the command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Folder containing the source files (default: {DEFAULT_SOURCE_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help=f"YAML catalog of the documents (default: {DEFAULT_CATALOG_PATH}).",
    )
    return parser.parse_args()


def main() -> int:
    """Discover the source files and generate the manifest."""

    arguments = parse_args()
    source_dir = arguments.source_dir.resolve()
    output_path = arguments.output.resolve()
    try:
        catalog = load_catalog(arguments.catalog.resolve())
    except ValueError as error:
        print(error)
        return 1

    previous_records = load_previous_manifest(output_path)
    documents = discover_documents(source_dir)
    records = [build_record(document, source_dir) for document in documents]
    removed_records = compare_with_previous(records, previous_records)
    absent_files = apply_catalog(records, catalog)
    duplicate_count = mark_duplicates(records)
    write_manifest(records, output_path)

    print(f"{len(records)} document(s) inventoried.")
    uncatalogued = [r["relative_path"] for r in records if r["catalog_status"] == "missing"]
    print(f"Catalog: {len(records) - len(uncatalogued)} entry(ies) applied.")
    for path in uncatalogued:
        print(f"  WARNING: {path} has no entry in the catalog.")
    for path in absent_files:
        print(f"  WARNING: {path} is catalogued but not found on disk.")
    status_counts = Counter(str(record["change_status"]) for record in records)
    status_summary = ", ".join(
        f"{status}={status_counts.get(status, 0)}"
        for status in (
            "new",
            "unchanged",
            "modified",
            "renamed",
            "moved",
            "copied",
            "ambiguous",
        )
    )
    print(f"Comparison: {status_summary}, removed={len(removed_records)}.")
    print(f"Duplicates found: {duplicate_count}.")
    print(f"Manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
