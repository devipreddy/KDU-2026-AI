import json
from pathlib import Path

from app.synthetic.generate_dataset import DEFAULT_SEED, generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generated_snapshot_contains_expected_record_count() -> None:
    records = load_jsonl(SYNTHETIC_DIR / "records.jsonl")
    manifest = json.loads((SYNTHETIC_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert len(records) == 1000
    assert manifest["record_count"] == 1000
    assert manifest["vector_store"] == "chromadb"
    assert manifest["ocr_engine"] == "openrouter:qianfan-ocr-fast"


def test_ground_truth_queries_have_expected_positive_sets() -> None:
    records = load_jsonl(SYNTHETIC_DIR / "records.jsonl")
    ground_truth = json.loads(
        (SYNTHETIC_DIR / "ground_truth_queries.json").read_text(encoding="utf-8")
    )
    records_by_label: dict[str, set[str]] = {}

    for record in records:
        for label in record["ground_truth_labels"]:
            records_by_label.setdefault(label, set()).add(record["record_id"])

    for query in ground_truth:
        expected_ids = set(query["expected_record_ids"])
        assert expected_ids
        assert expected_ids == records_by_label[query["positive_label"]]
        assert query["expected_record_count"] == len(expected_ids)


def test_dataset_generation_is_deterministic() -> None:
    records_one, manifest_one, ground_truth_one = generate_dataset(
        record_count=1000, seed=DEFAULT_SEED
    )
    records_two, manifest_two, ground_truth_two = generate_dataset(
        record_count=1000, seed=DEFAULT_SEED
    )

    assert records_one == records_two
    assert manifest_one == manifest_two
    assert ground_truth_one == ground_truth_two


def test_generated_snapshot_matches_generator() -> None:
    snapshot_records = load_jsonl(SYNTHETIC_DIR / "records.jsonl")
    snapshot_manifest = json.loads((SYNTHETIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    snapshot_ground_truth = json.loads(
        (SYNTHETIC_DIR / "ground_truth_queries.json").read_text(encoding="utf-8")
    )
    generated_records, generated_manifest, generated_ground_truth = generate_dataset(
        record_count=1000,
        seed=DEFAULT_SEED,
    )

    assert snapshot_records == generated_records
    assert snapshot_manifest == generated_manifest
    assert snapshot_ground_truth == generated_ground_truth
