"""
test_text_holdout_split.py — Unit tests for text holdout partitioning & data preparation CLI
"""

import pytest

from scripts.prepare_text_dataset import (
    parse_args,
    partition_manifest_holdout,
)


@pytest.fixture
def sample_nlp_records():
    """Generates synthetic balanced NLP records across two AI generators and human."""
    records = []
    # 20 human records
    for i in range(20):
        records.append(
            {
                "id": f"human-{i}",
                "dataset": "raid",
                "split": "train",
                "label": 0,
                "generator": "human",
                "domain": "abstracts",
                "attack": "none",
                "text": f"This is human abstract text sequence number {i}.",
            }
        )
    # 10 chatgpt records
    for i in range(10):
        records.append(
            {
                "id": f"chatgpt-{i}",
                "dataset": "raid",
                "split": "train",
                "label": 1,
                "generator": "chatgpt",
                "domain": "abstracts",
                "attack": "none",
                "text": f"This is chatgpt generated abstract text sequence number {i}.",
            }
        )
    # 10 mistral records
    for i in range(10):
        records.append(
            {
                "id": f"mistral-{i}",
                "dataset": "raid",
                "split": "train",
                "label": 1,
                "generator": "mistral",
                "domain": "abstracts",
                "attack": "none",
                "text": f"This is mistral generated abstract text sequence number {i}.",
            }
        )
    return records


def test_holdout_generator_isolation_and_balance(sample_nlp_records):
    main_rows, holdout_rows = partition_manifest_holdout(
        sample_nlp_records,
        holdout_generator="chatgpt",
        seed=42,
    )

    # 1. Holdout test set verification
    assert len(holdout_rows) == 20  # 10 chatgpt + 10 human controls
    ho_ai = [r for r in holdout_rows if r["label"] == 1]
    ho_human = [r for r in holdout_rows if r["label"] == 0]
    assert len(ho_ai) == 10
    assert len(ho_human) == 10
    assert all(r["generator"] == "chatgpt" for r in ho_ai)
    assert all(r["generator"] == "human" for r in ho_human)
    assert all(r["split"] == "test" for r in holdout_rows)

    # 2. Complete absence of held-out generator from main dataset
    assert not any(
        r["generator"].lower() == "chatgpt" for r in main_rows
    ), "Held-out generator 'chatgpt' leaked into main manifest!"

    # 3. In-domain main dataset verification
    assert len(main_rows) == 20  # 10 remaining AI (mistral) + 10 remaining human
    main_human = [r for r in main_rows if r["label"] == 0]
    main_ai = [r for r in main_rows if r["label"] == 1]
    assert len(main_human) == len(main_ai) == 10
    assert all(r["generator"] == "mistral" for r in main_ai)

    # 4. Zero leakage between main dataset and holdout test set
    main_texts = set(r["text"] for r in main_rows)
    ho_texts = set(r["text"] for r in holdout_rows)
    overlap = main_texts.intersection(ho_texts)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} leaking texts between main and holdout!"


def test_holdout_generator_not_found(sample_nlp_records):
    with pytest.raises(ValueError, match="not found in manifest AI records"):
        partition_manifest_holdout(
            sample_nlp_records, holdout_generator="nonexistent_llm"
        )


def test_holdout_domain_single_domain_limitation(sample_nlp_records):
    with pytest.raises(ValueError, match="contains only a single domain"):
        partition_manifest_holdout(sample_nlp_records, holdout_domain="abstracts")


def test_holdout_multi_domain_partitioning():
    records = [
        {
            "id": "1",
            "label": 0,
            "domain": "abstracts",
            "generator": "human",
            "text": "t1",
        },
        {
            "id": "2",
            "label": 1,
            "domain": "abstracts",
            "generator": "gpt2",
            "text": "t2",
        },
        {"id": "3", "label": 0, "domain": "news", "generator": "human", "text": "t3"},
        {"id": "4", "label": 1, "domain": "news", "generator": "gpt2", "text": "t4"},
    ]
    main_rows, holdout_rows = partition_manifest_holdout(records, holdout_domain="news")
    assert len(holdout_rows) == 2
    assert all(r["domain"] == "news" for r in holdout_rows)
    assert all(r["split"] == "test" for r in holdout_rows)
    assert len(main_rows) == 2
    assert all(r["domain"] == "abstracts" for r in main_rows)


def test_default_behavior_preservation(sample_nlp_records):
    main_rows, holdout_rows = partition_manifest_holdout(sample_nlp_records)
    assert main_rows == sample_nlp_records
    assert holdout_rows == []


def test_parse_args_defaults():
    args = parse_args([])
    assert args.holdout_generator is None
    assert args.holdout_domain is None
    assert args.input_manifest is None
    assert args.seed == 42
    assert "text_manifest.csv" in args.output_manifest
    assert "text_holdout_test.csv" in args.holdout_output
