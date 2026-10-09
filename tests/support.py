"""Shared receipt fixtures."""


def row(**overrides):
    base = {
        "id": "fixture/one",
        "task": "embed",
        "model": {
            "repo_or_tag": "BAAI/bge-base-en-v1.5",
            "revision_or_digest": "unrecorded",
            "quant": "unrecorded",
            "licence": "unrecorded",
        },
        "engine": {"name": "openvino", "version": "2026.4.1"},
        "device": "NPU",
        "settings": {
            "thinking_level": None,
            "num_predict": None,
            "num_ctx": None,
            "batch": 1,
            "static_shape": "1x128",
            "precision": "fp16",
        },
        "quality": [
            {
                "name": "min_cosine_vs_cpu",
                "value": 0.99999,
                "n": 0,
                "interval": {"low": 0.99999, "high": 0.99999, "method": "source-minimum"},
                "dataset_hash": "abc",
            }
        ],
        "latency": {
            "cold_ms": None,
            "warm_p50_ms": None,
            "warm_p95_ms": None,
            "throughput": 20.5,
            "throughput_unit": "items/s",
            "compile_ms": 7440,
        },
        "date": "2026-10-09",
        "repeats": 1,
        "provisional": True,
        "provenance": {
            "source_receipt_path": "E:/AI/Research and Development/experiments/npu-probe/results/2026-10-09-probe.json",
            "commit_sha": "013168a9d1eb0909f99cd9057d4c8970688fa5f0",
        },
    }
    base.update(overrides)
    return base
