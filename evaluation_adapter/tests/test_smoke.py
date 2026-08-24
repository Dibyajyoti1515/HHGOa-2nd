"""
evaluation_adapter/tests/test_smoke.py

Smoke tests for the evaluation adapter, independent of rag-local-eval-loop
itself -- these test the adapter's interface compliance directly (per the
integration brief's Step 11), before running the real evaluator.

Requires the real project dependencies installed (sentence-transformers,
torch, groq, python-dotenv, ...) plus a downloaded bge-small-en-v1.5 model
and a valid GROQ_API_KEY for the generator tests. Run from the repo root:

    python -m pytest evaluation_adapter/tests/test_smoke.py -v

or without pytest:

    python evaluation_adapter/tests/test_smoke.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_adapter import embedder, generator  # noqa: E402


@dataclass
class _Context:
    text: str
    source: str
    score: float = 0.0


def test_get_model():
    model = embedder.get_model()
    assert model is not None


def test_embed_one():
    vec = embedder.embed_one("What is the capital of France?")
    assert hasattr(vec, "shape")
    assert vec.shape[-1] > 0
    reshaped = vec.reshape(1, -1)
    assert reshaped.shape == (1, vec.shape[-1])


def test_embed_many():
    vecs = embedder.embed(["first passage", "second passage", "third passage"])
    assert vecs.shape[0] == 3
    assert vecs.shape[1] == embedder.embed_one("probe").shape[-1]


def test_embed_empty():
    vecs = embedder.embed([])
    assert vecs.shape[0] == 0


def test_generate_answer_high_confidence():
    results = [_Context(text="Paris is the capital of France.", source="test", score=0.99)]
    answer = generator.generate_answer("What is the capital of France?", results)
    assert isinstance(answer.text, str) and answer.text
    assert answer.grounded is True
    assert isinstance(answer.generation_ms, float)
    assert isinstance(answer.model, str) and answer.model


def test_generate_answer_low_confidence_calls_groq():
    results = [_Context(text="Paris is the capital of France.", source="test", score=0.10)]
    answer = generator.generate_answer("What is the capital of France?", results)
    assert isinstance(answer.text, str)
    assert isinstance(answer.grounded, bool)
    assert isinstance(answer.generation_ms, float)
    assert answer.model  # settings.GROQ_MODEL


def test_generate_answer_no_results():
    answer = generator.generate_answer("Some unanswerable question", [])
    assert isinstance(answer.text, str)
    assert answer.grounded is False  # no context at all -> not grounded
    assert isinstance(answer.generation_ms, float)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)