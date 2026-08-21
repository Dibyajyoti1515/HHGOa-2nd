"""
timing/stage_timer.py

NEW module (not from notebook). Reusable timing collector for a single request/turn.

Two timing styles supported:
1. Span timing — `with collector.stage("name"):` — for stages with a clear start/end
   (guardrails, Qdrant search, etc). Existing behaviour, unchanged from the text-pipeline phase.
2. Point-in-time marks — `collector.mark("name")` — for streaming stages (STT, TTS) where the
   interesting numbers are *gaps between events* (e.g. final-transcript-received -> first-audio-
   chunk-sent), not a single stage duration. Added this phase for the voice pipeline.

`gap_ms(from_mark, to_mark)` computes the ms between two marks, used to build
`stt_to_tts_gap_ms` and friends in api/ws_voice.py.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional


class TimingCollector:
    def __init__(self):
        self._spans_ms: Dict[str, float] = {}
        self._marks_s: Dict[str, float] = {}
        self._turn_start_s: float = time.monotonic()

    # ---- span timing (unchanged from text-pipeline phase) ----

    @contextmanager
    def stage(self, name: str):
        start = time.monotonic()
        try:
            yield
        finally:
            self._spans_ms[name] = (time.monotonic() - start) * 1000.0

    def record_ms(self, name: str, value_ms: Optional[float]) -> None:
        """For stages that already self-time and return their own ms figure
        (retrieval/groq/trim functions ported from the notebook) — just collect it."""
        if value_ms is not None:
            self._spans_ms[name] = value_ms

    def get_ms(self, name: str) -> Optional[float]:
        return self._spans_ms.get(name)

    # ---- point-in-time marks (NEW this phase) ----

    def mark(self, name: str) -> None:
        """Record the current monotonic time under `name`. Safe to call once per name;
        a second call for the same name overwrites (last write wins), useful for retries."""
        self._marks_s[name] = time.monotonic()

    def has_mark(self, name: str) -> bool:
        return name in self._marks_s

    def gap_ms(self, from_mark: str, to_mark: str) -> Optional[float]:
        """ms elapsed between two marks. Returns None if either mark is missing
        (e.g. confidence gate skipped Groq, so there's no 'groq_start' mark)."""
        if from_mark not in self._marks_s or to_mark not in self._marks_s:
            return None
        return (self._marks_s[to_mark] - self._marks_s[from_mark]) * 1000.0

    def since_turn_start_ms(self, mark_name: str) -> Optional[float]:
        if mark_name not in self._marks_s:
            return None
        return (self._marks_s[mark_name] - self._turn_start_s) * 1000.0

    # ---- final assembly ----

    def total_turn_ms(self) -> float:
        return (time.monotonic() - self._turn_start_s) * 1000.0

    def as_dict(self) -> Dict[str, Optional[float]]:
        """Assembles the full timings block for the final websocket message.
        Missing values (e.g. groq_call_ms when the confidence gate skipped Groq) are left as None
        rather than omitted, so the client-side shape is always stable."""
        return {
            "stt_time_to_first_partial_ms": self.since_turn_start_ms("stt_first_partial"),
            "stt_time_to_final_ms": self.since_turn_start_ms("stt_final"),
            "guardrail_check_ms": self.get_ms("guardrail_check"),
            "retrieval_embedding_ms": self.get_ms("retrieval_embedding"),
            "retrieval_qdrant_ms": self.get_ms("retrieval_qdrant"),
            "retrieval_fusion_ms": self.get_ms("retrieval_fusion"),
            "retrieval_total_ms": self.get_ms("retrieval_total"),
            "groq_call_ms": self.get_ms("groq_call"),
            "text_trim_ms": self.get_ms("text_trim"),
            "tts_time_to_first_chunk_ms": self.since_turn_start_ms("tts_first_chunk"),
            "tts_time_to_last_chunk_ms": self.since_turn_start_ms("tts_last_chunk"),
            "stt_to_tts_gap_ms": self.gap_ms("stt_final", "tts_first_chunk"),
            "total_turn_ms": self.total_turn_ms(),
        }