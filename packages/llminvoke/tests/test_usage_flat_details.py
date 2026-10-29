"""Hermetic tests: flat usage-detail normalization at the llminvoke seam.

Issue python-utils-agentos-cached-tokens-durchreichen: OpenAI-/vLLM-kompatible
Endpunkte melden cached/reasoning nur verschachtelt (prompt_tokens_details.
cached_tokens, completion_tokens_details.reasoning_tokens) — bisher warf jede
Schicht sie still weg. llminvoke ist DER Normalisierer: non-stream
(``_extract_usage``) UND Stream-Usage-Event (``_chunk_events``) stempeln
flache ``cached_tokens``/``reasoning_tokens`` auf. Die historische Verstoßform
(„cached nur nested → Tracker liest 0") darf nie wieder auftreten.
"""

from __future__ import annotations

import types

import llminvoke


class TestExtractUsageStampsFlat:
    def _resp(self, usage):
        return types.SimpleNamespace(usage=usage)

    def test_nested_cached_and_reasoning_stamped_flat(self):
        resp = self._resp({
            "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 12},
        })
        u = llminvoke._extract_usage(resp)
        assert u["cached_tokens"] == 80       # in: 100, davon 80 gecached
        assert u["reasoning_tokens"] == 12    # out: 40, davon 12 Denk-Tokens
        # rohe Felder unangetastet
        assert u["prompt_tokens"] == 100 and u["total_tokens"] == 140
        assert u["prompt_tokens_details"] == {"cached_tokens": 80}

    def test_flat_keys_win_no_override(self):
        resp = self._resp({
            "prompt_tokens": 10, "total_tokens": 12,
            "cached_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 99},
        })
        assert llminvoke._extract_usage(resp)["cached_tokens"] == 3

    def test_no_details_untouched(self):
        resp = self._resp({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
        u = llminvoke._extract_usage(resp)
        assert "cached_tokens" not in u and "reasoning_tokens" not in u

    def test_none_response_and_non_dict(self):
        assert llminvoke._extract_usage(None) is None
        assert llminvoke._extract_usage(types.SimpleNamespace(usage=object())) is None

    def test_input_not_mutated(self):
        raw = {"prompt_tokens": 5, "total_tokens": 6, "prompt_tokens_details": {"cached_tokens": 4}}
        llminvoke._extract_usage(self._resp(raw))
        assert "cached_tokens" not in raw  # Kopie, nie mutieren


class TestStreamUsageEventStampsFlat:
    def _usage_chunk(self, usage):
        return types.SimpleNamespace(
            message=types.SimpleNamespace(content=None, reasoning_content=None),
            thinking=None,
            usage=usage,
        )

    def test_usage_event_carries_flat_cached(self):
        chunk = self._usage_chunk({
            "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 6},
        })
        events = list(llminvoke._chunk_events(chunk))
        usage_events = [d for kind, d in events if kind == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["cached_tokens"] == 6

    def test_pydantic_usage_model_dumped_and_stamped(self):
        """Provider-SDKs liefern pydantic-Usage-Objekte — model_dump() +
        flacher Stempel, kein Crash an der Seam."""
        chunk = self._usage_chunk(None)
        chunk.usage = types.SimpleNamespace(
            prompt_tokens=3, completion_tokens=1, total_tokens=4,
            prompt_tokens_details={"cached_tokens": 2},
            model_dump=lambda: {
                "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        )
        events = list(llminvoke._chunk_events(chunk))
        usage_events = [d for kind, d in events if kind == "usage"]
        assert usage_events[0]["cached_tokens"] == 2
