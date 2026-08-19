"""Pollinations provider: list_models parsing — types, access, and cost policy.

Regression guard for three catalog bugs found 2026-08-19:

1. **Video mislabeled as chat** — ``nova-reel`` (output: video) was typed
   ``chat`` because the old classifier only checked for image output. Anything
   emitting video/audio/image is not a chat model.
2. **STT mislabeled as chat** — audio-in/text-out models (whisper, universal-*)
   relied on generate_models' derive_type fallback; now classified at the
   source.
3. **No cost** — free models now carry ``{input: 0, output: 0}`` so consumers
   see a real $0 instead of "unknown". Paid models are priced in pollen with no
   published pollen→USD rate, so they get NO cost (never pollen-as-USD) —
   ``access: "paid"`` carries the signal.
"""
from unittest.mock import patch

from uniinfer.providers.pollinations import (
    PollinationsProvider,
    _pollinations_model_type,
)


def _gen_model(name, *, in_mods=("text",), out_mods=("text",), pricing=None, **extra):
    m = {
        "name": name,
        "title": f"Title {name}",
        "input_modalities": list(in_mods),
        "output_modalities": list(out_mods),
    }
    if pricing is not None:
        m["pricing"] = pricing
    m.update(extra)
    return m


class TestModelTypeClassification:
    def test_video_output_is_video(self):
        assert _pollinations_model_type(["text", "image"], ["video"]) == "video"

    def test_audio_output_is_tts(self):
        assert _pollinations_model_type(["text"], ["audio"]) == "tts"

    def test_image_output_is_image(self):
        assert _pollinations_model_type(["text"], ["image"]) == "image"

    def test_audio_in_text_out_is_stt(self):
        assert _pollinations_model_type(["audio"], ["text"]) == "stt"

    def test_text_text_is_chat(self):
        assert _pollinations_model_type(["text"], ["text"]) == "chat"

    def test_multimodal_in_text_out_is_chat(self):
        assert _pollinations_model_type(["text", "image"], ["text"]) == "chat"


class TestListModelsParsing:
    @patch("uniinfer.providers.pollinations.requests.get")
    def test_types_access_and_cost(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            _gen_model(
                "nova-reel",
                in_mods=("text", "image"),
                out_mods=("video",),
                pricing={"currency": "pollen"},  # no token fields -> free
            ),
            _gen_model(
                "whisper",
                in_mods=("audio",),
                out_mods=("text",),
                pricing={"currency": "pollen"},
            ),
            _gen_model(
                "some/community-model:free",
                pricing={"currency": "pollen"},  # free community model
            ),
            _gen_model(
                "openai-fast",
                pricing={
                    "currency": "pollen",
                    "promptTextTokens": "0.0000000375",
                    "completionTextTokens": "0.0000003",
                },
                context_length=400000,
            ),
        ]

        models = {m.id: m for m in PollinationsProvider.list_models()}

        # 1. video never lands in the chat pool
        assert models["nova-reel"].type == "video"
        assert models["nova-reel"].access == "free"

        # 2. STT classified at the source
        assert models["whisper"].type == "stt"

        # 3. free -> explicit zero cost; paid -> no cost, access carries signal
        free = models["some/community-model:free"]
        assert free.access == "free"
        assert free.cost == {"input": 0, "output": 0}

        paid = models["openai-fast"]
        assert paid.access == "paid"
        assert paid.cost is None  # pollen is NOT USD — never fabricate a price
        assert paid.context_window == 400000

    @patch("uniinfer.providers.pollinations.requests.get")
    def test_zero_priced_fields_still_free(self, mock_get):
        """Pollinations expresses free as *absent* fields, but a literal 0
        string must also count as free (defensive)."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            _gen_model(
                "zero-priced",
                pricing={"promptTextTokens": "0", "completionTextTokens": "0"},
            ),
            _gen_model("garbage-price", pricing={"promptTextTokens": "n/a"}),
        ]
        models = {m.id: m for m in PollinationsProvider.list_models()}
        assert models["zero-priced"].access == "free"
        # unparsable pricing is not positive -> free (no false paid marks)
        assert models["garbage-price"].access == "free"
