from __future__ import annotations

import pytest

from hakka_speech_toolkit.articut_adapter import ArticutHakkaParser, HakkaParser


class FakeArticutClient:
    def __init__(self) -> None:
        """Initialize a fake external parser call log."""

        self.calls = []

    def parse(self, text, **kwargs):
        """Record parse calls and return a minimal Articut-like result."""

        self.calls.append((text, kwargs))
        return {
            "status": True,
            "msg": "Success!",
            "result_pos": ["<ENTITY_pronoun>𠊎</ENTITY_pronoun>"],
            "result_segmentation": "𠊎",
        }


def test_articut_adapter_delegates_to_external_client(tmp_path):
    """Verify the adapter forwards parse requests to an external Articut client."""

    dict_path = tmp_path / "udd.json"
    dict_path.write_text('{"ENTITY_noun": ["血氧"]}', encoding="utf-8")
    fake_client = FakeArticutClient()
    parser = ArticutHakkaParser(articut_client=fake_client)

    result = parser.parse("𠊎", userDefinedDictFILE=dict_path)

    assert result["parser_backend"] == "Droidtown ArticutAPI_Hakka"
    assert fake_client.calls[0][1]["userDefinedDictFILE"] == str(dict_path)


def test_articut_adapter_falls_back_when_package_missing():
    """Verify the adapter falls back offline when ArticutAPI_Hakka is unavailable."""

    parser = ArticutHakkaParser(allow_fallback=True)

    result = parser.parse("𠊎毋食藥。")

    assert result["parser_backend"] == "offline_fallback"
    assert result["result_pos"][0] == "<ENTITY_pronoun>𠊎</ENTITY_pronoun>"


def test_unified_parser_offline_backend():
    """Verify the unified facade can force the offline backend."""

    parser = HakkaParser(backend="offline")

    result = parser.parse("𠊎毋食藥。")

    assert result["parser"] == "HakkaRuleParser"


def test_unified_parser_rejects_unknown_backend():
    """Verify invalid backend names are rejected."""

    with pytest.raises(ValueError):
        HakkaParser(backend="not-real")
