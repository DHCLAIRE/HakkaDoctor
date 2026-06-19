from __future__ import annotations

from hakka_speech_toolkit.interpretation import (
    format_matches_table,
    interpret_text,
    normalize_text,
)


def test_normalize_text_handles_punctuation_and_pinyin_hint():
    """Verify text normalization removes punctuation and maps pinyin hints."""

    assert normalize_text("gao xue ya！") == "高血壓"
    assert normalize_text("請按時吃藥。") == "請按時吃藥"


def test_mandarin_to_hakka_detects_medication_instruction():
    """Verify Mandarin medication instructions map to Hakka output."""

    result = interpret_text("請按時吃藥", direction="mandarin_to_hakka")

    assert result["best_match"]["intent"] == "take_medicine"
    assert "食藥" in result["best_match"]["hakka"]
    assert result["best_match"]["hakka_romanization"]
    assert "Interpretation" in result["summary"]


def test_hakka_to_mandarin_detects_urgent_chest_pain():
    """Verify urgent Hakka chest-pain descriptions are detected."""

    result = interpret_text("𠊎胸坎痛，透氣毋順", direction="hakka_to_mandarin")

    assert result["best_match"]["intent"] == "chest_pain"
    assert result["best_match"]["urgency"] == "urgent"
    assert "urgent clinical attention" in result["summary"]


def test_unknown_phrase_returns_clear_no_match_summary():
    """Verify unknown text returns a clear no-match result."""

    result = interpret_text("今天天氣很好", direction="mandarin_to_hakka")

    assert result["matches"] == []
    assert result["best_match"] is None
    assert "No medical phrase matched" in result["summary"]


def test_matches_table_is_gradio_friendly():
    """Verify interpretation matches format into Gradio table rows."""

    result = interpret_text("血糖高而且口渴", direction="mandarin_to_hakka")
    rows = format_matches_table(result["matches"])

    assert rows
    assert len(rows[0]) == 7
    assert rows[0][3] == "diabetes"
