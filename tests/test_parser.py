from __future__ import annotations

from hakka_speech_toolkit.parser import (
    ArticutLikeHakkaParser,
    HakkaRuleParser,
    build_user_defined_dict,
    normalize_user_defined_dict,
)
from hakka_speech_toolkit.parser_rules import apply_pos_shift_rules


def test_parser_segments_and_tags_known_hakka_sentence():
    """Verify the parser tags a basic negated medication sentence."""

    parser = HakkaRuleParser()

    result = parser.parse("𠊎毋食藥。")
    tokens = result["tokens"]

    assert [token["text"] for token in tokens[:3]] == ["𠊎", "毋", "食藥"]
    assert [token["pos"] for token in tokens[:3]] == ["ENTITY_pronoun", "FUNC_negation", "ACTION_verb"]
    assert result["sentence_patterns"][0]["negated"] is True
    assert result["xbar_tree"]["label"] == "TP"
    assert {"nsubj", "neg", "root"}.issubset({arc["relation"] for arc in result["dependencies"]})


def test_user_defined_dict_overrides_oov_and_uses_articut_shape(tmp_path):
    """Verify user dictionaries override OOV behavior and keep Articut-like output."""

    dict_path = tmp_path / "hakka_rules.json"
    build_user_defined_dict([("血氧", "ENTITY_noun"), ("量", "ACTION_verb")], dict_path)
    parser = ArticutLikeHakkaParser()

    result = parser.parse("請量血氧", userDefinedDictFILE=dict_path)

    assert "<ENTITY_noun>血氧</ENTITY_noun>" in result["result_pos"]
    assert result["result_segmentation"] == "請量╱血氧"
    assert "<ACTION_verb>請量</ACTION_verb>" in result["result_pos"]


def test_simple_word_to_pos_dictionary_is_supported():
    """Verify simple word-to-POS dictionaries normalize correctly."""

    normalized = normalize_user_defined_dict({"當靚": "MODIFIER", "𠊎": "ENTITY_pronoun"})

    assert normalized["MODIFIER"] == ["當靚"]
    assert normalized["ENTITY_pronoun"] == ["𠊎"]


def test_pos_shift_marks_unknown_after_modal_as_verb():
    """Verify OOV material after a modal is parsed as a predicate verb."""

    parser = HakkaRuleParser()

    result = parser.parse("佢愛測血糖")
    token_map = {token["text"]: token["pos"] for token in result["tokens"]}

    assert token_map["測"] == "ACTION_verb"
    assert token_map["血糖"] == "ENTITY_noun"
    assert any(arc["relation"] == "obj" and arc["dependent_text"] == "血糖" for arc in result["dependencies"])
    assert "modal_or_negation_marks_oov_verb" in result["rules_applied"]


def test_possessive_particle_rule():
    """Verify Hakka possessive particles create possessive structures."""

    parser = HakkaRuleParser()

    result = parser.parse("𠊎介藥")
    token_map = {token["text"]: token["pos"] for token in result["tokens"]}

    assert token_map["介"] == "ENTITY_possessive"
    assert result["xbar_tree"]["label"] == "NP"
    assert {"nmod:poss", "case", "root"}.issubset({arc["relation"] for arc in result["dependencies"]})


def test_parser_returns_xbar_and_ud_annotation_metadata():
    """Verify parser output includes X-bar and UD-compatible metadata."""

    result = HakkaRuleParser().parse("佢愛測血糖")

    assert result["annotation_framework"]["framework"] == "UD-compatible dependencies plus X-bar phrase projections."
    assert result["grammar_references"][0]["key"] == "chomsky_1970_xbar"
    assert "xbar_dependency_rules" in result["parser_strategy"]


def test_regex_pos_shift_repairs_articut_style_xml():
    """Verify regex POS-shift repairs Articut-style XML fragments."""

    shifted, applied = apply_pos_shift_rules(
        "<ENTITY_pronoun>𠊎</ENTITY_pronoun>"
        "<FUNC_inner>个</FUNC_inner>"
        "<ENTITY_noun>藥</ENTITY_noun>"
    )

    assert "<ENTITY_possessive>个</ENTITY_possessive>" in shifted
    assert "pronoun_before_possessive_particle" in applied
