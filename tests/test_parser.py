from __future__ import annotations

from hakka_speech_toolkit.parser import (
    ArticutLikeHakkaParser,
    HakkaRuleParser,
    build_user_defined_dict,
    normalize_user_defined_dict,
)


def test_parser_segments_and_tags_known_hakka_sentence():
    parser = HakkaRuleParser()

    result = parser.parse("𠊎毋食藥。")
    tokens = result["tokens"]

    assert [token["text"] for token in tokens[:3]] == ["𠊎", "毋", "食藥"]
    assert [token["pos"] for token in tokens[:3]] == ["ENTITY_pronoun", "FUNC_negation", "ACTION_verb"]
    assert result["sentence_patterns"][0]["negated"] is True


def test_user_defined_dict_overrides_oov_and_uses_articut_shape(tmp_path):
    dict_path = tmp_path / "hakka_rules.json"
    build_user_defined_dict([("血氧", "ENTITY_noun"), ("量", "ACTION_verb")], dict_path)
    parser = ArticutLikeHakkaParser()

    result = parser.parse("請量血氧", userDefinedDictFILE=dict_path)

    assert "<ENTITY_noun>血氧</ENTITY_noun>" in result["result_pos"]
    assert result["result_segmentation"] == "請量╱血氧"
    assert "<ACTION_verb>請量</ACTION_verb>" in result["result_pos"]


def test_simple_word_to_pos_dictionary_is_supported():
    normalized = normalize_user_defined_dict({"當靚": "MODIFIER", "𠊎": "ENTITY_pronoun"})

    assert normalized["MODIFIER"] == ["當靚"]
    assert normalized["ENTITY_pronoun"] == ["𠊎"]


def test_pos_shift_marks_unknown_after_modal_as_verb():
    parser = HakkaRuleParser()

    result = parser.parse("佢愛測血糖")
    token_map = {token["text"]: token["pos"] for token in result["tokens"]}

    assert token_map["測血糖"] == "ACTION_verb"
    assert "modal_or_negation_marks_oov_verb" in result["rules_applied"]


def test_possessive_particle_rule():
    parser = HakkaRuleParser()

    result = parser.parse("𠊎介藥")
    token_map = {token["text"]: token["pos"] for token in result["tokens"]}

    assert token_map["介"] == "ENTITY_possessive"
