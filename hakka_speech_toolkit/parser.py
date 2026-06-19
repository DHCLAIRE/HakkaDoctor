"""Small-data Hakka parser inspired by ArticutAPI_Hakka.

This parser is intentionally rule-first. It uses layered POS dictionaries,
forward maximum matching, user-defined dictionary injection, and post-token POS
shift rules. That mirrors the useful parts of ArticutAPI_Hakka for low-resource
settings while keeping the core parser offline and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re

from .parser_rules import apply_pos_shift_rules


ARTICUT_COMPATIBLE_POS = (
    "ACTION_verb",
    "ACTION_lightVerb",
    "ACTION_quantifiedVerb",
    "ACTION_eventQuantifier",
    "ASPECT",
    "AUX",
    "CLAUSE_particle",
    "CLAUSE_Q",
    "ENTITY_classifier",
    "ENTITY_DetPhrase",
    "ENTITY_measurement",
    "ENTITY_noun",
    "ENTITY_num",
    "ENTITY_person",
    "ENTITY_possessive",
    "ENTITY_pronoun",
    "ENTITY_oov",
    "FUNC_conjunction",
    "FUNC_degreeHead",
    "FUNC_inner",
    "FUNC_inter",
    "FUNC_negation",
    "IDIOM",
    "LOCATION",
    "MODAL",
    "MODIFIER",
    "MODIFIER_color",
    "QUANTIFIER",
    "RANGE_locality",
    "RANGE_period",
    "TIME_justtime",
    "TIME_season",
)


DEFAULT_LEXICON: Dict[str, Tuple[str, ...]] = {
    "ENTITY_pronoun": ("𠊎", "吾", "我", "你", "汝", "佢", "𠊎兜", "你兜", "佢兜"),
    "ENTITY_possessive": ("个", "介", "嘅"),
    "FUNC_negation": ("毋", "唔", "無", "吂", "毋係", "毋好", "毋使"),
    "FUNC_inter": ("無", "敢", "係無", "有無"),
    "FUNC_conjunction": ("摎", "同", "還係", "抑", "毋過", "但係", "所以"),
    "FUNC_degreeHead": ("當", "恁", "盡", "蓋", "較", "過", "十分"),
    "FUNC_inner": ("个", "仔", "等", "還", "就", "斯"),
    "FUNC_determiner": ("這", "該", "這隻", "該隻", "這兜", "該兜"),
    "MODAL": ("愛", "會", "做得", "使得", "應該", "做毋得"),
    "ASPECT": ("等", "著", "過", "忒", "核", "在", "緊"),
    "CLAUSE_Q": ("無", "吂", "麼个", "哪位", "幾久", "幾多", "敢有"),
    "CLAUSE_particle": ("啊", "咧", "哪", "呢", "喔", "啦"),
    "ACTION_verb": (
        "食",
        "飲",
        "講",
        "行",
        "看",
        "聽",
        "做",
        "去",
        "來",
        "轉",
        "痛",
        "暈",
        "咳",
        "吐",
        "發燒",
        "食藥",
        "檢查",
        "深呼吸",
        "大氣透",
        "張嘴",
        "嘴擘開",
        "量",
        "請",
        "打針",
        "洗",
        "睡",
        "尞",
    ),
    "ENTITY_noun": (
        "藥",
        "藥仔",
        "水",
        "飯",
        "粥",
        "身體",
        "頭那",
        "肚屎",
        "胸坎",
        "喉嗹",
        "血壓",
        "血糖",
        "糖尿病",
        "高血壓",
        "醫生",
        "護士",
        "病院",
        "屋下",
        "症狀",
        "問題",
        "過敏",
        "藥仔過敏",
    ),
    "ENTITY_classifier": ("隻", "個", "條", "張", "項", "擺", "杯", "包", "粒"),
    "ENTITY_num": ("零", "一", "兩", "二", "三", "四", "五", "六", "七", "八", "九", "十", "半"),
    "TIME_justtime": ("今晡日", "韶早", "昨日", "等一下", "一下", "朝晨", "暗晡", "禮拜"),
    "RANGE_locality": ("上背", "下背", "裡背", "外背", "脣", "項", "肚"),
    "MODIFIER": ("好", "壞", "靚", "嚴重", "舒服", "毋舒服", "痛", "暈", "燒", "冷"),
    "LOCATION": ("醫院", "診所", "屋下", "病房", "急診"),
    "IDIOM": ("食飽吂", "無問題"),
}


PUNCTUATION = set("，,。.!！？?；;：:\n\t ")
LATIN_TOKEN_RE = re.compile(r"[A-Za-zÀ-ȕ\u0300-\u036f]+(?:[-'][A-Za-zÀ-ȕ\u0300-\u036f]+)*")


@dataclass(frozen=True)
class ParseToken:
    text: str
    pos: str
    start: int
    end: int
    source: str = "default"

    def to_obj(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "pos": self.pos,
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }

    def to_pos_xml(self) -> str:
        if self.pos == "PUNCTUATION":
            return self.text
        return "<{0}>{1}</{0}>".format(self.pos, self.text)


class HakkaRuleParser:
    """Offline Hakka parser for tiny datasets and domain dictionaries."""

    def __init__(
        self,
        user_defined_dict: Optional[Dict[str, Any]] = None,
        max_oov_group: int = 4,
    ) -> None:
        self.max_oov_group = max_oov_group
        self.lexicon = self._build_lexicon(user_defined_dict or {})

    @classmethod
    def from_user_defined_file(cls, user_defined_dict_file: str | Path) -> "HakkaRuleParser":
        return cls(load_user_defined_dict(user_defined_dict_file))

    def parse(
        self,
        input_text: str,
        level: str = "lv2",
        userDefinedDictFILE: str | Path | None = None,
        user_defined_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse Hakka text with Articut-like return fields."""

        if input_text is None:
            input_text = ""

        runtime_dict: Dict[str, Any] = {}
        if userDefinedDictFILE:
            runtime_dict.update(load_user_defined_dict(userDefinedDictFILE))
        if user_defined_dict:
            runtime_dict.update(user_defined_dict)

        lexicon = self.lexicon if not runtime_dict else self._build_lexicon(runtime_dict, base=self.lexicon)
        tokens = self._tokenize(input_text, lexicon)
        first_pass_tokens, heuristic_rules = self._apply_context_heuristics(tokens)
        pos_xml = "".join(token.to_pos_xml() for token in first_pass_tokens if token.pos != "PUNCTUATION")
        shifted_pos_xml, regex_rules = apply_pos_shift_rules(pos_xml)
        tokens = self._tokens_from_pos_xml(shifted_pos_xml, first_pass_tokens)
        spans = self._infer_sentence_spans(tokens)

        result_pos = [token.to_pos_xml() for token in tokens if token.pos != "PUNCTUATION"]
        result_segmentation = "╱".join(token.text for token in tokens if token.text.strip())
        result_obj = [[token.to_obj() for token in tokens if token.text.strip()]]
        rules_applied = heuristic_rules + regex_rules
        result = {
            "status": True,
            "msg": "Success!",
            "input": input_text,
            "level": level,
            "result_segmentation": result_segmentation,
            "result_pos": result_pos,
            "result_obj": result_obj,
            "tokens": [token.to_obj() for token in tokens if token.text.strip()],
            "sentence_patterns": spans,
            "rules_applied": rules_applied,
            "lexicon_sources": {
                "default": "Small seed lexicon modeled after Articut POS categories.",
                "user_defined": "Caller-provided domain or dataset vocabulary.",
                "pos_shift": "Regex grammar repairs applied after first-pass tagging.",
            },
            "parser": "HakkaRuleParser",
            "parser_strategy": [
                "forward_maximum_matching",
                "articut_style_pos_dictionary",
                "user_defined_dictionary_merge",
                "token_context_heuristics",
                "regex_pos_shift",
                "small_data_oov_heuristics",
            ],
        }
        return result

    def _build_lexicon(
        self,
        user_defined_dict: Dict[str, Any],
        base: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        lexicon = {pos: dict(words) for pos, words in (base or {}).items()}
        if not lexicon:
            for pos, words in DEFAULT_LEXICON.items():
                lexicon.setdefault(pos, {})
                for word in words:
                    lexicon[pos][word] = "default"

        normalized = normalize_user_defined_dict(user_defined_dict)
        for pos, words in normalized.items():
            lexicon.setdefault(pos, {})
            for word in words:
                for other_pos in list(lexicon.keys()):
                    if other_pos != pos:
                        lexicon[other_pos].pop(word, None)
                lexicon[pos][word] = "user_defined"
        return lexicon

    def _tokenize(self, text: str, lexicon: Dict[str, Dict[str, str]]) -> List[ParseToken]:
        words_by_length = sorted(
            ((word, pos, source) for pos, words in lexicon.items() for word, source in words.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        tokens: List[ParseToken] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char in PUNCTUATION:
                tokens.append(ParseToken(char, "PUNCTUATION", index, index + 1))
                index += 1
                continue

            latin = LATIN_TOKEN_RE.match(text, index)
            if latin:
                token_text = latin.group(0)
                tokens.append(ParseToken(token_text, "ENTITY_oov", index, latin.end(), "latin_oov"))
                index = latin.end()
                continue

            match = self._find_dictionary_match(text, index, words_by_length)
            if match:
                tokens.append(match)
                index = match.end
                continue

            end = self._find_oov_end(text, index, words_by_length)
            tokens.append(ParseToken(text[index:end], "ENTITY_oov", index, end, "oov"))
            index = end

        return self._split_oov_near_known_tokens(tokens, lexicon)

    @staticmethod
    def _find_dictionary_match(
        text: str,
        index: int,
        words_by_length: Sequence[Tuple[str, str, str]],
    ) -> Optional[ParseToken]:
        for word, pos, source in words_by_length:
            if text.startswith(word, index):
                return ParseToken(word, pos, index, index + len(word), source)
        return None

    def _find_oov_end(
        self,
        text: str,
        index: int,
        words_by_length: Sequence[Tuple[str, str, str]],
    ) -> int:
        end = index + 1
        while end < min(len(text), index + self.max_oov_group):
            if text[end] in PUNCTUATION:
                break
            if self._find_dictionary_match(text, end, words_by_length):
                break
            end += 1
        return end

    def _split_oov_near_known_tokens(
        self,
        tokens: List[ParseToken],
        lexicon: Dict[str, Dict[str, str]],
    ) -> List[ParseToken]:
        known_chars = {
            word
            for pos in ("ENTITY_pronoun", "FUNC_negation", "ENTITY_possessive", "CLAUSE_Q")
            for word in lexicon.get(pos, {})
            if len(word) == 1
        }
        output: List[ParseToken] = []
        for token in tokens:
            if token.pos != "ENTITY_oov" or len(token.text) <= 1:
                output.append(token)
                continue
            cursor = token.start
            buffer = ""
            buffer_start = cursor
            for char in token.text:
                if char in known_chars:
                    if buffer:
                        output.append(ParseToken(buffer, "ENTITY_oov", buffer_start, cursor, token.source))
                        buffer = ""
                    matched = self._single_char_token(char, cursor, lexicon)
                    output.append(matched)
                    buffer_start = cursor + 1
                else:
                    if not buffer:
                        buffer_start = cursor
                    buffer += char
                cursor += 1
            if buffer:
                output.append(ParseToken(buffer, "ENTITY_oov", buffer_start, cursor, token.source))
        return output

    @staticmethod
    def _single_char_token(char: str, index: int, lexicon: Dict[str, Dict[str, str]]) -> ParseToken:
        for pos in ("ENTITY_pronoun", "FUNC_negation", "ENTITY_possessive", "CLAUSE_Q"):
            if char in lexicon.get(pos, {}):
                return ParseToken(char, pos, index, index + 1, lexicon[pos][char])
        return ParseToken(char, "ENTITY_oov", index, index + 1, "oov")

    def _apply_context_heuristics(self, tokens: List[ParseToken]) -> Tuple[List[ParseToken], List[str]]:
        shifted = list(tokens)
        rules_applied: List[str] = []

        for index, token in enumerate(list(shifted)):
            previous = self._previous_content_token(shifted, index)
            nxt = self._next_content_token(shifted, index)

            if token.text in ("个", "介", "嘅") and previous and previous.pos == "ENTITY_pronoun":
                shifted[index] = self._replace_pos(token, "ENTITY_possessive")
                rules_applied.append("pronoun_possessive_particle")

            if token.text == "無" and nxt and nxt.pos in ("ACTION_verb", "MODIFIER", "MODAL", "ENTITY_noun"):
                shifted[index] = self._replace_pos(token, "FUNC_negation")
                rules_applied.append("wu_before_predicate_as_negation")

            if token.text in ("無", "吂") and nxt is None:
                shifted[index] = self._replace_pos(token, "CLAUSE_Q")
                rules_applied.append("final_negation_as_question_particle")

            if token.pos == "ENTITY_oov" and previous and previous.pos in ("FUNC_degreeHead", "MODIFIER"):
                shifted[index] = self._replace_pos(token, "MODIFIER")
                rules_applied.append("degree_head_marks_modifier")

            if token.pos == "ENTITY_oov" and previous and previous.pos in ("MODAL", "FUNC_negation"):
                shifted[index] = self._replace_pos(token, "ACTION_verb")
                rules_applied.append("modal_or_negation_marks_oov_verb")

            if token.pos == "ENTITY_oov" and previous and previous.pos == "ENTITY_classifier":
                shifted[index] = self._replace_pos(token, "ENTITY_noun")
                rules_applied.append("classifier_marks_oov_noun")

        return shifted, rules_applied

    @staticmethod
    def _tokens_from_pos_xml(pos_xml: str, fallback_tokens: Sequence[ParseToken]) -> List[ParseToken]:
        parsed = [
            ParseToken(match.group("text"), match.group("pos"), 0, 0, "pos_shift")
            for match in re.finditer(r"<(?P<pos>[^>]+)>(?P<text>[^<]+)</(?P=pos)>", pos_xml)
        ]
        if not parsed:
            return list(fallback_tokens)

        source_by_text: Dict[str, str] = {}
        cursor_by_text: Dict[str, int] = {}
        for token in fallback_tokens:
            source_by_text.setdefault(token.text, token.source)
            cursor_by_text.setdefault(token.text, token.start)

        rebuilt = []
        search_start = 0
        for token in parsed:
            start = cursor_by_text.get(token.text)
            if start is None or start < search_start:
                start = search_start
            end = start + len(token.text)
            rebuilt.append(ParseToken(token.text, token.pos, start, end, source_by_text.get(token.text, token.source)))
            search_start = end
        return rebuilt

    @staticmethod
    def _replace_pos(token: ParseToken, pos: str) -> ParseToken:
        return ParseToken(token.text, pos, token.start, token.end, token.source)

    @staticmethod
    def _previous_content_token(tokens: Sequence[ParseToken], index: int) -> Optional[ParseToken]:
        for cursor in range(index - 1, -1, -1):
            if tokens[cursor].pos != "PUNCTUATION":
                return tokens[cursor]
        return None

    @staticmethod
    def _next_content_token(tokens: Sequence[ParseToken], index: int) -> Optional[ParseToken]:
        for cursor in range(index + 1, len(tokens)):
            if tokens[cursor].pos != "PUNCTUATION":
                return tokens[cursor]
        return None

    def _infer_sentence_spans(self, tokens: Sequence[ParseToken]) -> List[Dict[str, Any]]:
        patterns = []
        content = [token for token in tokens if token.pos != "PUNCTUATION"]
        for index, token in enumerate(content):
            if token.pos == "ACTION_verb":
                subject = self._nearest_left(content, index, {"ENTITY_pronoun", "ENTITY_person", "ENTITY_noun"})
                obj = self._nearest_right(content, index, {"ENTITY_noun", "ENTITY_oov", "ENTITY_person"})
                neg = self._nearest_left(content, index, {"FUNC_negation"}, max_distance=2)
                patterns.append(
                    {
                        "pattern": "SVO" if subject and obj else "predicate",
                        "subject": subject.text if subject else None,
                        "predicate": token.text,
                        "object": obj.text if obj else None,
                        "negated": bool(neg),
                        "start": (subject or token).start,
                        "end": (obj or token).end,
                    }
                )
        return patterns

    @staticmethod
    def _nearest_left(
        tokens: Sequence[ParseToken],
        index: int,
        pos_set: set,
        max_distance: int = 6,
    ) -> Optional[ParseToken]:
        for distance, cursor in enumerate(range(index - 1, -1, -1), start=1):
            if distance > max_distance:
                break
            if tokens[cursor].pos in pos_set:
                return tokens[cursor]
        return None

    @staticmethod
    def _nearest_right(
        tokens: Sequence[ParseToken],
        index: int,
        pos_set: set,
        max_distance: int = 6,
    ) -> Optional[ParseToken]:
        for distance, cursor in enumerate(range(index + 1, len(tokens)), start=1):
            if distance > max_distance:
                break
            if tokens[cursor].pos in pos_set:
                return tokens[cursor]
        return None


class ArticutLikeHakkaParser(HakkaRuleParser):
    """Compatibility alias with Articut-style method naming."""


def normalize_user_defined_dict(user_defined_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """Accept Articut-style POS lists or simple ``word -> pos`` mappings."""

    normalized: Dict[str, List[str]] = {}
    for key, value in user_defined_dict.items():
        if key in ARTICUT_COMPATIBLE_POS or key.startswith("KNOWLEDGE_"):
            if isinstance(value, str):
                words = [value]
            else:
                words = [str(item) for item in value]
            normalized.setdefault(key, []).extend(word for word in words if word)
        elif isinstance(value, str):
            normalized.setdefault(value, []).append(key)
        elif isinstance(value, dict) and "pos" in value:
            normalized.setdefault(str(value["pos"]), []).append(key)
    return normalized


def load_user_defined_dict(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        return json.load(file)


def build_user_defined_dict(
    rows: Iterable[Tuple[str, str]],
    output_path: str | Path | None = None,
) -> Dict[str, List[str]]:
    """Build an Articut-style user-defined dictionary from small labeled data."""

    result: Dict[str, List[str]] = {}
    for word, pos in rows:
        if not word or not pos:
            continue
        result.setdefault(pos, [])
        if word not in result[pos]:
            result[pos].append(word)

    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
    return result
