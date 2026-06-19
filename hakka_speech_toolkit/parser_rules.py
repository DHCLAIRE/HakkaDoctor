"""Regex POS-shift rules for the small-data Hakka parser.

ArticutAPI_Hakka uses a large ``posShift.py`` table to repair tags produced by
the Mandarin Articut backbone after Hakka dictionaries have been injected. This
module keeps the same idea offline: rules operate on POS XML fragments and
rewrite the tags when Hakka grammar gives stronger evidence than the first-pass
dictionary tag.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PosShiftRule:
    name: str
    pattern: re.Pattern
    replacement: str
    description: str


POS_SHIFT_RULES = (
    PosShiftRule(
        name="pronoun_before_possessive_particle",
        pattern=re.compile(
            r"(<ENTITY_pronoun>[^<]+</ENTITY_pronoun>)"
            r"<(?:FUNC_inner|ENTITY_oov)>([个介嘅])</(?:FUNC_inner|ENTITY_oov)>"
        ),
        replacement=r"\1<ENTITY_possessive>\2</ENTITY_possessive>",
        description="A particle after a pronoun is possessive, mirroring Hakka 𠊎介/𠊎个.",
    ),
    PosShiftRule(
        name="wu_before_predicate_as_negation",
        pattern=re.compile(
            r"<FUNC_inter>無</FUNC_inter>"
            r"(?=<(?:ACTION_verb|MODIFIER|MODAL|ENTITY_noun|ENTITY_oov)>[^<]+</)"
        ),
        replacement="<FUNC_negation>無</FUNC_negation>",
        description="無 before a predicate or nominal predicate is negation, not only interrogative.",
    ),
    PosShiftRule(
        name="final_negation_as_question",
        pattern=re.compile(r"<FUNC_negation>(無|吂)</FUNC_negation>$"),
        replacement=r"<CLAUSE_Q>\1</CLAUSE_Q>",
        description="Final 無/吂 often marks a yes-no or completive question.",
    ),
    PosShiftRule(
        name="degree_head_oov_modifier",
        pattern=re.compile(
            r"(<FUNC_degreeHead>[^<]+</FUNC_degreeHead>)"
            r"<ENTITY_oov>([^<]+)</ENTITY_oov>"
        ),
        replacement=r"\1<MODIFIER>\2</MODIFIER>",
        description="A word after a degree head is likely adjectival/modifying material.",
    ),
    PosShiftRule(
        name="modal_oov_verb",
        pattern=re.compile(
            r"(<(?:MODAL|FUNC_negation)>[^<]+</(?:MODAL|FUNC_negation)>)"
            r"<ENTITY_oov>([^<]+)</ENTITY_oov>"
        ),
        replacement=r"\1<ACTION_verb>\2</ACTION_verb>",
        description="A word after a modal or negator is likely a predicate verb.",
    ),
    PosShiftRule(
        name="classifier_oov_noun",
        pattern=re.compile(
            r"(<ENTITY_classifier>[^<]+</ENTITY_classifier>)"
            r"<ENTITY_oov>([^<]+)</ENTITY_oov>"
        ),
        replacement=r"\1<ENTITY_noun>\2</ENTITY_noun>",
        description="A word after a classifier is likely a noun.",
    ),
    PosShiftRule(
        name="adjacent_action_compound",
        pattern=re.compile(r"<ACTION_verb>([^<]+)</ACTION_verb><ACTION_verb>([^<]+)</ACTION_verb>"),
        replacement=r"<ACTION_verb>\1\2</ACTION_verb>",
        description="Adjacent verb tokens often form a compound predicate.",
    ),
    PosShiftRule(
        name="adjacent_noun_compound",
        pattern=re.compile(r"<ENTITY_noun>([^<]+)</ENTITY_noun><ENTITY_noun>([^<]+)</ENTITY_noun>"),
        replacement=r"<ENTITY_noun>\1\2</ENTITY_noun>",
        description="Adjacent noun tokens often form a compound noun.",
    ),
    PosShiftRule(
        name="adjacent_modifier_compound",
        pattern=re.compile(r"<MODIFIER>([^<]+)</MODIFIER><MODIFIER>([^<]+)</MODIFIER>"),
        replacement=r"<MODIFIER>\1\2</MODIFIER>",
        description="Adjacent modifiers can combine into one modifier phrase.",
    ),
)


def apply_pos_shift_rules(pos_xml: str) -> tuple[str, list[str]]:
    """Apply regex POS-shift rules until no rule changes the string."""

    applied: list[str] = []
    changed = True
    while changed:
        changed = False
        for rule in POS_SHIFT_RULES:
            shifted, count = rule.pattern.subn(rule.replacement, pos_xml)
            if count:
                pos_xml = shifted
                applied.extend([rule.name] * count)
                changed = True
    return pos_xml, applied
