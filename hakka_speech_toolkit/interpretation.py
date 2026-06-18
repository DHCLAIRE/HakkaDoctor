"""Mandarin-Hakka medical interpretation helpers.

The functions in this module keep the clinical phrase matching independent from
ASR/TTS models so it can be tested quickly and reused by Gradio, notebooks, or
future mobile interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable
import re


Direction = str
Urgency = str


@dataclass(frozen=True)
class MedicalPhrase:
    intent: str
    category: str
    urgency: Urgency
    mandarin: str
    hakka: str
    hakka_romanization: str
    mandarin_keywords: tuple[str, ...]
    hakka_keywords: tuple[str, ...]
    response_hint: str


MEDICAL_PHRASES: tuple[MedicalPhrase, ...] = (
    MedicalPhrase(
        intent="take_medicine",
        category="Medication",
        urgency="routine",
        mandarin="請按時吃藥。",
        hakka="請照時間食藥。",
        hakka_romanization="qiung zeu sii gian siit rhog",
        mandarin_keywords=("吃藥", "服藥", "服用", "藥", "按時"),
        hakka_keywords=("食藥", "藥仔", "按時", "rhog", "yok"),
        response_hint="Confirm dose, frequency, and whether the patient has allergies.",
    ),
    MedicalPhrase(
        intent="drink_water",
        category="Self care",
        urgency="routine",
        mandarin="請多喝水，也要休息。",
        hakka="請加兜飲水，也愛歇睏。",
        hakka_romanization="qiung ga deu rhim sui rha oi hiet kun",
        mandarin_keywords=("喝水", "多喝水", "補充水分", "休息"),
        hakka_keywords=("飲水", "加兜水", "歇睏", "rhim sui"),
        response_hint="Useful for fever, dehydration risk, or general recovery instructions.",
    ),
    MedicalPhrase(
        intent="open_mouth",
        category="Examination",
        urgency="routine",
        mandarin="請張開嘴巴，我要檢查喉嚨。",
        hakka="請嘴擘開，𠊎愛檢查喉嗹。",
        hakka_romanization="qiung zoi bag koi ngai oi giam ca heu lien",
        mandarin_keywords=("張開嘴", "張嘴", "嘴巴", "喉嚨", "檢查喉嚨"),
        hakka_keywords=("嘴擘開", "喉嗹", "zoi", "heu lien"),
        response_hint="Use before throat or oral exam.",
    ),
    MedicalPhrase(
        intent="deep_breath",
        category="Examination",
        urgency="routine",
        mandarin="請深呼吸，我要聽肺部。",
        hakka="請大氣透，𠊎愛聽肺部。",
        hakka_romanization="qiung tai hi teu ngai oi ten fi pu",
        mandarin_keywords=("深呼吸", "吸氣", "吐氣", "聽肺", "肺部"),
        hakka_keywords=("大氣透", "透氣", "肺部", "tai hi teu"),
        response_hint="Use during auscultation.",
    ),
    MedicalPhrase(
        intent="high_blood_pressure",
        category="Diagnosis",
        urgency="warning",
        mandarin="您的血壓偏高，需要追蹤。",
        hakka="你个血壓較高，愛繼續追蹤。",
        hakka_romanization="ngi ge hiet ab ka go oi gi xiug zui zung",
        mandarin_keywords=("高血壓", "血壓高", "血壓偏高", "血壓"),
        hakka_keywords=("血壓高", "血壓", "hiet ab", "頭那暈", "暈暈"),
        response_hint="Ask about headache, dizziness, chest pain, medication use, and home BP logs.",
    ),
    MedicalPhrase(
        intent="diabetes",
        category="Diagnosis",
        urgency="warning",
        mandarin="您的血糖偏高，要注意糖尿病。",
        hakka="你个血糖較高，愛注意糖尿病。",
        hakka_romanization="ngi ge hiet tong ka go oi zu yi tong ngiau piang",
        mandarin_keywords=("糖尿病", "血糖", "血糖高", "口渴", "多尿"),
        hakka_keywords=("糖尿病", "血糖", "嘴渴", "尿多", "tong ngiau"),
        response_hint="Ask about thirst, urination, weight change, diet, and current medicine.",
    ),
    MedicalPhrase(
        intent="pain",
        category="Symptom",
        urgency="warning",
        mandarin="您哪裡會痛？痛多久了？",
        hakka="你哪位會痛？痛幾久咧？",
        hakka_romanization="ngi nai vi voi tung tung gid giu le",
        mandarin_keywords=("痛", "疼", "哪裡痛", "多久"),
        hakka_keywords=("會痛", "痛", "肚屎痛", "頭那痛", "胸坎痛", "tung"),
        response_hint="Ask location, onset, severity, triggers, and radiation.",
    ),
    MedicalPhrase(
        intent="chest_pain",
        category="Emergency symptom",
        urgency="urgent",
        mandarin="胸痛可能很危險，請立刻讓醫護人員知道。",
        hakka="胸坎痛可能當危險，請黏時分醫護人員知。",
        hakka_romanization="hiung kam tung ko nen dong fui hiam qiung niam sii bun rh i fu ngin rhan di",
        mandarin_keywords=("胸痛", "胸口痛", "喘", "呼吸困難"),
        hakka_keywords=("胸坎痛", "喘", "透氣毋順", "hiung kam tung"),
        response_hint="Treat as urgent: check vitals and escalate to clinical staff.",
    ),
    MedicalPhrase(
        intent="allergy",
        category="Safety",
        urgency="urgent",
        mandarin="您有沒有藥物過敏？",
        hakka="你有無藥仔過敏？",
        hakka_romanization="ngi rhiu mo rhog e go min",
        mandarin_keywords=("過敏", "藥物過敏", "會不會過敏"),
        hakka_keywords=("過敏", "藥仔過敏", "go min"),
        response_hint="Medication allergy should be checked before prescribing or injection.",
    ),
)


PINYIN_HINTS = {
    "gao xue ya": "高血壓",
    "xue ya": "血壓",
    "tang niao bing": "糖尿病",
    "chi yao": "吃藥",
    "he shui": "喝水",
    "shen hu xi": "深呼吸",
    "zhang zui": "張嘴",
    "xiong tong": "胸痛",
}


def normalize_text(text: str | None) -> str:
    """Normalize ASR output for keyword matching."""

    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = normalized.replace("臺", "台").replace("裏", "裡")
    normalized = re.sub(r"[\s，,。.!！？?；;：「」『』（）()\[\]\"']", "", normalized)
    for romanized, hanzi in PINYIN_HINTS.items():
        normalized = normalized.replace(romanized.replace(" ", ""), hanzi)
    return normalized


def keyword_score(text: str, keywords: Iterable[str]) -> tuple[float, list[str]]:
    """Score exact and fuzzy keyword evidence."""

    normalized = normalize_text(text)
    matched = []
    score = 0.0
    for keyword in keywords:
        key = normalize_text(keyword)
        if not key:
            continue
        if key in normalized:
            matched.append(keyword)
            score += 1.0 + min(len(key), 6) / 10.0
        else:
            similarity = SequenceMatcher(None, normalized, key).ratio()
            if similarity >= 0.82:
                matched.append(keyword)
                score += similarity * 0.6
    return score, matched


def interpret_text(
    text: str | None,
    direction: Direction,
    max_results: int = 3,
) -> dict[str, object]:
    """Interpret transcribed Mandarin or Hakka clinical text."""

    if not text or not text.strip():
        return {
            "input_text": text or "",
            "direction": direction,
            "matches": [],
            "best_match": None,
            "summary": "No text provided.",
        }

    scored = []
    for phrase in MEDICAL_PHRASES:
        keywords = phrase.mandarin_keywords if direction == "mandarin_to_hakka" else phrase.hakka_keywords
        score, matched_keywords = keyword_score(text, keywords)
        if score > 0:
            scored.append((score, phrase, matched_keywords))

    scored.sort(key=lambda item: (item[0], item[1].urgency == "urgent"), reverse=True)
    matches = [
        {
            "intent": phrase.intent,
            "category": phrase.category,
            "urgency": phrase.urgency,
            "score": round(score, 3),
            "matched_keywords": matched_keywords,
            "mandarin": phrase.mandarin,
            "hakka": phrase.hakka,
            "hakka_romanization": phrase.hakka_romanization,
            "response_hint": phrase.response_hint,
        }
        for score, phrase, matched_keywords in scored[:max_results]
    ]
    best_match = matches[0] if matches else None
    summary = build_summary(text, direction, matches)
    return {
        "input_text": text,
        "direction": direction,
        "matches": matches,
        "best_match": best_match,
        "summary": summary,
    }


def build_summary(text: str, direction: Direction, matches: list[dict[str, object]]) -> str:
    """Build a concise, user-facing interpretation summary."""

    label = "Mandarin" if direction == "mandarin_to_hakka" else "Hakka"
    lines = [f"{label} transcript: {text}"]
    if not matches:
        lines.append("No medical phrase matched. Try a shorter phrase or add this wording to the corpus.")
        return "\n".join(lines)

    best = matches[0]
    target = best["hakka"] if direction == "mandarin_to_hakka" else best["mandarin"]
    lines.append(f"Best match: {best['category']} / {best['intent']} ({best['urgency']})")
    lines.append(f"Interpretation: {target}")
    if direction == "mandarin_to_hakka":
        lines.append(f"TTS romanization: {best['hakka_romanization']}")
    lines.append(f"Matched keywords: {', '.join(best['matched_keywords'])}")
    lines.append(f"Clinical note: {best['response_hint']}")
    if best["urgency"] == "urgent":
        lines.append("Escalation: urgent clinical attention recommended.")
    return "\n".join(lines)


def format_matches_table(matches: list[dict[str, object]]) -> list[list[object]]:
    """Return rows suitable for a Gradio dataframe."""

    return [
        [
            match["score"],
            match["urgency"],
            match["category"],
            match["intent"],
            match["mandarin"],
            match["hakka"],
            ", ".join(match["matched_keywords"]),
        ]
        for match in matches
    ]
