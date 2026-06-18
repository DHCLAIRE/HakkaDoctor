#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Gradio app for Mandarin-Hakka medical interpretation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from hakka_speech_toolkit.interpretation import (
    format_matches_table,
    interpret_text,
)


def _device_config() -> tuple[str, Any]:
    import torch

    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def get_mandarin_asr():
    from transformers import pipeline

    device, dtype = _device_config()
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device,
        torch_dtype=dtype,
    )


@lru_cache(maxsize=1)
def get_hakka_asr():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device, dtype = _device_config()
    model_id = "formospeech/whisper-large-v3-taiwanese-hakka"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    return processor, model, device, torch


@lru_cache(maxsize=1)
def get_hakka_tts():
    import torch
    from transformers import AutoTokenizer, VitsModel

    device, _ = _device_config()
    model_id = "facebook/mms-tts-hak"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device, torch


def synthesize_hakka(romanization: str):
    if not romanization:
        return None
    try:
        tokenizer, model, device, torch = get_hakka_tts()
        inputs = tokenizer(romanization, return_tensors="pt").to(device)
        with torch.no_grad():
            waveform = model(**inputs).waveform
        return model.config.sampling_rate, waveform.detach().cpu().numpy().squeeze()
    except Exception:
        return None


def transcribe_mandarin(audio_path: str | None) -> str:
    if not audio_path:
        return ""
    result = get_mandarin_asr()(
        audio_path,
        generate_kwargs={"language": "zh", "task": "transcribe"},
    )
    return str(result.get("text", "")).strip()


def transcribe_hakka(audio_path: str | None) -> tuple[str, str]:
    if not audio_path:
        return "", "0.0%"

    import librosa

    processor, model, device, torch = get_hakka_asr()
    y, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(device)
    forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_decoder_ids=forced_ids,
            return_dict_in_generate=True,
            output_scores=True,
        )

    transcription = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )
    confidence = torch.exp(transition_scores).mean().item() * 100
    return transcription, f"{confidence:.1f}%"


def interpret_doctor_text(text: str):
    result = interpret_text(text, direction="mandarin_to_hakka")
    matches = result["matches"]
    best = result["best_match"] or {}
    audio = synthesize_hakka(str(best.get("hakka_romanization", ""))) if best else None
    return (
        result["summary"],
        str(best.get("hakka", "")),
        str(best.get("hakka_romanization", "")),
        format_matches_table(matches),
        audio,
    )


def interpret_doctor_audio(audio_path: str | None):
    text = transcribe_mandarin(audio_path)
    return interpret_doctor_text(text)


def interpret_patient_text(text: str):
    result = interpret_text(text, direction="hakka_to_mandarin")
    matches = result["matches"]
    best = result["best_match"] or {}
    return (
        result["summary"],
        str(best.get("mandarin", "")),
        format_matches_table(matches),
    )


def interpret_patient_audio(audio_path: str | None):
    text, confidence = transcribe_hakka(audio_path)
    summary, mandarin, table = interpret_patient_text(text)
    return f"{summary}\nASR confidence: {confidence}", mandarin, confidence, table


def build_demo() -> gr.Blocks:
    match_headers = ["Score", "Urgency", "Category", "Intent", "Mandarin", "Hakka", "Keywords"]

    with gr.Blocks(title="Hakka Doctor Medical Interpreter") as demo:
        gr.Markdown("# Hakka Doctor Medical Interpreter")
        gr.Markdown("Mandarin-to-Hakka doctor instructions and Hakka-to-Mandarin patient symptom support.")

        with gr.Tab("Doctor to Patient"):
            with gr.Row():
                doctor_text = gr.Textbox(
                    label="Mandarin text",
                    lines=3,
                    placeholder="例如：請按時吃藥，並且多喝水。",
                )
                doctor_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Mandarin audio")
            with gr.Row():
                doctor_summary = gr.Textbox(label="Interpretation", lines=7)
                hakka_text = gr.Textbox(label="Hakka output", lines=3)
                hakka_rom = gr.Textbox(label="Hakka romanization for TTS", lines=3)
            doctor_matches = gr.Dataframe(headers=match_headers, label="Candidate matches")
            hakka_audio = gr.Audio(label="Spoken Hakka")
            with gr.Row():
                gr.Button("Interpret text", variant="primary").click(
                    interpret_doctor_text,
                    inputs=doctor_text,
                    outputs=[doctor_summary, hakka_text, hakka_rom, doctor_matches, hakka_audio],
                )
                gr.Button("Transcribe and interpret audio").click(
                    interpret_doctor_audio,
                    inputs=doctor_audio,
                    outputs=[doctor_summary, hakka_text, hakka_rom, doctor_matches, hakka_audio],
                )

        with gr.Tab("Patient to Doctor"):
            with gr.Row():
                patient_text = gr.Textbox(
                    label="Hakka text or ASR transcript",
                    lines=3,
                    placeholder="例如：𠊎胸坎痛，透氣毋順。",
                )
                patient_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Hakka audio")
            with gr.Row():
                patient_summary = gr.Textbox(label="Analysis", lines=7)
                mandarin_output = gr.Textbox(label="Mandarin clinical meaning", lines=3)
                confidence = gr.Textbox(label="ASR confidence")
            patient_matches = gr.Dataframe(headers=match_headers, label="Candidate matches")
            with gr.Row():
                gr.Button("Interpret text", variant="primary").click(
                    interpret_patient_text,
                    inputs=patient_text,
                    outputs=[patient_summary, mandarin_output, patient_matches],
                )
                gr.Button("Transcribe and interpret audio").click(
                    interpret_patient_audio,
                    inputs=patient_audio,
                    outputs=[patient_summary, mandarin_output, confidence, patient_matches],
                )

    return demo


demo = build_demo()


if __name__ == "__main__":
    is_notebook = Path.cwd().name == "content"
    demo.launch(inbrowser=not is_notebook, share=is_notebook)
