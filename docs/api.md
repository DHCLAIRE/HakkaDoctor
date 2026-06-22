# Hakka Doctor API Reference

This reference covers the public functions and classes intended for app integration, notebooks, and research scripts.

## App Module: `HakkaDoctor_app.py`

Importing `HakkaDoctor_app.py` creates a Gradio `demo` object by calling `build_demo()`.

### `build_demo() -> gr.Blocks`

Builds the Gradio UI with two workflows:

* Doctor Mandarin text/audio to Hakka patient instructions.
* Patient Hakka text/audio to Mandarin clinical meaning.

Example:

```python
from HakkaDoctor_app import build_demo

demo = build_demo()
demo.launch()
```

### `interpret_doctor_text(text)`

Converts Mandarin clinical text into Hakka output.

Returns a tuple for Gradio outputs:

```python
(
    summary,
    hakka_text,
    hakka_romanization,
    candidate_rows,
    hakka_audio,
)
```

`hakka_audio` is either `(sample_rate, waveform)` or `None` if TTS is unavailable.

### `interpret_doctor_audio(audio_path)`

Runs Mandarin ASR with `transcribe_mandarin(audio_path)`, then calls `interpret_doctor_text(text)`.

Returns the same tuple as `interpret_doctor_text`.

### `interpret_patient_text(text)`

Converts Hakka patient text into Mandarin clinical meaning.

Returns:

```python
(
    summary,
    mandarin_clinical_meaning,
    candidate_rows,
)
```

### `interpret_patient_audio(audio_path)`

Runs Hakka ASR with `transcribe_hakka(audio_path)`, then calls `interpret_patient_text(text)`.

Returns:

```python
(
    summary_with_asr_confidence,
    mandarin_clinical_meaning,
    confidence,
    candidate_rows,
)
```

### `transcribe_mandarin(audio_path) -> str`

Uses `openai/whisper-base` through a cached Hugging Face ASR pipeline.

Returns a stripped transcript string. Empty input returns `""`.

### `transcribe_hakka(audio_path) -> tuple[str, str]`

Uses `formospeech/whisper-large-v3-taiwanese-hakka`.

Returns:

```python
(transcription, confidence_percent)
```

`confidence_percent` is a formatted string such as `"91.2%"`.

### `synthesize_hakka(romanization)`

Uses `facebook/mms-tts-hak` to synthesize speech from Hakka romanization.

Returns `(sample_rate, waveform)` when successful, otherwise `None`.

## Interpretation API

Module: `hakka_speech_toolkit.interpretation`

### `MedicalPhrase`

Frozen dataclass describing one phrase in the clinical phrase bank.

Fields:

* `intent`: stable machine-readable intent name.
* `category`: clinical or workflow category.
* `urgency`: usually `routine`, `warning`, or `urgent`.
* `mandarin`: Mandarin phrase.
* `hakka`: Hakka phrase.
* `hakka_romanization`: romanized Hakka for speech synthesis.
* `mandarin_keywords`: keywords used for Mandarin input matching.
* `hakka_keywords`: keywords used for Hakka input matching.
* `response_hint`: short clinical follow-up note.

### `interpret_text(text, direction, max_results=3) -> dict`

Matches clinical text against `MEDICAL_PHRASES`.

Parameters:

* `text`: Mandarin or Hakka input. `None` and blank text are accepted.
* `direction`: `mandarin_to_hakka` or `hakka_to_mandarin`.
* `max_results`: maximum number of ranked matches to return.

Returns:

```python
{
    "input_text": str,
    "direction": str,
    "matches": list[dict],
    "best_match": dict | None,
    "summary": str,
}
```

Each match contains:

```python
{
    "intent": str,
    "category": str,
    "urgency": str,
    "score": float,
    "matched_keywords": list[str],
    "mandarin": str,
    "hakka": str,
    "hakka_romanization": str,
    "response_hint": str,
}
```

### `normalize_text(text) -> str`

Normalizes punctuation, spaces, selected traditional variants, and a small set of pinyin hints for matching.

### `keyword_score(text, keywords) -> tuple[float, list[str]]`

Scores exact and fuzzy keyword evidence. Returns numeric score and matched keywords.

### `build_summary(text, direction, matches) -> str`

Builds the readable app summary from ranked matches.

### `format_matches_table(matches) -> list[list[object]]`

Converts match dictionaries into rows for a Gradio dataframe.

Row order:

```python
[score, urgency, category, intent, mandarin, hakka, matched_keywords]
```

## Parser API

Module: `hakka_speech_toolkit.parser`

### `HakkaRuleParser(user_defined_dict=None, max_oov_group=4)`

Offline rule-based Hakka parser for small data and domain dictionaries.

Example:

```python
from hakka_speech_toolkit import HakkaRuleParser

parser = HakkaRuleParser()
result = parser.parse("𠊎毋食藥。")
```

### `HakkaRuleParser.from_user_defined_file(path)`

Creates a parser with an Articut-style user dictionary JSON file.

### `HakkaRuleParser.parse(input_text, level="lv2", userDefinedDictFILE=None, user_defined_dict=None) -> dict`

Parses Hakka text with layered lexicon matching, context heuristics, POS-shift repairs, X-bar phrase analysis, and dependency heuristics.

Common return fields:

* `status`: `True` on successful offline parse.
* `input`: original text.
* `level`: requested parse level.
* `result_segmentation`: slash-delimited segmentation string.
* `result_pos`: Articut-like POS XML fragments.
* `result_obj`: nested token object list.
* `tokens`: flat token objects with `text`, `pos`, `start`, `end`, and `source`.
* `sentence_patterns`: lightweight predicate spans.
* `xbar_tree`: experimental X-bar-style phrase projection.
* `dependencies`: UD-compatible dependency baseline.
* `grammar_rules_applied`: grammar-layer rule names.
* `rules_applied`: context and regex POS repair rules.
* `parser`: parser name.
* `parser_strategy`: implementation strategy list.

### `ParseToken`

Dataclass used internally and returned through token dictionaries.

Methods:

* `to_obj()`: returns a JSON-ready token dictionary.
* `to_pos_xml()`: renders Articut-style POS XML for the token.

### `build_user_defined_dict(rows, output_path=None) -> dict`

Builds an Articut-style dictionary from `(word, pos)` rows.

```python
from hakka_speech_toolkit.parser import build_user_defined_dict

terms = build_user_defined_dict(
    [("血氧", "ENTITY_noun"), ("量", "ACTION_verb")],
    "hakka_terms.json",
)
```

### `load_user_defined_dict(path) -> dict`

Loads a JSON user dictionary.

### `normalize_user_defined_dict(user_defined_dict) -> dict`

Accepts either POS-to-word lists or word-to-POS mappings and normalizes them.

Accepted shapes include:

```python
{"ENTITY_noun": ["血氧", "血糖"]}
{"血氧": "ENTITY_noun"}
{"血氧": {"pos": "ENTITY_noun"}}
```

### `ArticutLikeHakkaParser`

Compatibility alias for `HakkaRuleParser`.

## Articut Adapter API

Module: `hakka_speech_toolkit.articut_adapter`

### `HakkaParser(backend="auto", user_defined_dict=None, **articut_kwargs)`

Unified parser facade.

Backends:

* `auto`: try Droidtown ArticutAPI_Hakka, then fall back to the offline parser.
* `articut`: require Droidtown ArticutAPI_Hakka.
* `offline`: use `HakkaRuleParser`.

Example:

```python
from hakka_speech_toolkit import HakkaParser

parser = HakkaParser(backend="offline")
result = parser.parse("𠊎毋食藥。")
```

### `HakkaParser.parse(input_text, **kwargs) -> dict`

Delegates to the selected backend.

### `ArticutHakkaParser(...)`

Lower-level adapter around Droidtown `ArticutAPI_Hakka`.

Constructor parameters:

* `username`, `apikey`, `usernameENG`, `apikeyENG`: credentials for the external package.
* `articut_client`: optional injected client for tests.
* `allow_fallback`: whether to use `HakkaRuleParser` when Articut is unavailable.
* `fallback_parser`: optional injected offline parser.

### `ArticutHakkaParser.parse(...) -> dict`

Passes text to ArticutAPI_Hakka when available. If unavailable and fallback is allowed, returns an offline parser result with:

* `parser_backend`: `offline_fallback`.
* `articut_unavailable_reason`: reason the external backend was not used.

### `load_articut_hakka_class()`

Imports Droidtown `ArticutAPI_Hakka` lazily. Raises `ArticutHakkaUnavailable` if unavailable.

## Speech Model API

Module: `hakka_speech_toolkit.models`

### `HakkaSpeechModel(device=None)`

Base class for device and dtype selection.

Device behavior:

* Uses CUDA when available.
* Uses `float16` on CUDA.
* Uses `float32` on CPU.

### `HakkaSTT(model_id="formospeech/whisper-large-v3-taiwanese-hakka", dialect="htia_sixian", device=None, **pipeline_kwargs)`

Loads a Hakka Whisper ASR model.

### `HakkaSTT.transcribe(audio_path, dialect=None, chunk_length_s=30, batch_size=8) -> str`

Transcribes an audio file.

Raises:

* `FileNotFoundError` for missing audio.
* `HakkaSpeechError` for model loading, inference, or CUDA out-of-memory failures.

### `HakkaTTS(model_id="formospeech/yourtts-htia-240704", device=None, **pipeline_kwargs)`

Loads a Hakka text-to-speech pipeline.

### `HakkaTTS.synthesize(text, output_path) -> Path`

Synthesizes speech to a waveform file.

Raises:

* `ValueError` when text is blank.
* `HakkaSpeechError` for model or synthesis failures.

### `HakkaSpeechError`

Runtime error raised for speech model failures.

## Accent Evaluation API

Module: `hakka_speech_toolkit.accent`

### `AccentEvaluator(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0, max_formant=5500.0)`

Configures Praat/parselmouth pitch and formant extraction.

### `AccentEvaluator.extract_features(audio_path) -> dict`

Returns acoustic metadata:

* `audio_path`
* `duration_seconds`
* `pitch`
* `formants`
* `mfcc`
* `hht`
* `calculation_references`

### `AccentEvaluator.compute_acoustic_distance(reference_audio, target_audio) -> dict`

Computes DTW-based distance between normalized F0 contours.

Common return fields:

* `reference_audio`
* `target_audio`
* `metric`
* `distance`
* `normalized_distance`
* `path_length`
* `normalization`
* `dtw_radius`
* `shape_distance`
* `reference_features`
* `target_features`
* `calculation_references`

### `AccentEvaluator.normalize_f0(f0, method="semitone") -> numpy.ndarray`

Normalizes pitch contours. Supported methods include semitone, z-score, and log-z-score.

### `AccentEvaluator.summarize_tone_shape(f0) -> dict`

Summarizes start, end, range, slope, and turning points for a pitch contour.

### `AccentEvaluator.extract_mfcc_features(waveform, sample_rate, n_mfcc=13) -> dict`

Returns MFCC, delta, and delta-delta summaries with `librosa`, or an unavailable result with a reason.

### `AccentEvaluator.extract_hht_features(waveform, sample_rate, max_imfs=8) -> dict`

Uses `HHSA-Py` when installed. Otherwise returns a Hilbert fallback summary and includes the backend reason.

## Accent Conversion API

Module: `hakka_speech_toolkit.conversion`

### `AccentConversionDecoder`

Protocol for a downstream voice-conversion decoder.

Required method:

```python
generate(content_tokens, target_speaker_embedding, attention_mask=None) -> torch.Tensor
```

The tensor should be shaped like `[channels, samples]` or `[batch, channels, samples]`.

### `HakkaAccentConverter(hubert_model_id="facebook/hubert-base-ls960", conversion_decoder=None, device=None)`

Loads HuBERT for content embedding extraction.

### `HakkaAccentConverter.extract_content_tokens(source_audio_path) -> dict`

Returns:

* `audio_path`
* `sample_rate`: always `16000` after resampling.
* `content_tokens`: CPU `torch.Tensor`.
* `shape`: tensor shape tuple.

### `HakkaAccentConverter.convert_accent(source_audio_path, target_speaker_embedding, output_path, output_sample_rate=16000) -> Path`

Extracts content tokens and calls the supplied decoder to synthesize converted speech.

Raises `HakkaSpeechError` if no decoder is supplied.

## Package Imports

The package root lazily exposes:

```python
from hakka_speech_toolkit import (
    AccentEvaluator,
    HakkaAccentConverter,
    HakkaSpeechModel,
    HakkaSTT,
    HakkaTTS,
    HakkaRuleParser,
    ArticutLikeHakkaParser,
    build_user_defined_dict,
    ArticutHakkaParser,
    HakkaParser,
)
```

Optional model-heavy dependencies are imported only when the corresponding class is requested or instantiated.
