# Hakka Doctor Tutorials

This guide is for two groups:

* Clinicians, researchers, and demo users who want to run the Hakka Doctor app.
* Developers who want to reuse or extend the Python toolkit.

The examples assume you are working from the repository root.

## 1. Install and Run the App

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build that matches your CUDA version before installing the rest of the requirements.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Start Hakka Doctor

```bash
python HakkaDoctor_app.py
```

Gradio will print a local URL, usually `http://127.0.0.1:7860`. Open that URL in a browser.

## 2. Use the Medical Interpreter

The app has two tabs.

### Doctor to Patient

Use this tab when a clinician starts in Mandarin and needs Hakka patient-facing wording.

1. Open **Doctor to Patient**.
2. Type a short Mandarin instruction, for example:

   ```text
   請按時吃藥，並且多喝水。
   ```

3. Click **Interpret text**.
4. Review the Hakka output, romanization, candidate matches, and clinical note.

If the TTS model can load on your machine, the app also returns spoken Hakka audio from the romanization field.

### Patient to Doctor

Use this tab when a patient gives Hakka symptoms and the clinician needs Mandarin clinical meaning.

1. Open **Patient to Doctor**.
2. Type Hakka text, for example:

   ```text
   𠊎胸坎痛，透氣毋順。
   ```

3. Click **Interpret text**.
4. Check the Mandarin clinical meaning, urgency, candidate matches, and escalation note.

Urgent examples such as chest pain are marked with `urgent` in the candidate table and summary.

## 3. Use Audio

Both tabs accept microphone or uploaded audio.

### Mandarin doctor audio

1. Record or upload Mandarin audio in **Doctor to Patient**.
2. Click **Transcribe and interpret audio**.
3. The app transcribes Mandarin with Whisper, then runs the same Mandarin-to-Hakka matcher.

### Hakka patient audio

1. Record or upload Hakka audio in **Patient to Doctor**.
2. Click **Transcribe and interpret audio**.
3. The app transcribes Hakka with the FormoSpeech Hakka Whisper model, reports approximate ASR confidence, then runs Hakka-to-Mandarin matching.

Audio workflows load large models lazily. The first run can be slow because the model may need to download and initialize.

## 4. Code Tutorial: Interpret Text in Python

The interpretation layer is intentionally lightweight. It does not require ASR, TTS, CUDA, or Hugging Face models.

```python
from hakka_speech_toolkit.interpretation import interpret_text

doctor_result = interpret_text("請按時吃藥", direction="mandarin_to_hakka")
print(doctor_result["summary"])
print(doctor_result["best_match"]["hakka"])
print(doctor_result["best_match"]["hakka_romanization"])

patient_result = interpret_text("𠊎胸坎痛，透氣毋順", direction="hakka_to_mandarin")
print(patient_result["best_match"]["urgency"])
print(patient_result["best_match"]["mandarin"])
```

Useful result keys:

* `input_text`: original text.
* `direction`: `mandarin_to_hakka` or `hakka_to_mandarin`.
* `matches`: ranked phrase matches.
* `best_match`: first match, or `None`.
* `summary`: readable explanation for app display.

## 5. Code Tutorial: Add Medical Phrases

Medical phrase matching lives in `hakka_speech_toolkit/interpretation.py`.

To add a phrase:

1. Open `MEDICAL_PHRASES`.
2. Add a new `MedicalPhrase`.
3. Include Mandarin keywords and Hakka keywords that users are likely to type or that ASR may output.
4. Add a focused test in `tests/test_interpretation.py`.

Example phrase shape:

```python
MedicalPhrase(
    intent="fever",
    category="Symptom",
    urgency="warning",
    mandarin="您有發燒嗎？",
    hakka="你有發燒無？",
    hakka_romanization="ngi rhiu fad seu mo",
    mandarin_keywords=("發燒", "燒", "體溫"),
    hakka_keywords=("發燒", "燒", "fad seu"),
    response_hint="Ask temperature, duration, chills, and medication use.",
)
```

Then run:

```bash
pytest tests/test_interpretation.py
```

## 6. Code Tutorial: Use the Offline Hakka Parser

The offline parser is useful for small datasets, annotation workflows, and domain vocabulary experiments.

```python
from hakka_speech_toolkit import HakkaRuleParser

parser = HakkaRuleParser()
result = parser.parse("𠊎毋食藥。")

print(result["result_segmentation"])
print(result["result_pos"])
print(result["sentence_patterns"])
print(result["dependencies"])
```

The parser returns Articut-like fields such as `result_segmentation`, `result_pos`, and `result_obj`, plus experimental grammar fields such as `xbar_tree` and `dependencies`.

## 7. Code Tutorial: Add Domain Vocabulary

For a small clinical dataset, start by adding known words and POS labels through a user-defined dictionary.

```python
from hakka_speech_toolkit.parser import build_user_defined_dict

build_user_defined_dict(
    [
        ("血氧", "ENTITY_noun"),
        ("量", "ACTION_verb"),
        ("血氧機", "ENTITY_noun"),
    ],
    "hakka_clinic_terms.json",
)
```

Use the dictionary at parse time:

```python
from hakka_speech_toolkit import HakkaRuleParser

parser = HakkaRuleParser.from_user_defined_file("hakka_clinic_terms.json")
result = parser.parse("請量血氧。")
print(result["result_segmentation"])
```

You can also pass a dictionary without writing a file:

```python
parser = HakkaRuleParser(
    user_defined_dict={
        "ENTITY_noun": ["血氧", "血氧機"],
        "ACTION_verb": ["量"],
    }
)
```

## 8. Code Tutorial: Use Speech Models

Speech models are heavier than text interpretation and parser code. They require `torch`, `transformers`, and enough memory for the selected model.

### Hakka speech-to-text

```python
from hakka_speech_toolkit import HakkaSTT

stt = HakkaSTT(dialect="htia_sixian")
text = stt.transcribe("patient_audio.wav", chunk_length_s=30, batch_size=4)
print(text)
```

### Hakka text-to-speech

```python
from hakka_speech_toolkit import HakkaTTS

tts = HakkaTTS()
path = tts.synthesize("ngi ho, ngai oi mun shin ti.", "outputs/hakka.wav")
print(path)
```

If CUDA runs out of memory, try a smaller `batch_size`, shorter clips, or CPU mode:

```python
stt = HakkaSTT(device="cpu")
```

## 9. Code Tutorial: Accent Evaluation

Use `AccentEvaluator` to compare pitch contours and inspect acoustic summaries.

```python
from hakka_speech_toolkit import AccentEvaluator

evaluator = AccentEvaluator()
features = evaluator.extract_features("speaker.wav")
print(features["pitch"])
print(features["formants"])

distance = evaluator.compute_acoustic_distance("reference.wav", "learner.wav")
print(distance["normalized_distance"])
```

The feature output includes pitch, formant, MFCC, HHT or Hilbert fallback summaries, and calculation references.

## 10. Code Tutorial: Accent Conversion Experiments

`HakkaAccentConverter` extracts HuBERT content embeddings. It does not synthesize converted speech unless you provide a compatible decoder.

```python
from hakka_speech_toolkit import HakkaAccentConverter

converter = HakkaAccentConverter()
tokens = converter.extract_content_tokens("learner_source.wav")
print(tokens["shape"])
```

To convert speech, pass a decoder object with this method:

```python
class MyDecoder:
    def generate(self, content_tokens, target_speaker_embedding, attention_mask=None):
        ...
```

Then:

```python
converter = HakkaAccentConverter(conversion_decoder=MyDecoder())
converter.convert_accent(
    "learner_source.wav",
    target_speaker_embedding,
    "outputs/converted.wav",
)
```

## 11. Developer Workflow

Run all tests:

```bash
pytest
```

Run parser-only tests:

```bash
pytest tests/test_parser.py tests/test_articut_adapter.py
```

Run interpretation tests:

```bash
pytest tests/test_interpretation.py
```

When changing clinical matching, add tests for:

* A positive Mandarin-to-Hakka phrase.
* A positive Hakka-to-Mandarin phrase.
* Unknown input returning no match.
* Urgency behavior for safety-sensitive phrases.

