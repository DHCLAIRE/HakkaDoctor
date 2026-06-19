# Hakka Doctor 客醫講 (Ongoing)

旨在降低使用客家話於醫療溝通上的門檻，並增進偏遠地區醫療資源的可取得性。

A designed platform to lower the barrier for people to communicate in Hakka about their health conditions. And to improve the resource accessibility in remote area. 


[<img width="548" height="396" alt="客醫講_pic2_翻譯現場" src="https://github.com/user-attachments/assets/fc4325cd-e1c0-4e66-90ba-d9a27c536d60" />](https://github.com/DHCLAIRE/HakkaDoctor/blob/main/Pics/%E5%AE%A2%E9%86%AB%E8%AC%9B_pic2_%E7%BF%BB%E8%AD%AF%E7%8F%BE%E5%A0%B4.png)

### 技術架構

![image](https://github.com/DHCLAIRE/HakkaDoctor/blob/main/Pics/%E5%AE%A2%E9%86%AB%E8%AC%9B_pic3_%E6%8A%80%E8%A1%93%E6%9E%B6%E6%A7%8B.png)

## Hakka Speech & Accent Processing Toolkit

This repository now includes an importable, class-based Python toolkit for Hakka speech evaluation, transcription, synthesis, and accent-conversion research. It is designed for CUDA-enabled environments while still allowing CPU fallback.

### Features

* **Accent Evaluation:** Computes acoustic distance with DTW on normalized F0 contours using `parselmouth` (Praat), NumPy, and SciPy.
* **ASR:** Wraps `formospeech/whisper-large-v3-taiwanese-hakka` for dialect-aware Hakka transcription.
* **TTS:** Wraps `formospeech/yourtts-htia-240704` through the Hugging Face `text-to-speech` pipeline.
* **Accent Conversion:** Uses HuBERT content embeddings as the frontend for a downstream Soft-VC or VITS-style decoder.

### Installation

Ensure you have a CUDA-enabled NVIDIA GPU for high-throughput inference.

```bash
# 1. Clone the repository
git clone https://github.com/DHCLAIRE/HakkaDoctor.git
cd HakkaDoctor

# 2. Install PyTorch with CUDA support, adjusting the CUDA index URL as needed
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install toolkit dependencies
pip install -r requirements.txt
```

### Quick Start

#### Mandarin-Hakka Medical Interpreter

```bash
python HakkaDoctor_app.py
```

The Gradio app supports Mandarin doctor text/audio to Hakka patient instructions and Hakka patient text/audio to Mandarin clinical meaning. Text interpretation works without loading ASR/TTS models; audio and speech synthesis load models lazily when those buttons are used.

##### App Function Reference

* `build_demo()`: Builds the Gradio interface for Mandarin-Hakka text/audio workflows.
* `interpret_doctor_text(text)`: Converts Mandarin doctor text into Hakka text, romanization, candidate matches, and optional spoken audio.
* `interpret_doctor_audio(audio_path)`: Transcribes Mandarin doctor audio before running doctor-to-patient interpretation.
* `interpret_patient_text(text)`: Converts Hakka patient text into Mandarin clinical meaning and candidate matches.
* `interpret_patient_audio(audio_path)`: Transcribes Hakka patient audio before running patient-to-doctor interpretation.
* `transcribe_mandarin(audio_path)`: Runs Mandarin ASR for doctor audio.
* `transcribe_hakka(audio_path)`: Runs Hakka ASR and returns transcription plus confidence.
* `synthesize_hakka(romanization)`: Generates Hakka speech from romanized Hakka when TTS is available.

#### Small-Data Hakka Parser

```python
from hakka_speech_toolkit import HakkaRuleParser

parser = HakkaRuleParser()
result = parser.parse("𠊎毋食藥。")
print(result["result_segmentation"])
print(result["result_pos"])
```

The parser follows the low-resource, ArticutAPI_Hakka-style approach without requiring the paid Mandarin Articut API at runtime. It uses layered POS dictionaries, forward maximum matching, user-defined dictionary injection, token-context heuristics, regex `posShift` repairs over Articut-like POS XML, and OOV heuristics. This repository does not redistribute Droidtown's `HAC_dict` or `IreneHAKKA_dict`; check that project's license before reusing their vocabulary lists directly.

To use Droidtown's actual parser instead of the offline fallback, install `ArticutAPI_Hakka` separately and pass your Articut credentials:

```python
from hakka_speech_toolkit import HakkaParser

parser = HakkaParser(
    backend="articut",
    username="YOUR_EMAIL",
    apikey="YOUR_API_KEY",
)
result = parser.parse("𠊎毋食藥。", userDefinedDictFILE="hakka_rules.json")
```

Use `backend="auto"` to try ArticutAPI_Hakka first and fall back to the local rule parser if the external package is unavailable.

```python
from hakka_speech_toolkit.parser import build_user_defined_dict

build_user_defined_dict(
    [("血氧", "ENTITY_noun"), ("量", "ACTION_verb")],
    "hakka_rules.json",
)

result = parser.parse("請量血氧", userDefinedDictFILE="hakka_rules.json")
```

Returned fields include `result_segmentation`, `result_pos`, `result_obj`, `tokens`, `sentence_patterns`, and `rules_applied`, so the output can be used in downstream clinical interpretation or annotation workflows.

For tiny datasets, start by converting reviewed rows into the user-defined dictionary format, then add systematic grammar repairs to `hakka_speech_toolkit/parser_rules.py`. This mirrors ArticutAPI_Hakka's practical split between curated POS vocabulary and regex correction rules.

The syntax layer adds an X-bar-inspired phrase projection and UD-compatible dependency baseline. Treat this as a grammar-writing workflow, not a trained parser:

1. **Stage 0 - Corpus triage:** record corpus size, register, Hakka variety, script/romanization, and whether the data is segmented.
2. **Stage 1 - Lexicon and segmenter:** build curated POS lexicons for your target variety, using sources such as Taiwan MOE's 客家語常用詞辭典 when appropriate.
3. **Stage 2 - Hand annotation:** annotate 100-300 representative sentences in a UD-compatible format before claiming syntactic coverage.
4. **Stage 3 - Grammar rules:** encode X-bar/dependency rules for verb-argument structure, classifier-noun phrases, negation/aspect scope, possessives, and clause-final particles.

Parser output now includes `xbar_tree`, `dependencies`, `grammar_rules_applied`, `annotation_framework`, and `grammar_references`. The implementation cites Chomsky's X-bar origin, Jackendoff's phrase-structure development, and Universal Dependencies for dependency labels.

Key references:

* Chomsky, N. (1970). Remarks on nominalization. In R. Jacobs & P. Rosenbaum (Eds.), *Readings in English Transformational Grammar*.
* Jackendoff, R. (1977). *X-bar Syntax: A Study of Phrase Structure*. MIT Press.
* de Marneffe, M.-C., Manning, C. D., Nivre, J., & Zeman, D. (2021). Universal Dependencies. *Computational Linguistics*, 47(2), 255-308. https://doi.org/10.1162/coli_a_00402

##### Parser Function Reference

* `HakkaRuleParser.parse(...)`: Runs the offline MaxMatch, POS-shift, X-bar, and dependency parser.
* `HakkaRuleParser.from_user_defined_file(path)`: Creates an offline parser from an Articut-style user dictionary file.
* `HakkaParser.parse(...)`: Runs the selected parser backend: `articut`, `offline`, or `auto`.
* `ArticutHakkaParser.parse(...)`: Delegates to Droidtown `ArticutAPI_Hakka` when installed, with optional offline fallback.
* `build_user_defined_dict(rows, output_path=None)`: Converts small labeled vocabulary rows into an Articut-style dictionary.
* `normalize_user_defined_dict(user_defined_dict)`: Accepts POS-to-word lists or word-to-POS mappings and normalizes them.
* `load_user_defined_dict(path)`: Loads an Articut-style dictionary JSON file.
* `apply_pos_shift_rules(pos_xml)`: Applies regex grammar repair rules to Articut-like POS XML.

##### Speech Toolkit Function Reference

* `HakkaSTT.transcribe(audio_path, dialect=...)`: Transcribes Hakka audio with the configured ASR model.
* `HakkaTTS.synthesize(text, output_path)`: Synthesizes Hakka speech and saves it to a waveform file.
* `AccentEvaluator.extract_features(audio_path)`: Extracts F0/formant summary features and tone-shape metadata.
  It now also returns `mfcc` summaries and `hht` signal-dissection summaries.
* `AccentEvaluator.compute_acoustic_distance(reference_audio, target_audio)`: Computes DTW-based F0 accent distance.
* `AccentEvaluator.normalize_f0(f0, method=...)`: Normalizes F0 contours using semitone, z-score, or log-z-score methods.
* `AccentEvaluator.summarize_tone_shape(f0)`: Summarizes tone contour start, end, range, slope, and turning points.
* `AccentEvaluator.extract_mfcc_features(waveform, sample_rate)`: Extracts 13-coefficient MFCC, delta, and delta-delta summary statistics with `librosa`.
* `AccentEvaluator.extract_hht_features(waveform, sample_rate)`: Extracts HHT/HHSA IMF, Hilbert spectrum, energy, entropy, and dominant carrier summaries. If `HHSA-Py` is not installed, it returns a NumPy FFT Hilbert fallback and marks the backend clearly.
* `HakkaAccentConverter.extract_content_tokens(source_audio_path)`: Extracts HuBERT content embeddings for conversion experiments.
* `HakkaAccentConverter.convert_accent(source_audio_path, target_speaker_embedding, output_path)`: Runs conversion through a supplied Soft-VC/VITS decoder.

For full EMD/HHT signal dissection, install HHSA-Py separately:

```bash
python3 -m pip install git+https://github.com/DHCLAIRE/HHSA-Py.git
```

The HHT feature path follows the accent-classification motivation in Walsh, Dev, and Nag (2022): use Hilbert-Huang features for non-linear, non-stationary speech where Fourier/Mel features can lose temporal-frequency detail. MFCC remains available as the conventional baseline.

#### Accent Evaluation

```python
from hakka_speech_toolkit import AccentEvaluator

evaluator = AccentEvaluator()
result = evaluator.compute_acoustic_distance("native_ref.wav", "learner_target.wav")
print(result["normalized_distance"])
```

#### Speech-to-Text

```python
from hakka_speech_toolkit import HakkaSTT

stt = HakkaSTT(dialect="htia_sixian")
text = stt.transcribe("patient_audio_sixian.wav")
print(text)
```

#### Text-to-Speech

```python
from hakka_speech_toolkit import HakkaTTS

tts = HakkaTTS()
tts.synthesize("ngi ho, ngai oi mun shin ti.", "hakka_output.wav")
```

#### HuBERT Content Tokens for Accent Conversion

```python
from hakka_speech_toolkit import HakkaAccentConverter

converter = HakkaAccentConverter()
tokens = converter.extract_content_tokens("learner_source.wav")
print(tokens["shape"])
```

Actual accent conversion requires a compatible downstream Soft-VC or VITS decoder. HuBERT extracts content embeddings but does not synthesize a converted waveform by itself.

### Hardware Optimization

The model wrappers automatically select `cuda` when available and use `float16` precision on CUDA to reduce VRAM usage. For long recordings, tune `batch_size` and `chunk_length_s` in `HakkaSTT.transcribe()`.

### Accent Tool Calculation References

The accent evaluator reports calculation metadata in `calculation_references` so downstream apps can show why each score exists.

* Boersma, P., & Weenink, D. Praat: doing phonetics by computer. Used for pitch and formant extraction.
* Jadoul, Y., de Boer, B., & Ravignani, A. (2023). Parselmouth for bioacoustics: automated acoustic analysis in Python. *Bioacoustics*. https://doi.org/10.1080/09524622.2023.2259327
* Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49. https://doi.org/10.1109/TASSP.1978.1163055
* El Kheir, Y., Ali, A., & Chowdhury, S. A. (2023). Automatic Pronunciation Assessment - A Review. *arXiv:2310.13974*. https://doi.org/10.48550/arXiv.2310.13974

### Recommended Papers for Tonal Accent Detection

* Jin, X., Ernestus, M., & Baayen, R. H. (2024). A corpus-based investigation of pitch contours of monosyllabic words in conversational Taiwan Mandarin. Useful for modeling contextual tone variation and coarticulation.
* Zhang, S. (2019). Data mining Mandarin tone contour shapes. Useful for representing tones as contour clusters rather than fixed labels.
* Li, B., Xie, J. Y., & Rudzicz, F. (2020). Representation Learning for Discovering Phonemic Tone Contours. Useful for embedding and clustering tone contours in Mandarin and Cantonese.
* Yuan, W., & Black, A. W. (2018). Generating Mandarin and Cantonese F0 Contours with Decision Trees and BLSTMs. Useful for F0 contour representations and tone-dependent modeling.
* Wang, T., Potter, C. E., & Saffran, J. R. (2020). Plasticity in Second Language Learning: The Case of Mandarin Tones. *Language Learning and Development*, 16(3), 231-243. https://doi.org/10.1080/15475441.2020.1737072
* Dong, Y. Difficulties in Perception and Pronunciation of Mandarin Chinese Disyllabic Word Tone Acquisition. Useful for disyllabic tone-combination error patterns.
