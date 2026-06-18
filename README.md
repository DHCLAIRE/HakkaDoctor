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




