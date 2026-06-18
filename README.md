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

