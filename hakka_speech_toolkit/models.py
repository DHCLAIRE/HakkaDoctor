"""GPU-aware Hakka ASR and TTS model wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class HakkaSpeechError(RuntimeError):
    """Raised when a speech model cannot complete an operation."""


class HakkaSpeechModel:
    """Base class that centralizes device and precision selection."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        """Select CUDA when available unless the caller provides a device."""

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    @staticmethod
    def _require_file(audio_path: str | Path) -> Path:
        """Validate and return an existing audio file path."""

        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file, got: {path}")
        return path

    @staticmethod
    def _handle_oom(error: RuntimeError) -> None:
        """Convert CUDA out-of-memory errors into a clearer toolkit exception."""

        if "out of memory" in str(error).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HakkaSpeechError(
                "CUDA ran out of memory. Try a smaller batch, shorter audio, or CPU mode."
            ) from error
        raise error


class HakkaSTT(HakkaSpeechModel):
    """Speech-to-text wrapper for the Taiwanese Hakka Whisper checkpoint."""

    def __init__(
        self,
        model_id: str = "formospeech/whisper-large-v3-taiwanese-hakka",
        dialect: str = "htia_sixian",
        device: str | torch.device | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        """Load the Hakka Whisper ASR model and processor."""

        super().__init__(device=device)

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self.device)
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=0 if self.device.type == "cuda" else -1,
                **pipeline_kwargs,
            )
        except RuntimeError as error:
            self._handle_oom(error)
        except Exception as error:
            raise HakkaSpeechError(f"Failed to load STT model '{model_id}': {error}") from error

        self.model_id = model_id
        self.dialect = dialect

    def transcribe(
        self,
        audio_path: str | Path,
        dialect: str | None = None,
        chunk_length_s: int = 30,
        batch_size: int = 8,
    ) -> str:
        """Transcribe a Hakka audio file."""

        path = self._require_file(audio_path)
        prompt = dialect or self.dialect
        generate_kwargs = {"task": "transcribe"}
        if prompt and hasattr(self.processor, "get_prompt_ids"):
            generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(prompt)

        try:
            result = self.pipeline(
                str(path),
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
            )
        except RuntimeError as error:
            self._handle_oom(error)
        except Exception as error:
            raise HakkaSpeechError(f"Failed to transcribe '{path}': {error}") from error

        return str(result.get("text", "")).strip()


class HakkaTTS(HakkaSpeechModel):
    """Text-to-speech wrapper for the FormoSpeech Hakka YourTTS checkpoint."""

    def __init__(
        self,
        model_id: str = "formospeech/yourtts-htia-240704",
        device: str | torch.device | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        """Load the Hakka text-to-speech pipeline."""

        super().__init__(device=device)

        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text-to-speech",
                model=model_id,
                torch_dtype=self.torch_dtype,
                device=0 if self.device.type == "cuda" else -1,
                **pipeline_kwargs,
            )
        except RuntimeError as error:
            self._handle_oom(error)
        except Exception as error:
            raise HakkaSpeechError(f"Failed to load TTS model '{model_id}': {error}") from error

        self.model_id = model_id

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        """Synthesize speech and save it as a waveform file."""

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = self.pipeline(text)
            audio = result["audio"]
            sampling_rate = int(result["sampling_rate"])

            import numpy as np
            import soundfile as sf

            waveform = np.asarray(audio)
            if waveform.ndim > 1 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            sf.write(output, waveform, sampling_rate)
        except RuntimeError as error:
            self._handle_oom(error)
        except Exception as error:
            raise HakkaSpeechError(f"Failed to synthesize speech to '{output}': {error}") from error

        return output
