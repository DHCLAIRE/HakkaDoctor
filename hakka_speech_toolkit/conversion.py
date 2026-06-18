"""HuBERT-based content extraction for Hakka accent conversion pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch
import torchaudio

from .models import HakkaSpeechError, HakkaSpeechModel


class AccentConversionDecoder(Protocol):
    """Protocol for a downstream Soft-VC/VITS-style waveform decoder."""

    def generate(
        self,
        content_tokens: torch.Tensor,
        target_speaker_embedding: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate waveform with shape [channels, samples] or [batch, channels, samples]."""


class HakkaAccentConverter(HakkaSpeechModel):
    """Extract HuBERT content tokens and pass them to a VC decoder."""

    def __init__(
        self,
        hubert_model_id: str = "facebook/hubert-base-ls960",
        conversion_decoder: AccentConversionDecoder | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(device=device)

        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_id)
            self.hubert = HubertModel.from_pretrained(
                hubert_model_id,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.hubert.eval()
        except RuntimeError as error:
            self._handle_oom(error)
        except Exception as error:
            raise HakkaSpeechError(f"Failed to load HuBERT model '{hubert_model_id}': {error}") from error

        self.hubert_model_id = hubert_model_id
        self.conversion_decoder = conversion_decoder

    def convert_accent(
        self,
        source_audio_path: str | Path,
        target_speaker_embedding: torch.Tensor,
        output_path: str | Path,
        output_sample_rate: int = 16000,
    ) -> Path:
        """Generate converted speech with a provided Soft-VC/VITS decoder."""

        if self.conversion_decoder is None:
            raise HakkaSpeechError(
                "Accent conversion requires a downstream Soft-VC/VITS decoder. "
                "HuBERT supplies content embeddings but does not synthesize waveforms by itself."
            )

        source_path = self._require_file(source_audio_path)
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            waveform, sample_rate = torchaudio.load(source_path)
            # waveform: [channels, samples]. HuBERT expects mono 16 kHz audio.
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

            # input_values: [batch=1, samples]. The feature extractor handles padding/masks.
            inputs = self.feature_extractor(
                waveform.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device, dtype=self.torch_dtype)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.inference_mode():
                # last_hidden_state: [batch=1, frames, hidden_size]. Frames are compressed in time
                # relative to waveform samples and act as continuous phonetic content tokens.
                content_tokens = self.hubert(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).last_hidden_state

                speaker_embedding = target_speaker_embedding.to(self.device, dtype=self.torch_dtype)
                if speaker_embedding.ndim == 1:
                    # speaker_embedding: [speaker_dim] -> [batch=1, speaker_dim].
                    speaker_embedding = speaker_embedding.unsqueeze(0)

                # converted: [channels, samples] or [batch, channels, samples], depending on decoder.
                converted = self.conversion_decoder.generate(
                    content_tokens=content_tokens,
                    target_speaker_embedding=speaker_embedding,
                    attention_mask=attention_mask,
                )

            converted = converted.detach().float().cpu()
            if converted.ndim == 3:
                converted = converted.squeeze(0)
            torchaudio.save(output, converted, output_sample_rate)
        except RuntimeError as error:
            self._handle_oom(error)
        except Exception as error:
            raise HakkaSpeechError(f"Failed to convert accent for '{source_path}': {error}") from error

        return output

    def extract_content_tokens(self, source_audio_path: str | Path) -> dict[str, Any]:
        """Expose HuBERT embeddings for experimentation and decoder training."""

        source_path = self._require_file(source_audio_path)
        waveform, sample_rate = torchaudio.load(source_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device, dtype=self.torch_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            content_tokens = self.hubert(
                input_values=input_values,
                attention_mask=attention_mask,
            ).last_hidden_state

        return {
            "audio_path": str(source_path),
            "sample_rate": 16000,
            "content_tokens": content_tokens.detach().float().cpu(),
            "shape": tuple(content_tokens.shape),
        }
