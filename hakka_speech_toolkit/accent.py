"""Acoustic accent evaluation with Praat/parselmouth and SciPy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class AccentEvaluator:
    """Extract F0/formant features and compare normalized pitch contours."""

    def __init__(
        self,
        time_step: float = 0.01,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
        max_formant: float = 5500.0,
    ) -> None:
        self.time_step = time_step
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.max_formant = max_formant

    @staticmethod
    def _require_file(audio_path: str | Path) -> Path:
        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file, got: {path}")
        return path

    def _load_sound(self, audio_path: str | Path) -> Any:
        import parselmouth

        return parselmouth.Sound(str(self._require_file(audio_path)))

    def _pitch_contour(self, audio_path: str | Path) -> np.ndarray:
        sound = self._load_sound(audio_path)
        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        f0 = pitch.selected_array["frequency"].astype(float)
        return f0[f0 > 0.0]

    def extract_features(self, audio_path: str | Path) -> dict[str, Any]:
        """Return mean F0, F1, F2, and contour metadata in JSON-ready form."""

        sound = self._load_sound(audio_path)
        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        f0 = pitch.selected_array["frequency"].astype(float)
        voiced_f0 = f0[f0 > 0.0]

        formant = sound.to_formant_burg(
            time_step=self.time_step,
            max_number_of_formants=5,
            maximum_formant=self.max_formant,
        )
        times = np.arange(0.0, sound.duration, self.time_step)
        f1_values = []
        f2_values = []
        for time in times:
            f1 = formant.get_value_at_time(1, float(time))
            f2 = formant.get_value_at_time(2, float(time))
            if np.isfinite(f1):
                f1_values.append(float(f1))
            if np.isfinite(f2):
                f2_values.append(float(f2))

        return {
            "audio_path": str(Path(audio_path).expanduser()),
            "duration_seconds": float(sound.duration),
            "pitch": {
                "mean_f0_hz": float(np.mean(voiced_f0)) if voiced_f0.size else None,
                "voiced_frame_count": int(voiced_f0.size),
                "unvoiced_frame_count": int(f0.size - voiced_f0.size),
            },
            "formants": {
                "mean_f1_hz": float(np.mean(f1_values)) if f1_values else None,
                "mean_f2_hz": float(np.mean(f2_values)) if f2_values else None,
            },
        }

    def compute_acoustic_distance(
        self,
        reference_audio: str | Path,
        target_audio: str | Path,
    ) -> dict[str, Any]:
        """Compare two normalized F0 contours with dynamic time warping."""

        reference_f0 = self._pitch_contour(reference_audio)
        target_f0 = self._pitch_contour(target_audio)
        if reference_f0.size < 2 or target_f0.size < 2:
            raise ValueError("Both audio files need at least two voiced pitch frames.")

        from scipy.stats import zscore

        reference_norm = np.nan_to_num(zscore(reference_f0), nan=0.0)
        target_norm = np.nan_to_num(zscore(target_f0), nan=0.0)
        distance, path_length = self._dtw_distance(reference_norm, target_norm)

        return {
            "reference_audio": str(Path(reference_audio).expanduser()),
            "target_audio": str(Path(target_audio).expanduser()),
            "metric": "dtw_normalized_f0",
            "distance": float(distance),
            "normalized_distance": float(distance / path_length),
            "path_length": int(path_length),
            "reference_features": self.extract_features(reference_audio),
            "target_features": self.extract_features(target_audio),
        }

    @staticmethod
    def _dtw_distance(reference: np.ndarray, target: np.ndarray) -> tuple[float, int]:
        from scipy.spatial.distance import cdist

        costs = cdist(reference[:, None], target[:, None], metric="euclidean")
        rows, cols = costs.shape
        accumulated = np.full((rows + 1, cols + 1), np.inf)
        lengths = np.zeros((rows + 1, cols + 1), dtype=int)
        accumulated[0, 0] = 0.0

        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                candidates = (
                    (accumulated[row - 1, col], lengths[row - 1, col]),
                    (accumulated[row, col - 1], lengths[row, col - 1]),
                    (accumulated[row - 1, col - 1], lengths[row - 1, col - 1]),
                )
                previous_cost, previous_length = min(candidates, key=lambda item: item[0])
                accumulated[row, col] = costs[row - 1, col - 1] + previous_cost
                lengths[row, col] = previous_length + 1

        return float(accumulated[rows, cols]), int(lengths[rows, cols])
