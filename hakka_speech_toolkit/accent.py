"""Acoustic accent evaluation with Praat/parselmouth and DTW.

The calculations follow common pronunciation-assessment and tonal-phonetics
practice: Praat-style F0/formant extraction, unvoiced-frame removal, relative
pitch normalization, contour-shape summaries, and dynamic time warping for
utterances that differ in speaking rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


ACCENT_TOOL_REFERENCES = [
    {
        "key": "boersma_weenink_praat",
        "citation": "Boersma, P., & Weenink, D. Praat: doing phonetics by computer.",
        "supports": "Pitch and formant extraction from speech waveforms.",
        "url": "https://www.praat.org/",
    },
    {
        "key": "jadoul_2023_parselmouth",
        "citation": (
            "Jadoul, Y., de Boer, B., & Ravignani, A. (2023). Parselmouth for "
            "bioacoustics: automated acoustic analysis in Python. Bioacoustics."
        ),
        "supports": "Python automation around Praat acoustic analysis.",
        "doi": "10.1080/09524622.2023.2259327",
    },
    {
        "key": "sakoe_chiba_1978_dtw",
        "citation": (
            "Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm "
            "optimization for spoken word recognition. IEEE TASSP, 26(1), 43-49."
        ),
        "supports": "Dynamic time warping for rate-normalized contour comparison.",
        "doi": "10.1109/TASSP.1978.1163055",
    },
    {
        "key": "kheir_2023_pronunciation_review",
        "citation": (
            "El Kheir, Y., Ali, A., & Chowdhury, S. A. (2023). Automatic "
            "Pronunciation Assessment - A Review. arXiv:2310.13974."
        ),
        "supports": "Combining segmental/prosodic acoustic measures for pronunciation assessment.",
        "doi": "10.48550/arXiv.2310.13974",
    },
]


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
        return self._clean_f0(f0)

    def extract_features(self, audio_path: str | Path) -> dict[str, Any]:
        """Return mean F0, F1, F2, and contour metadata in JSON-ready form."""

        sound = self._load_sound(audio_path)
        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        f0 = pitch.selected_array["frequency"].astype(float)
        voiced_f0 = self._clean_f0(f0)

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
                "median_f0_hz": float(np.median(voiced_f0)) if voiced_f0.size else None,
                "f0_range_hz": float(np.ptp(voiced_f0)) if voiced_f0.size else None,
                "voiced_frame_count": int(voiced_f0.size),
                "unvoiced_frame_count": int(f0.size - voiced_f0.size),
                "tone_shape": self.summarize_tone_shape(voiced_f0),
            },
            "formants": {
                "mean_f1_hz": float(np.mean(f1_values)) if f1_values else None,
                "mean_f2_hz": float(np.mean(f2_values)) if f2_values else None,
            },
            "calculation_references": ACCENT_TOOL_REFERENCES,
        }

    def compute_acoustic_distance(
        self,
        reference_audio: str | Path,
        target_audio: str | Path,
        normalization: str = "semitone",
        radius: int | None = None,
    ) -> dict[str, Any]:
        """Compare two normalized F0 contours with dynamic time warping."""

        reference_f0 = self._pitch_contour(reference_audio)
        target_f0 = self._pitch_contour(target_audio)
        if reference_f0.size < 2 or target_f0.size < 2:
            raise ValueError("Both audio files need at least two voiced pitch frames.")

        reference_norm = self.normalize_f0(reference_f0, method=normalization)
        target_norm = self.normalize_f0(target_f0, method=normalization)
        distance, path_length = self._dtw_distance(reference_norm, target_norm, radius=radius)
        reference_shape = self.summarize_tone_shape(reference_f0)
        target_shape = self.summarize_tone_shape(target_f0)

        return {
            "reference_audio": str(Path(reference_audio).expanduser()),
            "target_audio": str(Path(target_audio).expanduser()),
            "metric": f"dtw_{normalization}_f0",
            "distance": float(distance),
            "normalized_distance": float(distance / path_length),
            "path_length": int(path_length),
            "normalization": normalization,
            "dtw_radius": radius,
            "shape_distance": self._shape_distance(reference_shape, target_shape),
            "reference_features": self.extract_features(reference_audio),
            "target_features": self.extract_features(target_audio),
            "calculation_references": ACCENT_TOOL_REFERENCES,
        }

    @staticmethod
    def _clean_f0(f0: np.ndarray) -> np.ndarray:
        """Remove unvoiced, non-finite, and non-positive pitch frames."""

        values = np.asarray(f0, dtype=float)
        return values[np.isfinite(values) & (values > 0.0)]

    @classmethod
    def normalize_f0(cls, f0: np.ndarray, method: str = "semitone") -> np.ndarray:
        """Normalize F0 for tone-contour comparison.

        ``semitone`` maps Hz to semitones relative to the speaker's median F0,
        emphasizing tone shape and register while reducing speaker sex/age
        differences. ``zscore`` preserves the previous implementation's
        standard-score contour comparison.
        """

        cleaned = cls._clean_f0(f0)
        if cleaned.size == 0:
            return cleaned

        if method == "semitone":
            median = float(np.median(cleaned))
            if median <= 0.0:
                return np.zeros_like(cleaned)
            return 12.0 * np.log2(cleaned / median)
        if method == "zscore":
            std = float(np.std(cleaned))
            if std == 0.0:
                return np.zeros_like(cleaned)
            return (cleaned - float(np.mean(cleaned))) / std
        if method == "log_zscore":
            logged = np.log(cleaned)
            std = float(np.std(logged))
            if std == 0.0:
                return np.zeros_like(logged)
            return (logged - float(np.mean(logged))) / std

        raise ValueError("normalization must be one of: semitone, zscore, log_zscore")

    @classmethod
    def summarize_tone_shape(cls, f0: np.ndarray) -> dict[str, float | int | None]:
        """Summarize pitch register and contour shape for tonal accent feedback."""

        cleaned = cls._clean_f0(f0)
        if cleaned.size < 2:
            return {
                "start_st": None,
                "end_st": None,
                "range_st": None,
                "slope_st_per_frame": None,
                "turning_point_count": 0,
            }

        semitone = cls.normalize_f0(cleaned, method="semitone")
        slope = float((semitone[-1] - semitone[0]) / (semitone.size - 1))
        deltas = np.diff(semitone)
        signs = np.sign(deltas[np.abs(deltas) > 1e-6])
        turning_points = int(np.count_nonzero(np.diff(signs) != 0)) if signs.size > 1 else 0
        return {
            "start_st": float(semitone[0]),
            "end_st": float(semitone[-1]),
            "range_st": float(np.ptp(semitone)),
            "slope_st_per_frame": slope,
            "turning_point_count": turning_points,
        }

    @staticmethod
    def _shape_distance(
        reference_shape: dict[str, float | int | None],
        target_shape: dict[str, float | int | None],
    ) -> dict[str, float]:
        comparable_keys = ("start_st", "end_st", "range_st", "slope_st_per_frame")
        distances = {}
        for key in comparable_keys:
            reference_value = reference_shape.get(key)
            target_value = target_shape.get(key)
            if reference_value is not None and target_value is not None:
                distances[key] = float(abs(float(reference_value) - float(target_value)))
        distances["turning_point_count"] = float(
            abs(
                int(reference_shape.get("turning_point_count") or 0)
                - int(target_shape.get("turning_point_count") or 0)
            )
        )
        return distances

    @staticmethod
    def _dtw_distance(
        reference: np.ndarray,
        target: np.ndarray,
        radius: int | None = None,
    ) -> tuple[float, int]:
        costs = np.abs(reference[:, None] - target[None, :])
        rows, cols = costs.shape
        accumulated = np.full((rows + 1, cols + 1), np.inf)
        lengths = np.zeros((rows + 1, cols + 1), dtype=int)
        accumulated[0, 0] = 0.0

        for row in range(1, rows + 1):
            start_col = 1
            end_col = cols
            if radius is not None:
                scale = cols / rows
                center = int(round(row * scale))
                start_col = max(1, center - radius)
                end_col = min(cols, center + radius)

            for col in range(start_col, end_col + 1):
                candidates = (
                    (accumulated[row - 1, col], lengths[row - 1, col]),
                    (accumulated[row, col - 1], lengths[row, col - 1]),
                    (accumulated[row - 1, col - 1], lengths[row - 1, col - 1]),
                )
                previous_cost, previous_length = min(candidates, key=lambda item: item[0])
                accumulated[row, col] = costs[row - 1, col - 1] + previous_cost
                lengths[row, col] = previous_length + 1

        if not np.isfinite(accumulated[rows, cols]):
            raise ValueError("No valid DTW path found. Increase the DTW radius.")

        return float(accumulated[rows, cols]), int(lengths[rows, cols])
