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
    {
        "key": "davis_mermelstein_1980_mfcc",
        "citation": (
            "Davis, S., & Mermelstein, P. (1980). Comparison of parametric "
            "representations for monosyllabic word recognition in continuously spoken sentences."
        ),
        "supports": "MFCC baseline features for speech representation.",
        "doi": "10.1109/TASSP.1980.1163420",
    },
    {
        "key": "huang_1998_hht",
        "citation": (
            "Huang, N. E., et al. (1998). The empirical mode decomposition and "
            "the Hilbert spectrum for nonlinear and non-stationary time series analysis."
        ),
        "supports": "HHT/EMD signal dissection into IMFs and Hilbert instantaneous features.",
        "doi": "10.1098/rspa.1998.0193",
    },
    {
        "key": "walsh_2022_hht_accent",
        "citation": (
            "Walsh, D., Dev, S., & Nag, A. (2022). Hilbert-Huang-Transform "
            "Based Features for Accent Classification of Non-Native English Speakers."
        ),
        "supports": "HHT-derived Hilbert Mel-Spectrogram features for accent classification.",
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
        """Configure Praat pitch/formant extraction parameters."""

        self.time_step = time_step
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.max_formant = max_formant

    @staticmethod
    def _require_file(audio_path: str | Path) -> Path:
        """Validate and return an existing audio file path."""

        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file, got: {path}")
        return path

    def _load_sound(self, audio_path: str | Path) -> Any:
        """Load an audio file as a parselmouth Sound object."""

        import parselmouth

        return parselmouth.Sound(str(self._require_file(audio_path)))

    def _pitch_contour(self, audio_path: str | Path) -> np.ndarray:
        """Extract the cleaned voiced F0 contour from an audio file."""

        sound = self._load_sound(audio_path)
        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        f0 = pitch.selected_array["frequency"].astype(float)
        return self._clean_f0(f0)

    def extract_features(self, audio_path: str | Path) -> dict[str, Any]:
        """Return F0, F1/F2, MFCC, and HHT metadata in JSON-ready form."""

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
        waveform, sample_rate = self._sound_to_mono(sound)

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
            "mfcc": self.extract_mfcc_features(waveform, sample_rate),
            "hht": self.extract_hht_features(waveform, sample_rate),
            "calculation_references": ACCENT_TOOL_REFERENCES,
        }

    @staticmethod
    def _sound_to_mono(sound: Any) -> tuple[np.ndarray, float]:
        """Convert a parselmouth Sound into mono samples and sampling rate."""

        values = np.asarray(sound.values, dtype=float)
        if values.ndim == 2:
            waveform = np.mean(values, axis=0)
        else:
            waveform = values.reshape(-1)
        sample_rate = float(getattr(sound, "sampling_frequency", 1.0 / sound.dx))
        return waveform, sample_rate

    @classmethod
    def extract_mfcc_features(
        cls,
        waveform: np.ndarray,
        sample_rate: float,
        n_mfcc: int = 13,
    ) -> dict[str, Any]:
        """Extract MFCC coefficient summaries from a waveform."""

        y = cls._prepare_waveform(waveform)
        if y.size == 0:
            return {"n_mfcc": n_mfcc, "backend": "librosa", "available": False, "reason": "empty waveform"}
        try:
            import librosa

            mfcc = librosa.feature.mfcc(y=y, sr=int(sample_rate), n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
        except Exception as error:
            return {
                "n_mfcc": n_mfcc,
                "backend": "librosa",
                "available": False,
                "reason": str(error),
            }
        return {
            "n_mfcc": n_mfcc,
            "backend": "librosa",
            "available": True,
            "coefficients": cls._summarize_feature_matrix(mfcc, prefix="mfcc"),
            "delta": cls._summarize_feature_matrix(delta, prefix="delta"),
            "delta_delta": cls._summarize_feature_matrix(delta2, prefix="delta_delta"),
        }

    @classmethod
    def extract_hht_features(
        cls,
        waveform: np.ndarray,
        sample_rate: float,
        max_imfs: int = 8,
    ) -> dict[str, Any]:
        """Extract HHT/HHSA signal-dissection summaries from a waveform."""

        y = cls._prepare_waveform(waveform)
        if y.size < 4:
            return {"available": False, "backend": "hhsa_tools", "reason": "waveform too short"}
        try:
            from hhsa_tools import HHSAPipeline

            pipeline = HHSAPipeline(
                sample_rate=float(sample_rate),
                decomposition="sift",
                frequency_method="hybrid",
                max_imfs=max_imfs,
            )
            result = pipeline.fit(y)
            summary = pipeline.summarize(result)
            return cls._summarize_hhsa_result(result, summary)
        except Exception as error:
            fallback = cls._extract_hilbert_fallback_features(y, sample_rate)
            fallback["hhsa_tools_error"] = str(error)
            return fallback

    @staticmethod
    def _prepare_waveform(waveform: np.ndarray) -> np.ndarray:
        """Return a finite, centered mono waveform vector."""

        y = np.asarray(waveform, dtype=float).reshape(-1)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return y
        y = y - float(np.mean(y))
        peak = float(np.max(np.abs(y)))
        if peak > 0.0:
            y = y / peak
        return y

    @staticmethod
    def _summarize_feature_matrix(matrix: np.ndarray, prefix: str) -> dict[str, Any]:
        """Summarize a feature matrix by coefficient-wise mean and standard deviation."""

        values = np.asarray(matrix, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return {
            f"{prefix}_shape": list(values.shape),
            f"{prefix}_mean": [float(v) for v in np.mean(values, axis=1)],
            f"{prefix}_std": [float(v) for v in np.std(values, axis=1)],
        }

    @classmethod
    def _summarize_hhsa_result(cls, result: Any, summary: dict[str, Any]) -> dict[str, Any]:
        """Summarize HHSA-Py outputs for accent-feature JSON."""

        imfs = np.asarray(getattr(result, "imfs", np.empty((0, 0))), dtype=float)
        hht = np.asarray(getattr(result, "hht", np.empty((0, 0))), dtype=float)
        marginal = np.asarray(getattr(result, "marginal", np.empty(0)), dtype=float)
        carrier_bins = np.asarray(getattr(result, "carrier_bins", np.empty(0)), dtype=float)
        mode_energy = np.asarray(summary.get("mode_energy", []), dtype=float)
        dominant_frequency = None
        if marginal.size and carrier_bins.size:
            dominant_frequency = float(carrier_bins[int(np.nanargmax(marginal))])
        return {
            "available": True,
            "backend": "hhsa_tools.HHSAPipeline",
            "imf_count": int(imfs.shape[0]) if imfs.ndim == 2 else 0,
            "mode_energy": [float(v) for v in mode_energy],
            "mode_energy_entropy": cls._normalized_entropy(mode_energy),
            "reconstruction_error": float(summary.get("reconstruction_error", np.nan)),
            "dominant_carrier_hz": dominant_frequency,
            "hht_shape": list(hht.shape),
            "hht_energy": float(np.nansum(hht)) if hht.size else 0.0,
            "marginal_entropy": cls._normalized_entropy(marginal),
        }

    @classmethod
    def _extract_hilbert_fallback_features(cls, waveform: np.ndarray, sample_rate: float) -> dict[str, Any]:
        """Extract instantaneous amplitude/frequency summaries with a NumPy Hilbert fallback."""

        try:
            analytic = cls._analytic_signal(waveform)
            amplitude = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            frequency = np.diff(phase) * float(sample_rate) / (2.0 * np.pi)
            frequency = frequency[np.isfinite(frequency) & (frequency >= 0.0)]
            return {
                "available": True,
                "backend": "numpy.fft.hilbert_fallback",
                "imf_count": 1,
                "mode_energy": [float(np.sum(np.square(waveform)))],
                "mode_energy_entropy": 0.0,
                "instantaneous_amplitude_mean": float(np.mean(amplitude)),
                "instantaneous_amplitude_std": float(np.std(amplitude)),
                "instantaneous_frequency_mean_hz": float(np.mean(frequency)) if frequency.size else None,
                "instantaneous_frequency_std_hz": float(np.std(frequency)) if frequency.size else None,
                "instantaneous_frequency_median_hz": float(np.median(frequency)) if frequency.size else None,
                "note": "Fallback uses one analytic signal and does not perform EMD/IMF decomposition.",
            }
        except Exception as error:
            return {
                "available": False,
                "backend": "numpy.fft.hilbert_fallback",
                "reason": str(error),
            }

    @staticmethod
    def _analytic_signal(waveform: np.ndarray) -> np.ndarray:
        """Compute the analytic signal with the standard FFT Hilbert construction."""

        x = np.asarray(waveform, dtype=float).reshape(-1)
        n = x.size
        spectrum = np.fft.fft(x)
        multiplier = np.zeros(n)
        if n % 2 == 0:
            multiplier[0] = 1.0
            multiplier[n // 2] = 1.0
            multiplier[1 : n // 2] = 2.0
        else:
            multiplier[0] = 1.0
            multiplier[1 : (n + 1) // 2] = 2.0
        return np.fft.ifft(spectrum * multiplier)

    @staticmethod
    def _normalized_entropy(values: np.ndarray) -> float | None:
        """Compute normalized Shannon entropy for non-negative feature vectors."""

        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array) & (array >= 0.0)]
        total = float(np.sum(array))
        if array.size == 0 or total == 0.0:
            return None
        probability = array / total
        entropy = -float(np.sum(probability * np.log(probability + 1e-12)))
        return float(entropy / np.log(array.size)) if array.size > 1 else 0.0

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
        """Compute absolute distances between comparable tone-shape features."""

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
        """Compute DTW distance and path length between two numeric contours."""

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
