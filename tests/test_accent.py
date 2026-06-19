from __future__ import annotations

import numpy as np
import pytest

from hakka_speech_toolkit.accent import ACCENT_TOOL_REFERENCES, AccentEvaluator


class SyntheticAccentEvaluator(AccentEvaluator):
    def __init__(self, contours: dict[str, np.ndarray]) -> None:
        """Store synthetic F0 contours for parser-free evaluator tests."""

        super().__init__()
        self.contours = contours

    def _pitch_contour(self, audio_path):
        """Return a synthetic cleaned contour for the requested path key."""

        return self._clean_f0(self.contours[str(audio_path)])

    def extract_features(self, audio_path):
        """Return minimal feature output without loading real audio."""

        f0 = self._pitch_contour(audio_path)
        return {
            "audio_path": str(audio_path),
            "pitch": {
                "mean_f0_hz": float(np.mean(f0)),
                "median_f0_hz": float(np.median(f0)),
                "tone_shape": self.summarize_tone_shape(f0),
            },
            "formants": {"mean_f1_hz": None, "mean_f2_hz": None},
            "calculation_references": ACCENT_TOOL_REFERENCES,
        }


def test_clean_f0_removes_unvoiced_and_invalid_frames():
    """Verify invalid and unvoiced F0 frames are removed."""

    f0 = np.array([0.0, 120.0, np.nan, -1.0, np.inf, 180.0])

    cleaned = AccentEvaluator._clean_f0(f0)

    np.testing.assert_array_equal(cleaned, np.array([120.0, 180.0]))


def test_semitone_normalization_uses_median_reference():
    """Verify semitone normalization is relative to median F0."""

    f0 = np.array([100.0, 200.0, 400.0])

    normalized = AccentEvaluator.normalize_f0(f0, method="semitone")

    np.testing.assert_allclose(normalized, np.array([-12.0, 0.0, 12.0]))


def test_zscore_normalization_handles_flat_contours():
    """Verify flat contours produce zero z-score output."""

    f0 = np.array([150.0, 150.0, 150.0])

    normalized = AccentEvaluator.normalize_f0(f0, method="zscore")

    np.testing.assert_array_equal(normalized, np.zeros(3))


def test_tone_shape_counts_turning_points():
    """Verify tone-shape summaries count contour direction changes."""

    f0 = np.array([100.0, 130.0, 160.0, 120.0, 90.0])

    shape = AccentEvaluator.summarize_tone_shape(f0)

    assert shape["turning_point_count"] == 1
    assert shape["range_st"] > 0
    assert shape["slope_st_per_frame"] < 0


def test_dtw_distance_matches_shifted_contours():
    """Verify DTW aligns repeated contour points with zero distance."""

    reference = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([0.0, 1.0, 1.0, 2.0, 3.0])

    distance, path_length = AccentEvaluator._dtw_distance(reference, target)

    assert distance == pytest.approx(0.0)
    assert path_length >= len(reference)


def test_compute_acoustic_distance_returns_references_and_shape_distance():
    """Verify acoustic distance output includes references and shape distances."""

    evaluator = SyntheticAccentEvaluator(
        {
            "reference.wav": np.array([100.0, 120.0, 150.0, 180.0]),
            "target.wav": np.array([105.0, 125.0, 145.0, 160.0]),
        }
    )

    result = evaluator.compute_acoustic_distance("reference.wav", "target.wav")

    assert result["metric"] == "dtw_semitone_f0"
    assert result["normalized_distance"] >= 0
    assert result["shape_distance"]["range_st"] >= 0
    assert result["calculation_references"][0]["key"] == "boersma_weenink_praat"


def test_feature_matrix_summary_returns_shape_mean_and_std():
    """Verify matrix summaries expose coefficient-wise statistics."""

    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])

    summary = AccentEvaluator._summarize_feature_matrix(matrix, prefix="mfcc")

    assert summary["mfcc_shape"] == [2, 3]
    np.testing.assert_allclose(summary["mfcc_mean"], [2.0, 2.0])
    np.testing.assert_allclose(summary["mfcc_std"], [np.std([1.0, 2.0, 3.0]), 0.0])


def test_normalized_entropy_handles_energy_vectors():
    """Verify normalized entropy is bounded for non-negative vectors."""

    entropy = AccentEvaluator._normalized_entropy(np.array([1.0, 1.0, 1.0]))

    assert entropy == pytest.approx(1.0)
    assert AccentEvaluator._normalized_entropy(np.zeros(3)) is None


def test_hilbert_fallback_features_are_json_ready():
    """Verify SciPy Hilbert fallback returns instantaneous summaries."""

    sample_rate = 200.0
    t = np.arange(0.0, 1.0, 1.0 / sample_rate)
    waveform = np.sin(2.0 * np.pi * 12.0 * t)

    features = AccentEvaluator._extract_hilbert_fallback_features(waveform, sample_rate)

    assert features["backend"] == "numpy.fft.hilbert_fallback"
    assert features["available"] is True
    assert features["imf_count"] == 1
    assert features["instantaneous_frequency_mean_hz"] == pytest.approx(12.0, rel=0.05)
