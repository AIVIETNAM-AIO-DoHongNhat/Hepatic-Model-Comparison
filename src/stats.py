"""Statistical comparison and multiclass calibration utilities."""

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve

CLASS_ORDER = ("C", "CL", "D")


@dataclass(frozen=True)
class PairedTestResult:
    method: str
    statistic: float
    p_value: float
    mean_difference: float
    n_observations: int
    all_differences_zero: bool = False


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    method: str
    confidence: float
    mean_difference: float
    low: float
    high: float
    n_observations: int


@dataclass(frozen=True)
class BonferroniResult:
    raw_p_values: np.ndarray
    adjusted_p_values: np.ndarray
    reject: np.ndarray
    family_size: int
    alpha: float
    adjusted_alpha: float


def _paired_arrays(
    scores_a: Sequence[float], scores_b: Sequence[float], *, min_observations: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate paired one-dimensional finite score vectors."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Paired scores must be one-dimensional vectors.")
    if len(a) != len(b):
        raise ValueError("Paired score vectors must have the same length.")
    if len(a) < min_observations:
        raise ValueError(f"At least {min_observations} paired observations are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired score vectors must not contain NaN or infinite values.")
    return a, b, a - b


def paired_ttest(scores_a: Sequence[float], scores_b: Sequence[float]) -> PairedTestResult:
    """Paired t-test for ``difference = scores_a - scores_b``."""
    _, _, diff = _paired_arrays(scores_a, scores_b)
    all_zero = bool(np.all(diff == 0))
    if all_zero:
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.ttest_1samp(diff, popmean=0.0)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return PairedTestResult(
        method="paired_t_test",
        statistic=statistic,
        p_value=p_value,
        mean_difference=float(diff.mean()),
        n_observations=len(diff),
        all_differences_zero=all_zero,
    )


def wilcoxon_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> PairedTestResult:
    """Wilcoxon signed-rank sensitivity test on paired differences."""
    _, _, diff = _paired_arrays(scores_a, scores_b)
    all_zero = bool(np.all(diff == 0))
    if all_zero:
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return PairedTestResult(
        method="wilcoxon",
        statistic=statistic,
        p_value=p_value,
        mean_difference=float(diff.mean()),
        n_observations=len(diff),
        all_differences_zero=all_zero,
    )


def shapiro_test(differences: Sequence[float]) -> PairedTestResult:
    """Shapiro-Wilk normality check for an already-computed difference vector."""
    diff = np.asarray(differences, dtype=float)
    if diff.ndim != 1:
        raise ValueError("Differences must be a one-dimensional vector.")
    if len(diff) < 3:
        raise ValueError("Shapiro-Wilk requires at least 3 observations.")
    if not np.isfinite(diff).all():
        raise ValueError("Differences must not contain NaN or infinite values.")
    if np.all(diff == diff[0]):
        statistic, p_value = 1.0, 1.0
    else:
        result = stats.shapiro(diff)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return PairedTestResult(
        method="shapiro_wilk",
        statistic=statistic,
        p_value=p_value,
        mean_difference=float(diff.mean()),
        n_observations=len(diff),
        all_differences_zero=bool(np.all(diff == 0)),
    )


def ci_diff_t(
    scores_a: Sequence[float], scores_b: Sequence[float], confidence: float = 0.95
) -> ConfidenceIntervalResult:
    """Student-t confidence interval for the mean paired difference."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")
    _, _, diff = _paired_arrays(scores_a, scores_b)
    mean = float(diff.mean())
    if np.all(diff == diff[0]):
        low = high = mean
    else:
        se = float(stats.sem(diff))
        low, high = stats.t.interval(confidence, len(diff) - 1, loc=mean, scale=se)
    return ConfidenceIntervalResult(
        method="student_t",
        confidence=confidence,
        mean_difference=mean,
        low=float(low),
        high=float(high),
        n_observations=len(diff),
    )


def ci_diff_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 10_000,
    random_state: int = 42,
) -> ConfidenceIntervalResult:
    """Deterministic percentile bootstrap CI for the mean paired difference."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100.")
    _, _, diff = _paired_arrays(scores_a, scores_b)
    rng = np.random.default_rng(random_state)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return ConfidenceIntervalResult(
        method="percentile_bootstrap",
        confidence=confidence,
        mean_difference=float(diff.mean()),
        low=float(low),
        high=float(high),
        n_observations=len(diff),
    )


def bonferroni_correction(
    pvalues: Sequence[float], alpha: float = 0.05
) -> BonferroniResult:
    """Bonferroni correction for one explicitly supplied comparison family."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("pvalues must be a non-empty one-dimensional vector.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and within [0, 1].")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    family_size = len(p)
    adjusted = np.minimum(p * family_size, 1.0)
    return BonferroniResult(
        raw_p_values=p.copy(),
        adjusted_p_values=adjusted,
        reject=adjusted <= alpha,
        family_size=family_size,
        alpha=alpha,
        adjusted_alpha=alpha / family_size,
    )


def validate_multiclass_probabilities(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels and a probability matrix in fixed C, CL, D order."""
    y = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if proba.ndim != 2 or proba.shape[1] != len(CLASS_ORDER):
        raise ValueError("y_proba must have shape (n_samples, 3) in C, CL, D order.")
    if len(y) != len(proba) or len(y) == 0:
        raise ValueError("y_true and y_proba must contain the same non-zero number of rows.")
    if not np.isfinite(proba).all():
        raise ValueError("Probabilities must not contain NaN or infinite values.")
    if np.any((proba < 0) | (proba > 1)):
        raise ValueError("Every probability must lie within [0, 1].")
    if not np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1 within atol=1e-6.")
    try:
        y_int = y.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must use numeric labels 0=C, 1=CL, 2=D.") from exc
    if not np.array_equal(y, y_int) or not np.isin(y_int, np.arange(len(CLASS_ORDER))).all():
        raise ValueError("y_true must use numeric labels 0=C, 1=CL, 2=D.")
    return y_int, proba


def brier_multiclass(y_true: Sequence[int], y_proba: Sequence[Sequence[float]]) -> float:
    """Multiclass Brier score, lower is better."""
    y, proba = validate_multiclass_probabilities(y_true, y_proba)
    onehot = np.eye(len(CLASS_ORDER))[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def top_label_reliability_bins(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]], n_bins: int = 10
) -> pd.DataFrame:
    """Return all top-label reliability bins, including empty bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y, proba = validate_multiclass_probabilities(y_true, y_proba)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indexes = np.minimum(np.digitize(confidence, edges[1:-1], right=True), n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = indexes == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else np.nan
        observed_accuracy = float(correct[mask].mean()) if count else np.nan
        gap = abs(observed_accuracy - mean_confidence) if count else np.nan
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "absolute_gap": gap,
                "weight": count / len(y),
            }
        )
    return pd.DataFrame(rows)


def top_label_ece(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]], n_bins: int = 10
) -> float:
    bins = top_label_reliability_bins(y_true, y_proba, n_bins=n_bins)
    populated = bins["count"] > 0
    return float((bins.loc[populated, "weight"] * bins.loc[populated, "absolute_gap"]).sum())


def classwise_reliability_bins(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]], n_bins: int = 10
) -> pd.DataFrame:
    """One-vs-rest reliability bins for C, CL and D, with empty bins retained."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y, proba = validate_multiclass_probabilities(y_true, y_proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for class_index, class_name in enumerate(CLASS_ORDER):
        probability = proba[:, class_index]
        observed = (y == class_index).astype(float)
        indexes = np.minimum(np.digitize(probability, edges[1:-1], right=True), n_bins - 1)
        for index in range(n_bins):
            mask = indexes == index
            count = int(mask.sum())
            mean_probability = float(probability[mask].mean()) if count else np.nan
            observed_frequency = float(observed[mask].mean()) if count else np.nan
            gap = abs(observed_frequency - mean_probability) if count else np.nan
            rows.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                    "absolute_gap": gap,
                    "weight": count / len(y),
                }
            )
    return pd.DataFrame(rows)


def macro_classwise_ece(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]], n_bins: int = 10
) -> float:
    bins = classwise_reliability_bins(y_true, y_proba, n_bins=n_bins)
    class_eces = []
    for class_name in CLASS_ORDER:
        subset = bins[(bins["class_name"] == class_name) & (bins["count"] > 0)]
        class_eces.append(float((subset["weight"] * subset["absolute_gap"]).sum()))
    return float(np.mean(class_eces))


def plot_reliability_diagram(
    y_true: Sequence[int],
    y_proba: Sequence[Sequence[float]],
    n_bins: int = 10,
    include_classwise: bool = True,
):
    """Create top-label and optional one-vs-rest reliability diagrams."""
    top_bins = top_label_reliability_bins(y_true, y_proba, n_bins=n_bins)
    class_bins = classwise_reliability_bins(y_true, y_proba, n_bins=n_bins)
    n_axes = 2 if include_classwise else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5), squeeze=False)
    top_ax = axes[0, 0]
    top_ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    populated = top_bins[top_bins["count"] > 0]
    top_ax.plot(
        populated["mean_confidence"], populated["observed_accuracy"], marker="o", label="Top-label"
    )
    for row in populated.itertuples():
        top_ax.annotate(f"n={row.count}", (row.mean_confidence, row.observed_accuracy), fontsize=8)
    top_ax.set(title=f"Top-label reliability ({n_bins} uniform bins)", xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
    top_ax.legend()

    if include_classwise:
        class_ax = axes[0, 1]
        class_ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for class_name in CLASS_ORDER:
            subset = class_bins[(class_bins["class_name"] == class_name) & (class_bins["count"] > 0)]
            class_ax.plot(subset["mean_probability"], subset["observed_frequency"], marker="o", label=class_name)
        class_ax.set(title=f"One-vs-rest reliability — class order {', '.join(CLASS_ORDER)}", xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
        class_ax.legend()
    fig.tight_layout()
    return fig, axes.ravel()[:n_axes], top_bins, class_bins


# Backward-compatible calibration wrappers used in earlier documentation.
def expected_calibration_error(y_true, y_proba, n_bins=10):
    return top_label_ece(y_true, y_proba, n_bins=n_bins)


def reliability_data(y_true, y_proba, class_idx, n_bins=10):
    y, proba = validate_multiclass_probabilities(y_true, y_proba)
    if class_idx not in range(len(CLASS_ORDER)):
        raise ValueError("class_idx must be 0=C, 1=CL, or 2=D.")
    y_bin = (y == class_idx).astype(int)
    return calibration_curve(y_bin, proba[:, class_idx], n_bins=n_bins)
