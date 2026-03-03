"""
Fairness Metrics for Criminal Justice AI Evaluation

Implements fairness metrics from the literature:
- Berk et al. (2021) "Fairness in Criminal Justice Risk Assessments: The State of the Art"
- Verrey et al. (2025) "A fairness scale for real-time recidivism forecasts"
- Chouldechova (2017) Impossibility theorem metrics

Metrics Categories:
1. Group Fairness (Statistical)
2. Individual Fairness
3. Calibration-based Fairness
4. Error Rate Balance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class FairnessReport:
    """Container for fairness evaluation results."""
    dataset: str
    model: str
    protected_attribute: str
    metrics: Dict[str, float]
    group_metrics: Dict[str, Dict[str, float]]
    passes_thresholds: Dict[str, bool]


# =============================================================================
# Core Fairness Metrics
# =============================================================================

def statistical_parity_difference(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """
    Statistical Parity Difference (SPD)
    
    Measures difference in positive prediction rates between groups.
    SPD = P(Y_hat=1|A=0) - P(Y_hat=1|A=1)
    
    Ideal value: 0
    Threshold: |SPD| < 0.1
    
    Also known as: Demographic Parity Difference
    """
    mask_0 = protected == 0
    mask_1 = protected == 1
    
    rate_0 = y_pred[mask_0].mean() if mask_0.sum() > 0 else 0
    rate_1 = y_pred[mask_1].mean() if mask_1.sum() > 0 else 0
    
    return rate_0 - rate_1


def disparate_impact(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """
    Disparate Impact (DI)
    
    Ratio of positive prediction rates between groups.
    DI = P(Y_hat=1|A=0) / P(Y_hat=1|A=1)
    
    Ideal value: 1.0
    Legal threshold (80% rule): 0.8 <= DI <= 1.25
    
    Also known as: Statistical Parity Ratio
    """
    mask_0 = protected == 0
    mask_1 = protected == 1
    
    rate_0 = y_pred[mask_0].mean() if mask_0.sum() > 0 else 0
    rate_1 = y_pred[mask_1].mean() if mask_1.sum() > 0 else 0
    
    if rate_1 == 0:
        return np.inf if rate_0 > 0 else 1.0
    
    return rate_0 / rate_1


def equal_opportunity_difference(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    protected: np.ndarray
) -> float:
    """
    Equal Opportunity Difference (EOD)
    
    Difference in True Positive Rates between groups.
    EOD = TPR(A=0) - TPR(A=1)
    
    Ideal value: 0
    Threshold: |EOD| < 0.1
    
    Focuses on equal benefit for qualified individuals.
    """
    tpr_0 = _true_positive_rate(y_true, y_pred, protected, 0)
    tpr_1 = _true_positive_rate(y_true, y_pred, protected, 1)
    
    return tpr_0 - tpr_1


def predictive_equality_difference(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    protected: np.ndarray
) -> float:
    """
    Predictive Equality Difference (PED)
    
    Difference in False Positive Rates between groups.
    PED = FPR(A=0) - FPR(A=1)
    
    Ideal value: 0
    Threshold: |PED| < 0.1
    
    Focuses on equal burden for unqualified individuals.
    """
    fpr_0 = _false_positive_rate(y_true, y_pred, protected, 0)
    fpr_1 = _false_positive_rate(y_true, y_pred, protected, 1)
    
    return fpr_0 - fpr_1


def average_odds_difference(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    protected: np.ndarray
) -> float:
    """
    Average Odds Difference (AOD)
    
    Average of TPR and FPR differences.
    AOD = 0.5 * (|TPR_diff| + |FPR_diff|)
    
    Ideal value: 0
    Threshold: AOD < 0.1
    
    Also known as: Equalized Odds Difference
    """
    eod = equal_opportunity_difference(y_true, y_pred, protected)
    ped = predictive_equality_difference(y_true, y_pred, protected)
    
    return 0.5 * (abs(eod) + abs(ped))


def equalized_odds_ratio(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    protected: np.ndarray
) -> float:
    """
    Equalized Odds Ratio
    
    Minimum of TPR ratio and FPR ratio.
    
    Ideal value: 1.0
    """
    tpr_0 = _true_positive_rate(y_true, y_pred, protected, 0)
    tpr_1 = _true_positive_rate(y_true, y_pred, protected, 1)
    fpr_0 = _false_positive_rate(y_true, y_pred, protected, 0)
    fpr_1 = _false_positive_rate(y_true, y_pred, protected, 1)
    
    # Avoid division by zero
    tpr_ratio = min(tpr_0, tpr_1) / max(tpr_0, tpr_1) if max(tpr_0, tpr_1) > 0 else 1.0
    fpr_ratio = min(fpr_0, fpr_1) / max(fpr_0, fpr_1) if max(fpr_0, fpr_1) > 0 else 1.0
    
    return min(tpr_ratio, fpr_ratio)


# =============================================================================
# Calibration-based Fairness (Berk et al. 2021)
# =============================================================================

def calibration_difference(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    protected: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calibration Difference between groups.
    
    Measures whether predicted probabilities have the same meaning across groups.
    
    Based on: "Conditional Use Accuracy Equality" (Berk et al. 2021)
    """
    def _calibration_error(y_true, y_proba, n_bins):
        """Expected Calibration Error for one group."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                avg_pred = y_proba[mask].mean()
                avg_true = y_true[mask].mean()
                ece += mask.sum() * abs(avg_pred - avg_true)
        
        return ece / len(y_true)
    
    mask_0 = protected == 0
    mask_1 = protected == 1
    
    ece_0 = _calibration_error(y_true[mask_0], y_pred_proba[mask_0], n_bins) if mask_0.sum() > 0 else 0
    ece_1 = _calibration_error(y_true[mask_1], y_pred_proba[mask_1], n_bins) if mask_1.sum() > 0 else 0
    
    return abs(ece_0 - ece_1)


def positive_predictive_value_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> float:
    """
    Positive Predictive Value (Precision) Difference between groups.
    
    PPV_diff = PPV(A=0) - PPV(A=1)
    
    Related to "Predictive Parity" from Berk et al. (2021)
    """
    ppv_0 = _positive_predictive_value(y_true, y_pred, protected, 0)
    ppv_1 = _positive_predictive_value(y_true, y_pred, protected, 1)
    
    return ppv_0 - ppv_1


def negative_predictive_value_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> float:
    """
    Negative Predictive Value Difference between groups.
    
    NPV_diff = NPV(A=0) - NPV(A=1)
    """
    npv_0 = _negative_predictive_value(y_true, y_pred, protected, 0)
    npv_1 = _negative_predictive_value(y_true, y_pred, protected, 1)
    
    return npv_0 - npv_1


# =============================================================================
# Treatment Equality (Berk et al. 2021)
# =============================================================================

def treatment_equality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> float:
    """
    Treatment Equality
    
    Ratio of FN to FP should be equal across groups.
    TE = |FN/FP(A=0) - FN/FP(A=1)|
    
    From Berk et al. (2021): "Treatment Equality"
    """
    def _fn_fp_ratio(y_true, y_pred, protected, group):
        mask = protected == group
        tn, fp, fn, tp = confusion_matrix(
            y_true[mask], y_pred[mask], labels=[0, 1]
        ).ravel()
        
        if fp == 0:
            return np.inf if fn > 0 else 0
        return fn / fp
    
    ratio_0 = _fn_fp_ratio(y_true, y_pred, protected, 0)
    ratio_1 = _fn_fp_ratio(y_true, y_pred, protected, 1)
    
    if np.isinf(ratio_0) or np.isinf(ratio_1):
        return np.inf
    
    return abs(ratio_0 - ratio_1)


# =============================================================================
# Helper Functions
# =============================================================================

def _true_positive_rate(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    protected: np.ndarray, 
    group: int
) -> float:
    """TPR = TP / (TP + FN) for a specific group."""
    mask = protected == group
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    
    positives = y_true_g == 1
    if positives.sum() == 0:
        return 0.0
    
    return y_pred_g[positives].mean()


def _false_positive_rate(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    protected: np.ndarray, 
    group: int
) -> float:
    """FPR = FP / (FP + TN) for a specific group."""
    mask = protected == group
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    
    negatives = y_true_g == 0
    if negatives.sum() == 0:
        return 0.0
    
    return y_pred_g[negatives].mean()


def _positive_predictive_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    group: int
) -> float:
    """PPV = TP / (TP + FP) for a specific group."""
    mask = protected == group
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    
    predicted_positive = y_pred_g == 1
    if predicted_positive.sum() == 0:
        return 0.0
    
    return y_true_g[predicted_positive].mean()


def _negative_predictive_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    group: int
) -> float:
    """NPV = TN / (TN + FN) for a specific group."""
    mask = protected == group
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    
    predicted_negative = y_pred_g == 0
    if predicted_negative.sum() == 0:
        return 0.0
    
    return (1 - y_true_g[predicted_negative]).mean()


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def compute_all_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    protected: np.ndarray
) -> Dict[str, float]:
    """
    Compute all fairness metrics.
    
    Parameters
    ----------
    y_true : array
        Ground truth labels
    y_pred : array
        Predicted labels (binary)
    y_pred_proba : array, optional
        Predicted probabilities for positive class
    protected : array
        Protected attribute values (binary: 0/1)
        
    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)
    
    metrics = {
        # Statistical Parity
        'statistical_parity_difference': statistical_parity_difference(y_pred, protected),
        'disparate_impact': disparate_impact(y_pred, protected),
        
        # Error Rate Balance
        'equal_opportunity_difference': equal_opportunity_difference(y_true, y_pred, protected),
        'predictive_equality_difference': predictive_equality_difference(y_true, y_pred, protected),
        'average_odds_difference': average_odds_difference(y_true, y_pred, protected),
        'equalized_odds_ratio': equalized_odds_ratio(y_true, y_pred, protected),
        
        # Predictive Parity
        'ppv_difference': positive_predictive_value_difference(y_true, y_pred, protected),
        'npv_difference': negative_predictive_value_difference(y_true, y_pred, protected),
        
        # Treatment Equality
        'treatment_equality': treatment_equality(y_true, y_pred, protected),
    }
    
    # Calibration metrics (require probabilities)
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba)
        metrics['calibration_difference'] = calibration_difference(
            y_true, y_pred_proba, protected
        )
    
    return metrics


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    protected: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each protected group.
    
    Returns
    -------
    dict
        {'group_0': {...metrics...}, 'group_1': {...metrics...}}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)
    
    results = {}
    
    for group in [0, 1]:
        mask = protected == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        if len(y_true_g) == 0:
            continue
            
        tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
        
        group_metrics = {
            'n_samples': int(mask.sum()),
            'base_rate': float(y_true_g.mean()),
            'positive_rate': float(y_pred_g.mean()),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
        }
        
        if y_pred_proba is not None:
            y_pred_proba = np.asarray(y_pred_proba)
            try:
                group_metrics['auc'] = roc_auc_score(y_true_g, y_pred_proba[mask])
            except ValueError:
                group_metrics['auc'] = np.nan
        
        results[f'group_{group}'] = group_metrics
    
    return results


def check_fairness_thresholds(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, bool]:
    """
    Check if metrics pass standard fairness thresholds.
    
    Default thresholds from literature:
    - SPD: |value| < 0.1
    - DI: 0.8 <= value <= 1.25 (80% rule)
    - EOD: |value| < 0.1
    - AOD: value < 0.1
    """
    if thresholds is None:
        thresholds = {
            'statistical_parity_difference': 0.1,
            'disparate_impact': (0.8, 1.25),
            'equal_opportunity_difference': 0.1,
            'predictive_equality_difference': 0.1,
            'average_odds_difference': 0.1,
            'ppv_difference': 0.1,
            'npv_difference': 0.1,
            'calibration_difference': 0.05,
        }
    
    results = {}
    
    for metric, value in metrics.items():
        if metric not in thresholds:
            continue
            
        threshold = thresholds[metric]
        
        if isinstance(threshold, tuple):
            # Range threshold (e.g., disparate impact)
            results[metric] = threshold[0] <= value <= threshold[1]
        else:
            # Absolute threshold
            results[metric] = abs(value) < threshold
    
    return results


def evaluate_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    protected: np.ndarray,
    dataset_name: str = "unknown",
    model_name: str = "unknown",
    protected_attr_name: str = "unknown"
) -> FairnessReport:
    """
    Complete fairness evaluation with report generation.
    """
    metrics = compute_all_fairness_metrics(y_true, y_pred, y_pred_proba, protected)
    group_metrics = compute_group_metrics(y_true, y_pred, y_pred_proba, protected)
    passes = check_fairness_thresholds(metrics)
    
    return FairnessReport(
        dataset=dataset_name,
        model=model_name,
        protected_attribute=protected_attr_name,
        metrics=metrics,
        group_metrics=group_metrics,
        passes_thresholds=passes
    )


def print_fairness_report(report: FairnessReport):
    """Pretty print a fairness report."""
    print("\n" + "=" * 70)
    print(f"FAIRNESS REPORT: {report.model} on {report.dataset}")
    print(f"Protected Attribute: {report.protected_attribute}")
    print("=" * 70)
    
    print("\n--- Group Statistics ---")
    for group, stats in report.group_metrics.items():
        print(f"\n{group}:")
        for key, val in stats.items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    print("\n--- Fairness Metrics ---")
    for metric, value in report.metrics.items():
        passes = report.passes_thresholds.get(metric, None)
        status = "✓" if passes else "✗" if passes is not None else "-"
        print(f"  {metric}: {value:.4f} [{status}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.binomial(1, 0.3, n)
    y_pred = np.random.binomial(1, 0.35, n)
    y_pred_proba = np.clip(y_pred + np.random.normal(0, 0.2, n), 0, 1)
    protected = np.random.binomial(1, 0.4, n)
    
    report = evaluate_fairness(
        y_true, y_pred, y_pred_proba, protected,
        dataset_name="Synthetic",
        model_name="Random Baseline",
        protected_attr_name="Group"
    )
    
    print_fairness_report(report)
