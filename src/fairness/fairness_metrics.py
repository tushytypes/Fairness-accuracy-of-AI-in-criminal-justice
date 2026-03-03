"""
Fairness Metrics for Criminal Justice AI Evaluation

Based on:
- Berk et al. (2021) "Fairness in Criminal Justice Risk Assessments"
- Chouldechova (2017) Impossibility theorem
- Verrey et al. (2025) Hierarchical fairness scale
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Dict, Optional, List
from dataclasses import dataclass


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
    SPD = P(Y_hat=1|A=0) - P(Y_hat=1|A=1)
    Ideal: 0, Threshold: |SPD| < 0.1
    """
    mask_0 = protected == 0
    mask_1 = protected == 1
    rate_0 = y_pred[mask_0].mean() if mask_0.sum() > 0 else 0
    rate_1 = y_pred[mask_1].mean() if mask_1.sum() > 0 else 0
    return rate_0 - rate_1


def disparate_impact(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """
    Disparate Impact (DI)
    DI = P(Y_hat=1|A=0) / P(Y_hat=1|A=1)
    Ideal: 1.0, Threshold: 0.8 <= DI <= 1.25 (80% rule)
    """
    mask_0 = protected == 0
    mask_1 = protected == 1
    rate_0 = y_pred[mask_0].mean() if mask_0.sum() > 0 else 0
    rate_1 = y_pred[mask_1].mean() if mask_1.sum() > 0 else 0
    if rate_1 == 0:
        return np.inf if rate_0 > 0 else 1.0
    return rate_0 / rate_1


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                                  protected: np.ndarray) -> float:
    """
    Equal Opportunity Difference (EOD)
    EOD = TPR(A=0) - TPR(A=1)
    Ideal: 0, Threshold: |EOD| < 0.1
    """
    tpr_0 = _true_positive_rate(y_true, y_pred, protected, 0)
    tpr_1 = _true_positive_rate(y_true, y_pred, protected, 1)
    return tpr_0 - tpr_1


def predictive_equality_difference(y_true: np.ndarray, y_pred: np.ndarray,
                                   protected: np.ndarray) -> float:
    """
    Predictive Equality Difference (PED) - FPR difference
    PED = FPR(A=0) - FPR(A=1)
    Ideal: 0, Threshold: |PED| < 0.1
    """
    fpr_0 = _false_positive_rate(y_true, y_pred, protected, 0)
    fpr_1 = _false_positive_rate(y_true, y_pred, protected, 1)
    return fpr_0 - fpr_1


def average_odds_difference(y_true: np.ndarray, y_pred: np.ndarray,
                           protected: np.ndarray) -> float:
    """
    Average Odds Difference (AOD)
    AOD = 0.5 * (|TPR_diff| + |FPR_diff|)
    Ideal: 0, Threshold: AOD < 0.1
    """
    eod = equal_opportunity_difference(y_true, y_pred, protected)
    ped = predictive_equality_difference(y_true, y_pred, protected)
    return 0.5 * (abs(eod) + abs(ped))


def equalized_odds_ratio(y_true: np.ndarray, y_pred: np.ndarray,
                        protected: np.ndarray) -> float:
    """Minimum of TPR ratio and FPR ratio. Ideal: 1.0"""
    tpr_0 = _true_positive_rate(y_true, y_pred, protected, 0)
    tpr_1 = _true_positive_rate(y_true, y_pred, protected, 1)
    fpr_0 = _false_positive_rate(y_true, y_pred, protected, 0)
    fpr_1 = _false_positive_rate(y_true, y_pred, protected, 1)
    
    tpr_ratio = min(tpr_0, tpr_1) / max(tpr_0, tpr_1) if max(tpr_0, tpr_1) > 0 else 1.0
    fpr_ratio = min(fpr_0, fpr_1) / max(fpr_0, fpr_1) if max(fpr_0, fpr_1) > 0 else 1.0
    
    return min(tpr_ratio, fpr_ratio)


def calibration_difference(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          protected: np.ndarray, n_bins: int = 10) -> float:
    """ECE difference between groups."""
    def _ece(y_true, y_proba, n_bins):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                ece += mask.sum() * abs(y_proba[mask].mean() - y_true[mask].mean())
        return ece / len(y_true)
    
    mask_0 = protected == 0
    mask_1 = protected == 1
    ece_0 = _ece(y_true[mask_0], y_pred_proba[mask_0], n_bins) if mask_0.sum() > 0 else 0
    ece_1 = _ece(y_true[mask_1], y_pred_proba[mask_1], n_bins) if mask_1.sum() > 0 else 0
    return abs(ece_0 - ece_1)


def ppv_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                   protected: np.ndarray) -> float:
    """Positive Predictive Value (Precision) difference."""
    ppv_0 = _positive_predictive_value(y_true, y_pred, protected, 0)
    ppv_1 = _positive_predictive_value(y_true, y_pred, protected, 1)
    return ppv_0 - ppv_1


def npv_difference(y_true: np.ndarray, y_pred: np.ndarray,
                   protected: np.ndarray) -> float:
    """Negative Predictive Value difference."""
    npv_0 = _negative_predictive_value(y_true, y_pred, protected, 0)
    npv_1 = _negative_predictive_value(y_true, y_pred, protected, 1)
    return npv_0 - npv_1


def treatment_equality(y_true: np.ndarray, y_pred: np.ndarray,
                       protected: np.ndarray) -> float:
    """
    Treatment Equality: |FN/FP(A=0) - FN/FP(A=1)|
    From Berk et al. (2021)
    """
    def _fn_fp_ratio(y_true, y_pred, protected, group):
        mask = protected == group
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
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

def _true_positive_rate(y_true, y_pred, protected, group) -> float:
    mask = protected == group
    positives = y_true[mask] == 1
    if positives.sum() == 0:
        return 0.0
    return y_pred[mask][positives].mean()


def _false_positive_rate(y_true, y_pred, protected, group) -> float:
    mask = protected == group
    negatives = y_true[mask] == 0
    if negatives.sum() == 0:
        return 0.0
    return y_pred[mask][negatives].mean()


def _positive_predictive_value(y_true, y_pred, protected, group) -> float:
    mask = protected == group
    pred_pos = y_pred[mask] == 1
    if pred_pos.sum() == 0:
        return 0.0
    return y_true[mask][pred_pos].mean()


def _negative_predictive_value(y_true, y_pred, protected, group) -> float:
    mask = protected == group
    pred_neg = y_pred[mask] == 0
    if pred_neg.sum() == 0:
        return 0.0
    return (1 - y_true[mask][pred_neg]).mean()


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def compute_all_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray],
                                 protected: np.ndarray) -> Dict[str, float]:
    """Compute all fairness metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)
    
    metrics = {
        'spd': statistical_parity_difference(y_pred, protected),
        'di': disparate_impact(y_pred, protected),
        'eod': equal_opportunity_difference(y_true, y_pred, protected),
        'ped': predictive_equality_difference(y_true, y_pred, protected),
        'aod': average_odds_difference(y_true, y_pred, protected),
        'eor': equalized_odds_ratio(y_true, y_pred, protected),
        'ppv_diff': ppv_difference(y_true, y_pred, protected),
        'npv_diff': npv_difference(y_true, y_pred, protected),
        'treatment_equality': treatment_equality(y_true, y_pred, protected),
    }
    
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba)
        metrics['calibration_diff'] = calibration_difference(y_true, y_pred_proba, protected)
    
    return metrics


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray],
                          protected: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute metrics separately for each group."""
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
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
        }
        
        if y_pred_proba is not None:
            y_pred_proba = np.asarray(y_pred_proba)
            try:
                group_metrics['auc'] = roc_auc_score(y_true_g, y_pred_proba[mask])
            except:
                group_metrics['auc'] = np.nan
        
        results[f'group_{group}'] = group_metrics
    
    return results


def check_fairness_thresholds(metrics: Dict[str, float]) -> Dict[str, bool]:
    """Check if metrics pass standard thresholds."""
    thresholds = {
        'spd': 0.1,
        'di': (0.8, 1.25),
        'eod': 0.1,
        'ped': 0.1,
        'aod': 0.1,
        'ppv_diff': 0.1,
        'npv_diff': 0.1,
        'calibration_diff': 0.05,
    }
    
    results = {}
    for metric, value in metrics.items():
        if metric not in thresholds:
            continue
        threshold = thresholds[metric]
        if isinstance(threshold, tuple):
            results[metric] = threshold[0] <= value <= threshold[1]
        else:
            results[metric] = abs(value) < threshold
    
    return results


def evaluate_fairness(y_true, y_pred, y_pred_proba, protected,
                      dataset_name="unknown", model_name="unknown",
                      protected_attr_name="unknown") -> FairnessReport:
    """Complete fairness evaluation."""
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
    """Pretty print fairness report."""
    print("\n" + "=" * 60)
    print(f"FAIRNESS REPORT: {report.model} on {report.dataset}")
    print(f"Protected Attribute: {report.protected_attribute}")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.3, n)
    y_pred = np.random.binomial(1, 0.35, n)
    y_pred_proba = np.clip(y_pred + np.random.normal(0, 0.2, n), 0, 1)
    protected = np.random.binomial(1, 0.4, n)
    
    report = evaluate_fairness(y_true, y_pred, y_pred_proba, protected,
                               "Synthetic", "Random Baseline", "Group")
    print_fairness_report(report)
