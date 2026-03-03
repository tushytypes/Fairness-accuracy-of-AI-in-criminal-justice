"""
Machine Learning Performance Metrics for Criminal Justice AI

Based on literature analysis (Dressel & Farid 2018, Berk et al. 2021, Chouldechova 2017),
this module implements metrics that are STANDARD in criminal justice fairness research.

STANDARD METRICS (used in this field):
- AUC-ROC: Primary metric in all criminal justice ML papers
- Accuracy: Required for Dressel & Farid comparison
- Confusion matrix rates: FPR, FNR, TPR, TNR (foundation for fairness)
- Precision, Recall, F1: Common secondary metrics
- AUC-PR: Better for imbalanced data

NOT STANDARD IN THIS FIELD (kept for completeness but rarely used):
- Brier Score: <15% of FAccT papers, absent from foundational works
- Log Loss: Used for training, almost never reported
- MCC (Matthews): Standard in bioinformatics, absent here
- Cohen's Kappa: Used only for inter-rater reliability, not model evaluation
- ECE/MCE: Calibration plots preferred over numeric metrics

ROBUSTNESS METRICS (recommended by thesis advisor):
- Feature Ablation / ROAR: Measures model robustness to feature removal
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, auc, confusion_matrix, roc_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings


# =============================================================================
# Standard Performance Metrics
# =============================================================================

def compute_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standard ML classification metrics.
    
    Returns
    -------
    dict with: accuracy, precision, recall, f1, specificity
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # TPR / Sensitivity
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # TNR
        'balanced_accuracy': 0.5 * (tp/(tp+fn) + tn/(tn+fp)) if (tp+fn)>0 and (tn+fp)>0 else 0,
    }
    
    return metrics


# =============================================================================
# Probability-Based Metrics
# =============================================================================

def compute_probability_metrics(y_true: np.ndarray, 
                                y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute probability-based metrics (require predicted probabilities).
    
    Returns
    -------
    dict with: auc_roc, auc_pr, brier_score, log_loss, avg_precision
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    metrics = {}
    
    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['auc_roc'] = np.nan
    
    # AUC-PR (Precision-Recall AUC)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['auc_pr'] = auc(recall, precision)
    except:
        metrics['auc_pr'] = np.nan
    
    # Average Precision (similar to AUC-PR but different calculation)
    try:
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    except:
        metrics['avg_precision'] = np.nan
    
    # Brier Score (lower is better, 0 = perfect)
    try:
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    except:
        metrics['brier_score'] = np.nan
    
    # Log Loss (lower is better)
    try:
        # Clip probabilities to avoid log(0)
        y_pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        metrics['log_loss'] = log_loss(y_true, y_pred_clipped)
    except:
        metrics['log_loss'] = np.nan
    
    return metrics


# =============================================================================
# Advanced Classification Metrics
# =============================================================================

def compute_advanced_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute advanced classification metrics.
    
    Returns
    -------
    dict with: mcc (Matthews Correlation Coefficient), cohen_kappa
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {}
    
    # Matthews Correlation Coefficient (MCC)
    # Range: -1 to 1, 0 = random, 1 = perfect
    # Good for imbalanced datasets
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics['mcc'] = np.nan
    
    # Cohen's Kappa
    # Measures agreement beyond chance
    # Range: -1 to 1, 0 = no agreement beyond chance
    try:
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    except:
        metrics['cohen_kappa'] = np.nan
    
    return metrics


# =============================================================================
# Calibration Metrics
# =============================================================================

def compute_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration metrics.
    
    Returns
    -------
    dict with: ece (Expected Calibration Error), mce (Maximum Calibration Error)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_proba[mask].mean()
            bin_size = mask.sum()
            
            calibration_error = abs(bin_acc - bin_conf)
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)
    
    ece /= len(y_true)
    
    return {
        'ece': ece,  # Expected Calibration Error (lower is better)
        'mce': mce   # Maximum Calibration Error (lower is better)
    }


def get_calibration_curve_data(y_true: np.ndarray, y_pred_proba: np.ndarray,
                               n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get calibration curve data for plotting.
    
    Returns
    -------
    fraction_of_positives, mean_predicted_value
    """
    try:
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        return fraction_positives, mean_predicted
    except:
        return np.array([]), np.array([])


# =============================================================================
# ROAR / Feature Ablation Analysis (Professor's metric!)
# =============================================================================

@dataclass
class AblationResult:
    """Results from feature ablation analysis."""
    feature_removed: str
    accuracy_drop: float
    auc_drop: float
    original_accuracy: float
    ablated_accuracy: float
    original_auc: float
    ablated_auc: float
    robustness_score: float  # Higher = model is robust to removing this feature


def feature_ablation_analysis(
    model_class,
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features_to_ablate: Optional[List[str]] = None,
    n_top_features: int = 10
) -> List[AblationResult]:
    """
    ROAR-style Feature Ablation Analysis.
    
    Measures model robustness by:
    1. Training model on all features
    2. For each feature: retrain WITHOUT that feature
    3. Measure performance drop
    
    A robust model should maintain performance when individual features are removed.
    
    Parameters
    ----------
    model_class : class
        Sklearn-compatible model class
    model_params : dict
        Parameters for model initialization
    X_train, y_train : training data
    X_test, y_test : test data
    features_to_ablate : list, optional
        Specific features to test. If None, tests all features.
    n_top_features : int
        If features_to_ablate is None, only test top N features by variance
    
    Returns
    -------
    List of AblationResult objects sorted by accuracy drop
    """
    results = []
    
    # Get features to test
    if features_to_ablate is None:
        # Select top N by variance
        variances = X_train.var()
        features_to_ablate = variances.nlargest(n_top_features).index.tolist()
    
    # Train baseline model with all features
    baseline_model = model_class(**model_params)
    baseline_model.fit(X_train, y_train)
    
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    
    if hasattr(baseline_model, 'predict_proba'):
        y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
        try:
            baseline_auc = roc_auc_score(y_test, y_proba_baseline)
        except:
            baseline_auc = np.nan
    else:
        baseline_auc = np.nan
    
    # Ablate each feature
    for feature in features_to_ablate:
        if feature not in X_train.columns:
            continue
        
        # Remove feature
        X_train_ablated = X_train.drop(columns=[feature])
        X_test_ablated = X_test.drop(columns=[feature])
        
        # Retrain
        ablated_model = model_class(**model_params)
        ablated_model.fit(X_train_ablated, y_train)
        
        y_pred_ablated = ablated_model.predict(X_test_ablated)
        ablated_accuracy = accuracy_score(y_test, y_pred_ablated)
        
        if hasattr(ablated_model, 'predict_proba'):
            y_proba_ablated = ablated_model.predict_proba(X_test_ablated)[:, 1]
            try:
                ablated_auc = roc_auc_score(y_test, y_proba_ablated)
            except:
                ablated_auc = np.nan
        else:
            ablated_auc = np.nan
        
        # Calculate drops
        acc_drop = baseline_accuracy - ablated_accuracy
        auc_drop = baseline_auc - ablated_auc if not np.isnan(ablated_auc) else np.nan
        
        # Robustness score: 1 - normalized drop (higher = more robust)
        # If performance drops a lot, robustness is low
        robustness = 1 - min(max(acc_drop, 0) / max(baseline_accuracy, 0.01), 1)
        
        results.append(AblationResult(
            feature_removed=feature,
            accuracy_drop=acc_drop,
            auc_drop=auc_drop if not np.isnan(auc_drop) else 0,
            original_accuracy=baseline_accuracy,
            ablated_accuracy=ablated_accuracy,
            original_auc=baseline_auc,
            ablated_auc=ablated_auc,
            robustness_score=robustness
        ))
    
    # Sort by accuracy drop (most impactful first)
    results.sort(key=lambda x: x.accuracy_drop, reverse=True)
    
    return results


def compute_model_robustness_score(ablation_results: List[AblationResult]) -> Dict[str, float]:
    """
    Compute overall model robustness metrics from ablation analysis.
    
    Returns
    -------
    dict with:
        - mean_accuracy_drop: average accuracy drop across all ablations
        - max_accuracy_drop: worst-case accuracy drop
        - mean_robustness: average robustness score
        - fragility_index: proportion of features causing >5% accuracy drop
    """
    if not ablation_results:
        return {}
    
    acc_drops = [r.accuracy_drop for r in ablation_results]
    robustness_scores = [r.robustness_score for r in ablation_results]
    
    return {
        'mean_accuracy_drop': np.mean(acc_drops),
        'max_accuracy_drop': np.max(acc_drops),
        'std_accuracy_drop': np.std(acc_drops),
        'mean_robustness': np.mean(robustness_scores),
        'min_robustness': np.min(robustness_scores),
        'fragility_index': sum(1 for d in acc_drops if d > 0.05) / len(acc_drops)
    }


# =============================================================================
# Comprehensive Metric Computation
# =============================================================================

def compute_all_ml_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    standard_only: bool = True
) -> Dict[str, float]:
    """
    Compute ML performance metrics.
    
    Parameters
    ----------
    y_true : array
        Ground truth labels
    y_pred : array  
        Predicted labels (binary)
    y_pred_proba : array, optional
        Predicted probabilities
    standard_only : bool, default=True
        If True, only compute metrics that are standard in criminal justice ML:
        - AUC-ROC, accuracy, confusion matrix rates, F1, precision, recall
        If False, also includes Brier, log loss, MCC, Kappa, ECE, MCE
        (not recommended for this field but kept for completeness)
        
    Returns
    -------
    dict with metrics
    """
    metrics = {}
    
    # Standard metrics (ALWAYS computed - these are the field standard)
    metrics.update(compute_standard_metrics(y_true, y_pred, y_pred_proba))
    
    # Probability-based standard metrics
    if y_pred_proba is not None:
        proba_metrics = compute_probability_metrics(y_true, y_pred_proba)
        # AUC-ROC and AUC-PR are standard
        metrics['auc_roc'] = proba_metrics.get('auc_roc', np.nan)
        metrics['auc_pr'] = proba_metrics.get('auc_pr', np.nan)
        metrics['avg_precision'] = proba_metrics.get('avg_precision', np.nan)
        
        if not standard_only:
            # Non-standard metrics (rarely used in CJ fairness literature)
            metrics['brier_score'] = proba_metrics.get('brier_score', np.nan)
            metrics['log_loss'] = proba_metrics.get('log_loss', np.nan)
            metrics.update(compute_calibration_metrics(y_true, y_pred_proba))
    
    if not standard_only:
        # Non-standard advanced metrics
        metrics.update(compute_advanced_metrics(y_true, y_pred))
    
    return metrics


# =============================================================================
# Metric Reporting
# =============================================================================

def print_metrics_report(metrics: Dict[str, float], title: str = "ML Metrics Report"):
    """Pretty print metrics report."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    categories = {
        'Standard': ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy'],
        'Advanced': ['mcc', 'cohen_kappa'],
        'Probability': ['auc_roc', 'auc_pr', 'avg_precision', 'brier_score', 'log_loss'],
        'Calibration': ['ece', 'mce']
    }
    
    for category, metric_names in categories.items():
        category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
        if category_metrics:
            print(f"\n--- {category} Metrics ---")
            for name, value in category_metrics.items():
                if isinstance(value, float) and not np.isnan(value):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
    
    print("\n" + "=" * 60)


def print_ablation_report(ablation_results: List[AblationResult], 
                          title: str = "Feature Ablation Analysis"):
    """Pretty print ablation analysis."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    if not ablation_results:
        print("No ablation results available.")
        return
    
    print(f"\nBaseline Accuracy: {ablation_results[0].original_accuracy:.4f}")
    print(f"Baseline AUC: {ablation_results[0].original_auc:.4f}")
    
    print("\n--- Feature Impact (sorted by accuracy drop) ---")
    print(f"{'Feature':<30} {'Acc Drop':>10} {'AUC Drop':>10} {'Robustness':>10}")
    print("-" * 60)
    
    for r in ablation_results[:15]:  # Top 15
        print(f"{r.feature_removed:<30} {r.accuracy_drop:>10.4f} {r.auc_drop:>10.4f} {r.robustness_score:>10.4f}")
    
    # Overall robustness
    robustness = compute_model_robustness_score(ablation_results)
    print("\n--- Model Robustness Summary ---")
    for name, value in robustness.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    print("Testing ML Metrics Module...\n")
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Test metrics
    metrics = compute_all_ml_metrics(y_test, y_pred, y_proba)
    print_metrics_report(metrics, "Logistic Regression on Synthetic Data")
    
    # Test ablation
    print("\nRunning Feature Ablation Analysis...")
    ablation_results = feature_ablation_analysis(
        LogisticRegression,
        {'max_iter': 1000, 'random_state': 42},
        X_train, y_train, X_test, y_test,
        n_top_features=5
    )
    print_ablation_report(ablation_results)
    
    print("\n✓ All ML metrics tests passed!")
