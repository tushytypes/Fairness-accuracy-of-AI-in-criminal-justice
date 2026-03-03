"""
Main Experiment Runner for Fairness & Accuracy in Criminal Justice Thesis

Experiments:
1. Full model evaluation across datasets
2. Simple baseline comparison (Dressel & Farid replication)
3. Feature reduction analysis with tipping points
4. Feature ablation / ROAR analysis (multi-model)
5. Cross-validation robustness

Usage:
    python -m src.experiments.run_experiments --all --output results/
    python -m src.experiments.run_experiments --dataset compas --output results/
    python -m src.experiments.run_experiments --dataset ssl --output results/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from fairness.ml_metrics import (
    compute_all_ml_metrics, feature_ablation_analysis,
    compute_model_robustness_score
)
from fairness.fairness_metrics import (
    compute_all_fairness_metrics, compute_group_metrics
)
from fairness.feature_selection import (
    FeatureSelector, progressive_feature_analysis
)


# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42}
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    }
}

# TASK 1: SSL added to DATASETS
DATASETS = ['compas', 'communities_crime', 'ssl']

FEATURE_SELECTION_METHODS = ['mutual_information', 'random_forest', 'lasso']

# Metrics to include in the report (TASK 4)
REPORT_ML_METRICS = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1']
REPORT_FAIRNESS_METRICS = ['spd', 'eod', 'di', 'aod']


# =============================================================================
# Helpers
# =============================================================================

def _to_numpy(series):
    """Convert pandas Series to numpy array if needed."""
    return series.values if hasattr(series, 'values') else np.asarray(series)


def _get_baseline_features(dataset_name: str, data: Dict) -> List[str]:
    """
    Determine the 2 baseline features for a dataset.

    COMPAS / Communities use known features from the literature.
    SSL (and any unknown dataset) uses the top-2 Mutual Information features.
    """
    if dataset_name == 'compas':
        return ['age', 'priors_count']
    if dataset_name in ('communities_crime', 'communities'):
        return ['PctPopUnderPov', 'PctKids2Par']

    # --- SSL / fallback: top-2 MI features ---
    y_train = _to_numpy(data['y_train'])
    selector = FeatureSelector(method='mutual_information', n_features=2)
    selector.fit(data['X_train'], y_train)
    return selector.selected_features_


def _convert_numpy(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    return obj


# =============================================================================
# Experiment 1: Full Model Evaluation
# =============================================================================

def run_full_model_evaluation(data: Dict, models: Dict = MODELS) -> pd.DataFrame:
    """Evaluate all models on full feature set."""
    results = []

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = _to_numpy(data['y_train'])
    y_test = _to_numpy(data['y_test'])
    protected_test = _to_numpy(data['protected_test'])

    for model_name, model_config in models.items():
        print(f"\n  Training {model_name}...")

        model = model_config['class'](**model_config['params'])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # ML metrics (standard_only=True keeps field-standard metrics)
        ml_metrics = compute_all_ml_metrics(y_test, y_pred, y_proba)

        result = {
            'model': model_name,
            'n_features': X_train.shape[1],
        }
        result.update({f'ml_{k}': v for k, v in ml_metrics.items()})

        # Fairness metrics
        fairness = compute_all_fairness_metrics(y_test, y_pred, y_proba, protected_test)
        result.update({f'fairness_{k}': v for k, v in fairness.items()})

        # Group metrics
        group_metrics = compute_group_metrics(y_test, y_pred, y_proba, protected_test)
        for group, metrics in group_metrics.items():
            for metric_name, value in metrics.items():
                result[f'{group}_{metric_name}'] = value

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# Experiment 2: Simple Baseline (Dressel & Farid)
# =============================================================================

def run_simple_baseline(data: Dict, baseline_features: List[str]) -> Dict:
    """Evaluate simple 2-feature baseline."""
    available = [f for f in baseline_features if f in data['X_train'].columns]

    if len(available) < 2:
        warnings.warn(f"Not enough baseline features. Available: {data['X_train'].columns.tolist()}")
        return {}

    X_train = data['X_train'][available]
    X_test = data['X_test'][available]
    y_train = _to_numpy(data['y_train'])
    y_test = _to_numpy(data['y_test'])
    protected_test = _to_numpy(data['protected_test'])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    ml_metrics = compute_all_ml_metrics(y_test, y_pred, y_proba)
    fairness = compute_all_fairness_metrics(y_test, y_pred, y_proba, protected_test)

    result = {
        'model': 'simple_baseline',
        'features': available,
        'n_features': len(available),
    }
    result.update({f'ml_{k}': v for k, v in ml_metrics.items()})
    result.update({f'fairness_{k}': v for k, v in fairness.items()})

    return result


# =============================================================================
# Experiment 3: Feature Reduction Analysis
# =============================================================================

def run_feature_reduction_analysis(
    data: Dict,
    model_name: str = 'logistic_regression',
    selection_methods: List[str] = None,
    feature_counts: List[int] = None
) -> Dict[str, pd.DataFrame]:
    """Progressive feature reduction analysis."""
    if selection_methods is None:
        selection_methods = FEATURE_SELECTION_METHODS

    y_train = _to_numpy(data['y_train'])
    y_test = _to_numpy(data['y_test'])
    protected_test = _to_numpy(data['protected_test'])

    if feature_counts is None:
        n_total = data['X_train'].shape[1]
        feature_counts = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
        feature_counts = sorted(set([k for k in feature_counts if k <= n_total]))

    model_config = MODELS[model_name]
    results = {}

    for method in selection_methods:
        print(f"\n  Feature selection: {method}")

        def fairness_func(y_true, y_pred, protected):
            metrics = compute_all_fairness_metrics(y_true, y_pred, None, protected)
            return {
                'spd': metrics['spd'],
                'di': metrics['di'],
                'eod': metrics['eod'],
                'aod': metrics['aod']
            }

        df = progressive_feature_analysis(
            X_train=data['X_train'],
            y_train=y_train,
            X_test=data['X_test'],
            y_test=y_test,
            protected_test=protected_test,
            model_class=model_config['class'],
            model_params=model_config['params'],
            selection_method=method,
            feature_counts=feature_counts,
            fairness_func=fairness_func
        )

        results[method] = df

    return results


# =============================================================================
# Experiment 4: Feature Ablation / ROAR (multi-model)
# =============================================================================

def run_ablation_analysis(data: Dict, n_features: int = 15) -> Dict:
    """
    Run ROAR-style feature ablation with multiple models.

    Tests Logistic Regression and Random Forest to compare fragility.
    Integrated from run_roar_analysis.py (TASK 2).
    """
    roar_models = {
        'logistic_regression': MODELS['logistic_regression'],
        'random_forest': MODELS['random_forest'],
    }

    y_train = _to_numpy(data['y_train'])
    y_test = _to_numpy(data['y_test'])
    n_feat = min(n_features, data['X_train'].shape[1])

    all_results = {}

    for model_name, model_config in roar_models.items():
        print(f"\n  ROAR with {model_name}...")

        ablation_results = feature_ablation_analysis(
            model_config['class'],
            model_config['params'],
            data['X_train'],
            y_train,
            data['X_test'],
            y_test,
            n_top_features=n_feat
        )

        robustness = compute_model_robustness_score(ablation_results)

        ablation_list = [
            {
                'feature': r.feature_removed,
                'accuracy_drop': r.accuracy_drop,
                'auc_drop': r.auc_drop,
                'original_accuracy': r.original_accuracy,
                'ablated_accuracy': r.ablated_accuracy,
                'robustness_score': r.robustness_score
            }
            for r in ablation_results
        ]

        all_results[model_name] = {
            'baseline_accuracy': ablation_results[0].original_accuracy if ablation_results else None,
            'baseline_auc': ablation_results[0].original_auc if ablation_results else None,
            'robustness_metrics': robustness,
            'ablation_results': ablation_list
        }

        # Console summary
        if ablation_results:
            print(f"    Baseline accuracy: {ablation_results[0].original_accuracy:.2%}")
            print(f"    Mean drop: {robustness.get('mean_accuracy_drop', 0):.2%}")
            print(f"    Max drop: {robustness.get('max_accuracy_drop', 0):.2%}")
            print(f"    Fragility: {robustness.get('fragility_index', 0):.2%}")
            print(f"    Top 3 critical features:")
            for i, r in enumerate(ablation_results[:3]):
                print(f"      {i+1}. {r.feature_removed}: -{r.accuracy_drop:.2%}")

    return all_results


# =============================================================================
# Experiment 5: Cross-Validation
# =============================================================================

def run_cross_validation(data: Dict, n_folds: int = 5) -> Dict:
    """Run cross-validation for all models."""
    X = pd.concat([data['X_train'], data['X_test']])
    y = np.concatenate([_to_numpy(data['y_train']), _to_numpy(data['y_test'])])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for model_name, model_config in MODELS.items():
        model = model_config['class'](**model_config['params'])

        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        results[model_name] = {
            'accuracy_mean': acc_scores.mean(),
            'accuracy_std': acc_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std(),
            'accuracy_scores': acc_scores.tolist(),
            'auc_scores': auc_scores.tolist()
        }

    return results


# =============================================================================
# Tipping Point Analysis
# =============================================================================

def find_tipping_points(feature_reduction_results: pd.DataFrame,
                        accuracy_threshold: float = 0.02,
                        fairness_threshold: float = 0.1) -> Dict:
    """Identify tipping points where metrics degrade."""
    df = feature_reduction_results.copy()

    max_acc_idx = df['accuracy'].idxmax()
    max_accuracy = df.loc[max_acc_idx, 'accuracy']

    # Accuracy tipping point
    acc_threshold = max_accuracy - accuracy_threshold
    acc_tipping = df[df['accuracy'] < acc_threshold]
    acc_tipping_k = acc_tipping['n_features'].max() if len(acc_tipping) > 0 else df['n_features'].min()

    # Fairness tipping points
    fairness_cols = [c for c in df.columns if c in ['spd', 'eod', 'aod']]
    fairness_tipping = {}
    for col in fairness_cols:
        violations = df[df[col].abs() > fairness_threshold]
        if len(violations) > 0:
            fairness_tipping[col] = int(violations['n_features'].max())

    return {
        'max_accuracy': float(max_accuracy),
        'max_accuracy_k': int(df.loc[max_acc_idx, 'n_features']),
        'accuracy_tipping_point': int(acc_tipping_k),
        'fairness_tipping_points': fairness_tipping,
        'recommended_k': int(min(
            acc_tipping_k + 1,
            min(fairness_tipping.values()) + 1 if fairness_tipping else df['n_features'].max()
        ))
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_all_experiments(dataset_name: str, output_dir: str,
                        data_path: Optional[str] = None) -> Dict:
    """Run complete experiment suite for one dataset."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Load data
    print("\n[1/5] Loading dataset...")
    try:
        data = load_dataset(dataset_name, data_path=data_path)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return {}

    info = data['dataset_info']
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Base rate: {info['base_rate']:.3f}")

    results = {
        'dataset': dataset_name,
        'dataset_info': info,
        'timestamp': datetime.now().isoformat()
    }

    # Experiment 1: Full models
    print("\n[2/5] Full model evaluation...")
    full_results = run_full_model_evaluation(data)
    results['full_model_evaluation'] = full_results.to_dict('records')

    # Experiment 2: Baseline (top-2 MI for SSL)
    print("\n[3/5] Simple baseline...")
    baseline_features = _get_baseline_features(dataset_name, data)
    print(f"  Baseline features: {baseline_features}")
    baseline = run_simple_baseline(data, baseline_features)
    results['simple_baseline'] = baseline

    # Experiment 3: Feature reduction
    print("\n[4/5] Feature reduction analysis...")
    reduction_results = run_feature_reduction_analysis(data)

    results['feature_reduction'] = {}
    results['tipping_points'] = {}

    for method, df in reduction_results.items():
        df_serializable = df.copy()
        df_serializable['features'] = df_serializable['features'].apply(str)
        results['feature_reduction'][method] = df_serializable.to_dict('records')
        results['tipping_points'][method] = find_tipping_points(df)

    # Experiment 4: ROAR ablation (multi-model)
    print("\n[5/5] Feature ablation (ROAR) analysis...")
    ablation = run_ablation_analysis(data, n_features=min(15, data['X_train'].shape[1]))
    results['ablation_analysis'] = ablation

    # Experiment 5: Cross-validation
    print("\n[bonus] Cross-validation...")
    cv_results = run_cross_validation(data)
    results['cross_validation'] = cv_results

    # Save results JSON
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / f'{dataset_name}_results.json', 'w') as f:
        json.dump(_convert_numpy(results), f, indent=2)

    # Save CSVs
    full_results.to_csv(output_path / f'{dataset_name}_full_models.csv', index=False)

    for method, df in reduction_results.items():
        df_csv = df.drop(columns=['features'], errors='ignore')
        df_csv.to_csv(output_path / f'{dataset_name}_reduction_{method}.csv', index=False)

    print(f"\n  Results saved to {output_path}")

    return results


# =============================================================================
# TASK 3: Unified TXT Report Generation
# =============================================================================

def _fmt(value, fmt='.4f'):
    """Format a numeric value safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f'{value:{fmt}}'


def _get_ml(record: Dict, key: str):
    """Extract an ML metric from a result record (handles prefix variants)."""
    return record.get(f'ml_{key}', record.get(key, record.get(f'ml_{key}', None)))


def _get_fair(record: Dict, key: str):
    """Extract a fairness metric from a result record (handles prefix variants)."""
    # Try short names first, then long names used in old JSON format
    long_names = {
        'spd': 'statistical_parity_difference',
        'eod': 'equal_opportunity_difference',
        'di': 'disparate_impact',
        'aod': 'average_odds_difference',
    }
    val = record.get(f'fairness_{key}')
    if val is None and key in long_names:
        val = record.get(f'fairness_{long_names[key]}')
    return val


def generate_report(all_results: Dict[str, Dict], output_path: str):
    """
    Generate results/experiment_report.txt with the unified thesis report.

    Parameters
    ----------
    all_results : dict
        {dataset_name: results_dict} for each dataset that was run.
    output_path : str
        Path to the output TXT file.
    """
    lines = []

    def w(text=''):
        lines.append(text)

    sep = '=' * 80

    w(sep)
    w('THESIS EXPERIMENT REPORT')
    w(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    w(sep)

    # ── Per-dataset sections ────────────────────────────────────────────
    for ds_name, results in all_results.items():
        info = results.get('dataset_info', {})

        base_rate = info.get('base_rate', 0)
        g0_rate = info.get('base_rate_group_0', 0)
        g1_rate = info.get('base_rate_group_1', 0)
        gap = abs(g1_rate - g0_rate)

        w()
        w('-' * 80)
        w(f'DATASET: {info.get("name", ds_name).upper()}')
        w(f'Samples: {info.get("n_samples", "N/A")} | '
          f'Features: {info.get("n_features", "N/A")} | '
          f'Base rate: {base_rate:.0%} | '
          f'Base rate gap: {gap:.0%}')
        w('-' * 80)

        # ── Full model results ──────────────────────────────────────────
        full_models = results.get('full_model_evaluation', [])
        if full_models:
            w()
            w('FULL MODEL RESULTS')
            header = (f'{"Model":<25} {"Accuracy":>10} {"AUC-ROC":>10} '
                      f'{"Precision":>10} {"Recall":>10} {"F1":>10} '
                      f'{"SPD":>10} {"EOD":>10} {"DI":>10} {"AOD":>10}')
            w(header)
            w('-' * len(header))

            for r in full_models:
                w(f'{r["model"]:<25} '
                  f'{_fmt(_get_ml(r, "accuracy")):>10} '
                  f'{_fmt(_get_ml(r, "auc_roc")):>10} '
                  f'{_fmt(_get_ml(r, "precision")):>10} '
                  f'{_fmt(_get_ml(r, "recall")):>10} '
                  f'{_fmt(_get_ml(r, "f1")):>10} '
                  f'{_fmt(_get_fair(r, "spd")):>10} '
                  f'{_fmt(_get_fair(r, "eod")):>10} '
                  f'{_fmt(_get_fair(r, "di")):>10} '
                  f'{_fmt(_get_fair(r, "aod")):>10}')

        # ── Simple baseline ─────────────────────────────────────────────
        baseline = results.get('simple_baseline', {})
        if baseline:
            bl_acc = _get_ml(baseline, 'accuracy')
            bl_auc = _get_ml(baseline, 'auc_roc')
            bl_features = baseline.get('features', [])

            best_acc = max(
                (_get_ml(r, 'accuracy') or 0) for r in full_models
            ) if full_models else 0
            delta = ((bl_acc or 0) - best_acc) * 100

            w()
            w(f'SIMPLE BASELINE (Dressel & Farid)')
            w(f'  Features: {", ".join(bl_features)}')
            w(f'  Accuracy: {_fmt(bl_acc)} | AUC: {_fmt(bl_auc)} | '
              f'Delta vs best: {delta:+.2f}%')

        # ── Feature reduction ───────────────────────────────────────────
        tipping = results.get('tipping_points', {})
        if tipping:
            w()
            w('FEATURE REDUCTION')
            w(f'{"Method":<25} {"Optimal k":>12} {"Max Accuracy":>14}')
            w('-' * 55)
            for method, tp in tipping.items():
                w(f'{method:<25} {tp.get("max_accuracy_k", "N/A"):>12} '
                  f'{_fmt(tp.get("max_accuracy")):>14}')

        # ── ROAR ablation ───────────────────────────────────────────────
        ablation = results.get('ablation_analysis', {})
        if ablation:
            w()
            w('ROAR ABLATION')

            for model_name, model_abl in ablation.items():
                rob = model_abl.get('robustness_metrics', {})
                bl_acc_val = model_abl.get('baseline_accuracy')
                abl_list = model_abl.get('ablation_results', [])

                w(f'  {model_name}:')
                w(f'    Baseline: {_fmt(bl_acc_val, ".2%") if bl_acc_val else "N/A"} | '
                  f'Mean drop: {_fmt(rob.get("mean_accuracy_drop"), ".2%")} | '
                  f'Max drop: {_fmt(rob.get("max_accuracy_drop"), ".2%")} | '
                  f'Fragility: {_fmt(rob.get("fragility_index"), ".2%")}')

                if abl_list:
                    w(f'    Critical features:')
                    for i, feat in enumerate(abl_list[:3]):
                        w(f'      {i+1}. {feat["feature"]}: '
                          f'-{_fmt(feat["accuracy_drop"], ".2%")} accuracy')

        # ── Fairness summary ────────────────────────────────────────────
        if full_models:
            w()
            w('FAIRNESS')
            w(f'  Base rate gap: {gap:.2%}')

            # Use first model for threshold checks
            r0 = full_models[0]
            spd = _get_fair(r0, 'spd')
            eod = _get_fair(r0, 'eod')
            di = _get_fair(r0, 'di')
            aod = _get_fair(r0, 'aod')

            def _check(name, val, threshold, is_ratio=False):
                if val is None:
                    return f'{name}: N/A'
                if is_ratio:
                    ok = 0.8 <= val <= 1.25
                else:
                    ok = abs(val) < threshold
                status = 'PASS' if ok else 'FAIL'
                return f'{name}: {status} ({val:.2f})'

            checks = [
                _check('SPD', spd, 0.1),
                _check('EOD', eod, 0.1),
                _check('DI', di, 0, is_ratio=True),
                _check('AOD', aod, 0.1),
            ]
            w(f'  {" | ".join(checks)}')

        # ── Cross-validation ────────────────────────────────────────────
        cv = results.get('cross_validation', {})
        if cv:
            w()
            w('CROSS-VALIDATION (5-fold)')
            w(f'{"Model":<25} {"Acc (mean +/- std)":>22} {"AUC (mean +/- std)":>22}')
            w('-' * 72)
            for model_name, cv_res in cv.items():
                acc_str = f'{cv_res["accuracy_mean"]:.4f} +/- {cv_res["accuracy_std"]:.4f}'
                auc_str = f'{cv_res["auc_mean"]:.4f} +/- {cv_res["auc_std"]:.4f}'
                w(f'{model_name:<25} {acc_str:>22} {auc_str:>22}')

    # ── Cross-dataset comparison ────────────────────────────────────────
    if len(all_results) > 1:
        w()
        w(sep)
        w('CROSS-DATASET COMPARISON')
        w(sep)

        header = (f'{"Dataset":<20} {"Best Acc":>10} {"Baseline":>10} '
                  f'{"Delta":>10} {"Fragility":>10} {"Gap":>10}')
        w(header)
        w('-' * len(header))

        for ds_name, results in all_results.items():
            info = results.get('dataset_info', {})
            full_models = results.get('full_model_evaluation', [])
            baseline = results.get('simple_baseline', {})
            ablation = results.get('ablation_analysis', {})

            best_acc = max((_get_ml(r, 'accuracy') or 0) for r in full_models) if full_models else 0
            bl_acc = _get_ml(baseline, 'accuracy') or 0
            delta = (bl_acc - best_acc) * 100

            # Use logistic regression fragility (primary model)
            lr_abl = ablation.get('logistic_regression', {})
            fragility = lr_abl.get('robustness_metrics', {}).get('fragility_index', 0)

            g0 = info.get('base_rate_group_0', 0)
            g1 = info.get('base_rate_group_1', 0)
            gap = abs(g1 - g0)

            ds_label = info.get('name', ds_name)
            w(f'{ds_label:<20} {best_acc:>10.2%} {bl_acc:>10.2%} '
              f'{delta:>+10.2f}% {fragility:>10.2%} {gap:>10.0%}')

    # ── Key findings ────────────────────────────────────────────────────
    w()
    w(sep)
    w('KEY FINDINGS')
    w(sep)

    finding_num = 1

    # Check Dressel & Farid for COMPAS
    if 'compas' in all_results:
        compas = all_results['compas']
        bl = compas.get('simple_baseline', {})
        fm = compas.get('full_model_evaluation', [])
        if bl and fm:
            bl_acc = _get_ml(bl, 'accuracy') or 0
            best_acc = max((_get_ml(r, 'accuracy') or 0) for r in fm)
            if abs(bl_acc - best_acc) < 0.02:
                w(f'{finding_num}. Dressel & Farid confirmed for COMPAS '
                  f'(baseline {bl_acc:.2%} vs best {best_acc:.2%})')
            else:
                w(f'{finding_num}. Dressel & Farid NOT confirmed for COMPAS '
                  f'(baseline {bl_acc:.2%} vs best {best_acc:.2%})')
            finding_num += 1

    # Check Communities complexity benefit
    if 'communities_crime' in all_results:
        comm = all_results['communities_crime']
        bl = comm.get('simple_baseline', {})
        fm = comm.get('full_model_evaluation', [])
        if bl and fm:
            bl_acc = _get_ml(bl, 'accuracy') or 0
            best_acc = max((_get_ml(r, 'accuracy') or 0) for r in fm)
            delta = (best_acc - bl_acc) * 100
            if delta > 2:
                w(f'{finding_num}. Communities benefits from complexity (+{delta:.1f}%)')
            finding_num += 1

    # COMPAS critical feature
    if 'compas' in all_results:
        abl = all_results['compas'].get('ablation_analysis', {})
        lr_abl = abl.get('logistic_regression', {}).get('ablation_results', [])
        if lr_abl:
            top_feat = lr_abl[0]
            w(f'{finding_num}. {top_feat["feature"]} is COMPAS single point of failure '
              f'(-{top_feat["accuracy_drop"]:.2%})')
            finding_num += 1

    # Fairness violations
    all_violate = True
    for ds_name, results in all_results.items():
        for r in results.get('full_model_evaluation', []):
            spd = _get_fair(r, 'spd')
            eod = _get_fair(r, 'eod')
            if spd is not None and eod is not None:
                if abs(spd) < 0.1 and abs(eod) < 0.1:
                    all_violate = False

    if all_violate and all_results:
        w(f'{finding_num}. All models violate fairness thresholds')
        finding_num += 1

    # Base rate gap correlation
    gaps_and_violations = []
    for ds_name, results in all_results.items():
        info = results.get('dataset_info', {})
        g0 = info.get('base_rate_group_0', 0)
        g1 = info.get('base_rate_group_1', 0)
        gap = abs(g1 - g0)
        fm = results.get('full_model_evaluation', [])
        if fm:
            avg_spd = np.mean([abs(_get_fair(r, 'spd') or 0) for r in fm])
            gaps_and_violations.append((gap, avg_spd))

    if len(gaps_and_violations) > 1:
        sorted_gaps = sorted(gaps_and_violations)
        if sorted_gaps[-1][1] > sorted_gaps[0][1]:
            w(f'{finding_num}. Base rate gap predicts fairness violations')
            finding_num += 1

    w()
    w(sep)

    # ── Write to file ───────────────────────────────────────────────────
    report_text = '\n'.join(lines)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  Report written to {output_path}")
    return report_text


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Fairness & Accuracy experiments")
    parser.add_argument('--dataset', type=str, choices=['compas', 'communities_crime', 'ssl'])
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--output', type=str, default='results/')
    parser.add_argument('--data-path', type=str, default=None)

    args = parser.parse_args()

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    all_results = {}
    for dataset in datasets:
        results = run_all_experiments(dataset, args.output, args.data_path)
        if results:
            all_results[dataset] = results

    # Generate unified TXT report (TASK 3)
    if all_results:
        report_path = os.path.join(args.output, 'experiment_report.txt')
        generate_report(all_results, report_path)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
