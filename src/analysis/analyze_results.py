"""
Analysis Module for Thesis Results

Analyzes existing experiment results and generates:
1. Summary statistics
2. Tipping point analysis
3. Comparison tables
4. Key findings extraction
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def load_results(results_path: str) -> Dict:
    """Load results JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_full_models(results: Dict) -> pd.DataFrame:
    """Analyze full model evaluation results."""
    records = []
    for model_result in results.get('full_model_evaluation', []):
        record = {
            'Model': model_result['model'],
            'Features': model_result['n_features'],
            'Accuracy': model_result.get('ml_accuracy', model_result.get('accuracy', np.nan)),
            'AUC': model_result.get('ml_auc_roc', model_result.get('auc', np.nan)),
            'Precision': model_result.get('ml_precision', model_result.get('precision', np.nan)),
            'Recall': model_result.get('ml_recall', model_result.get('recall', np.nan)),
            'F1': model_result.get('ml_f1', model_result.get('f1', np.nan)),
            'SPD': model_result.get('fairness_spd',
                                    model_result.get('fairness_statistical_parity_difference', np.nan)),
            'EOD': model_result.get('fairness_eod',
                                    model_result.get('fairness_equal_opportunity_difference', np.nan)),
            'AOD': model_result.get('fairness_aod',
                                    model_result.get('fairness_average_odds_difference', np.nan)),
            'DI': model_result.get('fairness_di',
                                   model_result.get('fairness_disparate_impact', np.nan)),
        }
        records.append(record)
    return pd.DataFrame(records)


def analyze_baseline_comparison(results: Dict) -> Dict:
    """Compare baseline with full models."""
    baseline = results.get('simple_baseline', {})
    full_models = results.get('full_model_evaluation', [])
    
    if not baseline or not full_models:
        return {}
    
    baseline_acc = baseline.get('ml_accuracy', baseline.get('accuracy', 0))
    baseline_auc = baseline.get('ml_auc_roc', baseline.get('auc', 0))
    
    comparisons = []
    for model in full_models:
        model_acc = model.get('ml_accuracy', model.get('accuracy', 0))
        model_auc = model.get('ml_auc_roc', model.get('auc', 0))
        
        comparisons.append({
            'model': model['model'],
            'accuracy_diff': model_acc - baseline_acc,
            'auc_diff': model_auc - baseline_auc,
            'baseline_matches': abs(model_acc - baseline_acc) < 0.02
        })
    
    return {
        'baseline_features': baseline.get('features', []),
        'baseline_accuracy': baseline_acc,
        'baseline_auc': baseline_auc,
        'comparisons': comparisons,
        'dressel_farid_confirmed': all(c['baseline_matches'] for c in comparisons)
    }


def analyze_feature_reduction(results: Dict) -> Dict:
    """Analyze feature reduction experiments."""
    reduction_data = results.get('feature_reduction', {})
    tipping_points = results.get('tipping_points', {})
    
    analysis = {}
    
    for method, data in reduction_data.items():
        df = pd.DataFrame(data)
        
        # Find optimal k
        max_acc_idx = df['accuracy'].idxmax()
        max_acc = df.loc[max_acc_idx, 'accuracy']
        optimal_k = df.loc[max_acc_idx, 'n_features']
        
        # Analyze fairness trend
        if 'aod' in df.columns:
            fairness_at_optimal = df.loc[max_acc_idx, 'aod']
            fairness_at_min = df.loc[df['n_features'].idxmin(), 'aod']
        else:
            fairness_at_optimal = np.nan
            fairness_at_min = np.nan
        
        analysis[method] = {
            'optimal_k': optimal_k,
            'max_accuracy': max_acc,
            'min_features_tested': df['n_features'].min(),
            'max_features_tested': df['n_features'].max(),
            'accuracy_at_min_features': df.loc[df['n_features'].idxmin(), 'accuracy'],
            'accuracy_drop_at_min': max_acc - df.loc[df['n_features'].idxmin(), 'accuracy'],
            'fairness_at_optimal': fairness_at_optimal,
            'fairness_at_min': fairness_at_min,
            'tipping_point': tipping_points.get(method, {})
        }
    
    return analysis


def analyze_group_disparities(results: Dict) -> Dict:
    """Analyze disparities between protected groups."""
    full_models = results.get('full_model_evaluation', [])
    
    if not full_models:
        return {}
    
    disparities = []
    for model in full_models:
        g0_base = model.get('group_0_base_rate', 0)
        g1_base = model.get('group_1_base_rate', 0)
        g0_tpr = model.get('group_0_tpr', 0)
        g1_tpr = model.get('group_1_tpr', 0)
        g0_fpr = model.get('group_0_fpr', 0)
        g1_fpr = model.get('group_1_fpr', 0)
        
        disparities.append({
            'model': model['model'],
            'base_rate_diff': g1_base - g0_base,
            'tpr_diff': g1_tpr - g0_tpr,
            'fpr_diff': g1_fpr - g0_fpr,
            'accuracy_group_0': model.get('group_0_accuracy', 0),
            'accuracy_group_1': model.get('group_1_accuracy', 0),
            'accuracy_diff': model.get('group_1_accuracy', 0) - model.get('group_0_accuracy', 0)
        })
    
    return {
        'base_rate_gap': disparities[0]['base_rate_diff'] if disparities else 0,
        'model_disparities': disparities
    }


def generate_key_findings(results: Dict, dataset_name: str) -> List[str]:
    """Generate key findings from results."""
    findings = []
    
    info = results.get('dataset_info', {})
    findings.append(f"Dataset: {dataset_name} ({info.get('n_samples', 'N/A')} samples, {info.get('n_features', 'N/A')} features)")
    
    # Base rate finding
    base_rate = info.get('base_rate', 0)
    findings.append(f"Base rate: {base_rate:.1%}")
    
    # Model accuracy finding
    full_models = results.get('full_model_evaluation', [])
    if full_models:
        best_model = max(full_models, key=lambda x: x.get('ml_accuracy', x.get('accuracy', 0)))
        best_acc = best_model.get('ml_accuracy', best_model.get('accuracy', 0))
        findings.append(f"Best model accuracy: {best_acc:.1%} ({best_model['model']})")
    
    # Baseline comparison
    baseline = results.get('simple_baseline', {})
    if baseline:
        baseline_acc = baseline.get('ml_accuracy', baseline.get('accuracy', 0))
        findings.append(f"Simple baseline (2 features): {baseline_acc:.1%} accuracy")

        if full_models:
            best_acc = best_model.get('ml_accuracy', best_model.get('accuracy', 0))
            if abs(best_acc - baseline_acc) < 0.02:
                findings.append("✓ CONFIRMED: Simple baseline matches complex models (Dressel & Farid)")
    
    # Fairness findings
    if full_models:
        model = full_models[0]
        spd = model.get('fairness_spd',
                        model.get('fairness_statistical_parity_difference', 0))
        eod = model.get('fairness_eod',
                        model.get('fairness_equal_opportunity_difference', 0))
        
        findings.append(f"Fairness violations: SPD={spd:.3f}, EOD={eod:.3f}")
        
        if abs(spd) > 0.1 or abs(eod) > 0.1:
            findings.append("✗ ALL models violate fairness thresholds (|SPD|, |EOD| > 0.1)")
    
    # Tipping point
    tipping = results.get('tipping_points', {})
    if tipping:
        for method, tp in tipping.items():
            findings.append(f"Optimal feature count ({method}): k={tp.get('max_accuracy_k', 'N/A')}")
    
    return findings


def print_analysis_report(results: Dict, dataset_name: str):
    """Print comprehensive analysis report."""
    print("\n" + "="*70)
    print(f"ANALYSIS REPORT: {dataset_name.upper()}")
    print("="*70)
    
    # Key findings
    print("\n--- KEY FINDINGS ---")
    findings = generate_key_findings(results, dataset_name)
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")
    
    # Full model comparison
    print("\n--- FULL MODEL COMPARISON ---")
    df = analyze_full_models(results)
    if not df.empty:
        print(df.to_string(index=False))
    
    # Baseline comparison
    print("\n--- BASELINE COMPARISON (Dressel & Farid) ---")
    baseline = analyze_baseline_comparison(results)
    if baseline:
        print(f"  Baseline features: {baseline['baseline_features']}")
        print(f"  Baseline accuracy: {baseline['baseline_accuracy']:.4f}")
        print(f"  Baseline AUC: {baseline['baseline_auc']:.4f}")
        for c in baseline['comparisons']:
            status = "✓ MATCHES" if c['baseline_matches'] else "≠ DIFFERS"
            print(f"  vs {c['model']}: acc diff={c['accuracy_diff']:+.4f} {status}")
    
    # Feature reduction
    print("\n--- FEATURE REDUCTION ANALYSIS ---")
    reduction = analyze_feature_reduction(results)
    for method, data in reduction.items():
        print(f"\n  {method.upper()}:")
        print(f"    Optimal k: {data['optimal_k']} features")
        print(f"    Max accuracy: {data['max_accuracy']:.4f}")
        print(f"    Accuracy at k=2: {data['accuracy_at_min_features']:.4f}")
        print(f"    Accuracy drop at k=2: {data['accuracy_drop_at_min']:.4f}")
    
    # Group disparities
    print("\n--- GROUP DISPARITIES ---")
    disparities = analyze_group_disparities(results)
    if disparities:
        print(f"  Base rate gap: {disparities['base_rate_gap']:.1%}")
        for d in disparities['model_disparities'][:3]:
            print(f"  {d['model']}: TPR diff={d['tpr_diff']:.3f}, FPR diff={d['fpr_diff']:.3f}")
    
    print("\n" + "="*70)


def compare_datasets(results_list: List[Dict], names: List[str]) -> pd.DataFrame:
    """Compare results across multiple datasets."""
    records = []
    
    for results, name in zip(results_list, names):
        info = results.get('dataset_info', {})
        
        # Get best model
        full_models = results.get('full_model_evaluation', [])
        best_model = max(full_models, key=lambda x: x.get('ml_accuracy', x.get('accuracy', 0))) if full_models else {}

        # Get baseline
        baseline = results.get('simple_baseline', {})

        record = {
            'Dataset': name,
            'Samples': info.get('n_samples', 0),
            'Features': info.get('n_features', 0),
            'Base Rate': info.get('base_rate', 0),
            'Best Accuracy': best_model.get('ml_accuracy', best_model.get('accuracy', 0)),
            'Best AUC': best_model.get('ml_auc_roc', best_model.get('auc', 0)),
            'Baseline Accuracy': baseline.get('ml_accuracy', baseline.get('accuracy', 0)),
            'SPD': best_model.get('fairness_spd',
                                  best_model.get('fairness_statistical_parity_difference', 0)),
            'EOD': best_model.get('fairness_eod',
                                  best_model.get('fairness_equal_opportunity_difference', 0)),
        }
        records.append(record)
    
    return pd.DataFrame(records)


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    # Load available results
    results_dir = Path(__file__).parent.parent / "experiments" / "results"

    all_results = []
    all_names = []

    for ds_name, label in [("compas", "COMPAS"),
                           ("communities_crime", "Communities & Crime"),
                           ("ssl", "Chicago SSL")]:
        path = results_dir / f"{ds_name}_results.json"
        if path.exists():
            print(f"Loading {label} results...")
            data = load_results(path)
            print_analysis_report(data, label)
            all_results.append(data)
            all_names.append(label)

    # Cross-dataset comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-DATASET COMPARISON")
        print("=" * 70)

        comparison = compare_datasets(all_results, all_names)
        print(comparison.to_string(index=False))
