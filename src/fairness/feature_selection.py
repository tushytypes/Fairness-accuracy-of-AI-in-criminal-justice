"""
Feature Selection Methods for Criminal Justice ML

Methods:
- Filter: Mutual Information, Correlation, Variance
- Embedded: LASSO (L1), Random Forest Importance
- Wrapper: Forward Selection, Backward Elimination, RFE
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, RFE,
    SequentialFeatureSelector
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Callable
import warnings


class FeatureSelector:
    """Unified feature selection interface."""
    
    def __init__(self, method: str = 'mutual_information', n_features: int = 10):
        """
        Parameters
        ----------
        method : str
            Selection method: 'mutual_information', 'random_forest', 'lasso',
            'correlation', 'variance', 'forward', 'backward', 'rfe'
        n_features : int
            Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selector_ = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureSelector':
        """Fit the selector and identify top features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = np.asarray(y).ravel()
        
        method_map = {
            'mutual_information': self._fit_mutual_info,
            'mi': self._fit_mutual_info,
            'random_forest': self._fit_random_forest,
            'rf': self._fit_random_forest,
            'lasso': self._fit_lasso,
            'l1': self._fit_lasso,
            'correlation': self._fit_correlation,
            'corr': self._fit_correlation,
            'variance': self._fit_variance,
            'var': self._fit_variance,
            'forward': self._fit_forward,
            'forward_selection': self._fit_forward,
            'backward': self._fit_backward,
            'backward_elimination': self._fit_backward,
            'rfe': self._fit_rfe,
        }
        
        if self.method.lower() not in method_map:
            raise ValueError(f"Unknown method: {self.method}")
        
        method_map[self.method.lower()](X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from X."""
        if self.selected_features_ is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _fit_mutual_info(self, X: pd.DataFrame, y: np.ndarray):
        """Mutual Information selection."""
        scores = mutual_info_classif(X, y, random_state=42)
        self.feature_scores_ = dict(zip(X.columns, scores))
        top_indices = np.argsort(scores)[-self.n_features:][::-1]
        self.selected_features_ = X.columns[top_indices].tolist()
    
    def _fit_random_forest(self, X: pd.DataFrame, y: np.ndarray):
        """Random Forest importance selection."""
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        scores = rf.feature_importances_
        self.feature_scores_ = dict(zip(X.columns, scores))
        top_indices = np.argsort(scores)[-self.n_features:][::-1]
        self.selected_features_ = X.columns[top_indices].tolist()
    
    def _fit_lasso(self, X: pd.DataFrame, y: np.ndarray):
        """LASSO (L1) selection."""
        # Scale for LASSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try different alphas to get desired number of features
        for alpha in np.logspace(-4, 1, 50):
            lasso = LogisticRegression(penalty='l1', solver='saga', C=1/alpha,
                                       max_iter=1000, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso.fit(X_scaled, y)
            
            n_selected = np.sum(np.abs(lasso.coef_[0]) > 1e-6)
            if n_selected <= self.n_features and n_selected > 0:
                break
        
        scores = np.abs(lasso.coef_[0])
        self.feature_scores_ = dict(zip(X.columns, scores))
        top_indices = np.argsort(scores)[-self.n_features:][::-1]
        self.selected_features_ = X.columns[top_indices].tolist()
    
    def _fit_correlation(self, X: pd.DataFrame, y: np.ndarray):
        """Correlation-based selection."""
        correlations = X.apply(lambda x: np.abs(np.corrcoef(x, y)[0, 1]))
        correlations = correlations.fillna(0)
        self.feature_scores_ = correlations.to_dict()
        self.selected_features_ = correlations.nlargest(self.n_features).index.tolist()
    
    def _fit_variance(self, X: pd.DataFrame, y: np.ndarray):
        """Variance-based selection (unsupervised)."""
        variances = X.var()
        self.feature_scores_ = variances.to_dict()
        self.selected_features_ = variances.nlargest(self.n_features).index.tolist()
    
    def _fit_forward(self, X: pd.DataFrame, y: np.ndarray):
        """Forward sequential selection."""
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        sfs = SequentialFeatureSelector(
            estimator, n_features_to_select=min(self.n_features, X.shape[1]),
            direction='forward', cv=3, n_jobs=-1
        )
        sfs.fit(X, y)
        self.selected_features_ = X.columns[sfs.get_support()].tolist()
        self.feature_scores_ = {f: 1 if f in self.selected_features_ else 0 
                                for f in X.columns}
    
    def _fit_backward(self, X: pd.DataFrame, y: np.ndarray):
        """Backward sequential selection."""
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        sfs = SequentialFeatureSelector(
            estimator, n_features_to_select=min(self.n_features, X.shape[1]),
            direction='backward', cv=3, n_jobs=-1
        )
        sfs.fit(X, y)
        self.selected_features_ = X.columns[sfs.get_support()].tolist()
        self.feature_scores_ = {f: 1 if f in self.selected_features_ else 0 
                                for f in X.columns}
    
    def _fit_rfe(self, X: pd.DataFrame, y: np.ndarray):
        """Recursive Feature Elimination."""
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        rfe = RFE(estimator, n_features_to_select=min(self.n_features, X.shape[1]))
        rfe.fit(X, y)
        self.selected_features_ = X.columns[rfe.get_support()].tolist()
        self.feature_scores_ = dict(zip(X.columns, rfe.ranking_))


def progressive_feature_analysis(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    protected_test: np.ndarray,
    model_class,
    model_params: dict,
    selection_method: str = 'mutual_information',
    feature_counts: List[int] = None,
    fairness_func: Callable = None
) -> pd.DataFrame:
    """
    Progressive feature reduction analysis.
    
    Track accuracy and fairness as features are reduced.
    """
    if feature_counts is None:
        n_total = X_train.shape[1]
        feature_counts = [2, 3, 5, 7, 10, 15, 20, 30, 50]
        feature_counts = [k for k in feature_counts if k < n_total] + [n_total]
        feature_counts = sorted(set(feature_counts))
    
    results = []
    
    for k in feature_counts:
        if k > X_train.shape[1]:
            continue
        
        # Select features
        selector = FeatureSelector(method=selection_method, n_features=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train_selected, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        result = {
            'n_features': k,
            'features': selector.selected_features_,
            'accuracy': accuracy_score(y_test, y_pred),
        }
        
        if y_proba is not None:
            try:
                result['auc'] = roc_auc_score(y_test, y_proba)
            except:
                result['auc'] = np.nan
        
        # Fairness metrics
        if fairness_func is not None:
            fairness = fairness_func(y_test, y_pred, protected_test)
            result.update(fairness)
        
        results.append(result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("Testing Feature Selection Module...\n")
    
    # Create synthetic data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
    
    methods = ['mutual_information', 'random_forest', 'lasso', 'correlation', 'variance']
    
    for method in methods:
        print(f"\n{method.upper()}:")
        selector = FeatureSelector(method=method, n_features=5)
        selector.fit(X, y)
        print(f"  Selected: {selector.selected_features_}")
    
    print("\n✓ Feature selection tests passed!")
