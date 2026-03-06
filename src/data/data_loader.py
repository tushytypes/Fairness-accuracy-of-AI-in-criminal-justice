"""
Data Loaders for Criminal Justice Datasets.

Supports three criminal justice domains:
- COMPAS (recidivism prediction) — downloaded from ProPublica GitHub
- Communities & Crime (predictive policing) — downloaded from UCI ML Repository
- Chicago SSL (victimisation prediction) — local CSV or Socrata API auto-download

Each loader returns a standardised dict:
    X_train, X_test, y_train, y_test, protected_train, protected_test,
    feature_names, protected_attribute, dataset_info
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Optional, Tuple, List
import warnings
import os


# =============================================================================
# COMPAS Dataset Loader
# =============================================================================

def load_compas(data_path: Optional[str] = None, test_size: float = 0.2, 
                random_state: int = 42) -> Dict:
    """
    Load and preprocess COMPAS dataset following ProPublica methodology.
    
    Returns dict with X_train, X_test, y_train, y_test, protected attributes.
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Download from ProPublica GitHub
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        df = pd.read_csv(url)
    
    # ProPublica filtering criteria
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ]
    
    # Filter to African-American and Caucasian only (standard practice)
    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    
    # Features
    features = [
        'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 
        'juv_other_count', 'c_charge_degree', 'sex', 'days_b_screening_arrest'
    ]
    
    X = df[features].copy()
    y = df['two_year_recid'].copy()
    
    # Encode categorical
    X['c_charge_degree'] = (X['c_charge_degree'] == 'F').astype(int)
    X['sex'] = (X['sex'] == 'Male').astype(int)
    
    # Protected attribute: race (1 = African-American, 0 = Caucasian)
    protected = (df['race'] == 'African-American').astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize numeric features
    scaler = StandardScaler()
    numeric_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 
                    'juv_other_count', 'days_b_screening_arrest']
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'protected_train': prot_train,
        'protected_test': prot_test,
        'feature_names': features,
        'protected_attribute': 'race',
        'dataset_info': {
            'name': 'COMPAS',
            'n_samples': len(df),
            'n_features': len(features),
            'base_rate': y.mean(),
            'base_rate_group_0': y[protected == 0].mean(),
            'base_rate_group_1': y[protected == 1].mean(),
            'group_0_name': 'Caucasian',
            'group_1_name': 'African-American'
        }
    }


# =============================================================================
# Communities and Crime Dataset Loader
# =============================================================================

def load_communities_crime(data_path: Optional[str] = None, test_size: float = 0.2,
                           random_state: int = 42) -> Dict:
    """
    Load UCI Communities and Crime dataset.
    Binarizes target at median for classification.
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Download from UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
        names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names"
        
        # Column names (from documentation)
        columns = [
            'state', 'county', 'community', 'communityname', 'fold',
            'population', 'householdsize', 'racepctblack', 'racePctWhite', 
            'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29',
            'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
            'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
            'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap',
            'indianPerCap', 'AssianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov',
            'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
            'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ',
            'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr',
            'FesalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
            'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
            'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
            'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
            'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
            'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous',
            'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR',
            'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded',
            'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
            'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian',
            'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
            'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn',
            'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
            'LessFamHh', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
            'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'ViolentCrimesPerPop'
        ]
        
        df = pd.read_csv(url, names=columns, na_values='?')
    
    # Drop non-predictive columns
    drop_cols = ['state', 'county', 'community', 'communityname', 'fold']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Target variable
    target_col = 'ViolentCrimesPerPop'
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Drop columns with >20% missing
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.2].index.tolist()
    df = df.drop(columns=cols_to_drop)
    
    # Drop remaining rows with missing values
    df = df.dropna()
    
    # Separate features and target
    y_continuous = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Binarize target at median
    y = (y_continuous > y_continuous.median()).astype(int)
    
    # Protected attribute: racepctblack binarized at median
    if 'racepctblack' in X.columns:
        protected = (X['racepctblack'] > X['racepctblack'].median()).astype(int)
    else:
        protected = pd.Series(0, index=X.index)
    
    feature_names = X.columns.tolist()
    
    # Train/test split
    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize all features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'protected_train': prot_train,
        'protected_test': prot_test,
        'feature_names': feature_names,
        'protected_attribute': 'racepctblack_binary',
        'dataset_info': {
            'name': 'Communities and Crime',
            'n_samples': len(df),
            'n_features': len(feature_names),
            'base_rate': y.mean(),
            'base_rate_group_0': y[protected == 0].mean(),
            'base_rate_group_1': y[protected == 1].mean(),
            'group_0_name': 'Low % Black',
            'group_1_name': 'High % Black'
        }
    }


# =============================================================================
# Chicago SSL Dataset Loader (local CSV or Socrata API)
# =============================================================================

# Default path for the local SSL Historical CSV (next to this module)
_SSL_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__),
    'Strategic_Subject_List_-_Historical_20260302.csv'
)

# Socrata API endpoint for the Historical SSL dataset
_SSL_SOCRATA_DOMAIN = 'data.cityofchicago.org'
_SSL_SOCRATA_DATASET = '4aki-r3np'

# Age-range ordinal mapping used for age_curr / predictor_rat_age_at_latest_arrest
_AGE_RANGE_MAP = {
    'less than 20': 0,
    '20-30': 1,
    '30-40': 2,
    '40-50': 3,
    '50-60': 4,
    '60-70': 5,
    '70-80': 6,
}


def load_ssl(data_path: Optional[str] = None, test_size: float = 0.2,
             random_state: int = 42, score_threshold: int = 300,
             limit: int = 400_000, verbose: bool = True) -> Dict:
    """
    Load and preprocess the Chicago Strategic Subject List (Historical).

    Data source resolution (in order):
    1. *data_path* if explicitly provided and file exists.
    2. Default local CSV next to this module
       (``src/data/Strategic_Subject_List_-_Historical_20260302.csv``).
    3. **Automatic download** via Socrata Open Data API (dataset 4aki-r3np).
       Requires ``sodapy`` (``pip install sodapy``).

    Features kept (7 total after encoding)
    ----------------------------------------
    Ordinal (2 — age-range mapped to 0-6):
        age_at_arrest_ord, age_curr_ord

    Binary flags (3 — Y/N to 1/0):
        weapon_flag, drug_flag, cpd_arrest_flag

    Binary demographic (2):
        sex_male  (M=1, else=0)
        latest_year_recent  (LATEST DATE >= 2014)

    NOTE: The 7 ``PREDICTOR RAT *`` columns (victim_shooting, victim_battery,
    arrests_violent, gang_affiliation, narcotic_arrests, trend_criminal,
    uuw_arrests) are **excluded** because they are the sub-scores that the
    CPD algorithm sums to compute SSL SCORE.  Including them as features
    when predicting ``SSL SCORE >= threshold`` constitutes structural data
    leakage: the target is a near-deterministic function of those inputs.
    Only raw individual characteristics independent of the CPD formula
    are used.

    Target
    ------
    ``SSL SCORE >= score_threshold``  (default 300 -> high risk, ~43 % positive)

    Protected attribute
    -------------------
    ``RACE CODE CD == 'BLK'``  ->  1 (Black), 0 (Non-Black)

    Parameters
    ----------
    data_path : str, optional
        Path to a local SSL CSV.  When ``None`` the loader tries the
        default local file, then falls back to the Socrata API.
    test_size : float
        Fraction reserved for the test set (default 0.2).
    random_state : int
        Random seed for reproducibility.
    score_threshold : int
        Binarisation cut-off for SSL SCORE (default 300).
        300 -> ~43 % positive (balanced).
        400 -> ~1 % positive  (very imbalanced — not recommended).
    limit : int
        Maximum number of records to fetch from the Socrata API
        (ignored when reading from a local file).
    verbose : bool
        Print loading diagnostics.

    Returns
    -------
    dict
        Standard loader output: X_train, X_test, y_train, y_test,
        protected_train, protected_test, feature_names, dataset_info.
    """
    # --- resolve data source --------------------------------------------
    df = None

    # 1. Explicit path
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path, low_memory=False)
        if verbose:
            print(f"SSL loaded from explicit path: {data_path}")

    # 2. Default local file
    if df is None and os.path.exists(_SSL_DEFAULT_PATH):
        df = pd.read_csv(_SSL_DEFAULT_PATH, low_memory=False)
        if verbose:
            print(f"SSL loaded from default local file")

    # 3. Socrata API fallback
    if df is None:
        try:
            from sodapy import Socrata
        except ImportError:
            raise ImportError(
                "SSL dataset not found locally and sodapy is not installed.\n"
                "Either place the CSV next to this module or install sodapy:\n"
                "  pip install sodapy"
            )
        if verbose:
            print(f"Downloading SSL via Socrata API (limit={limit:,})...")
        client = Socrata(_SSL_SOCRATA_DOMAIN, None)  # no app token (rate-limited)
        records = client.get(_SSL_SOCRATA_DATASET, limit=limit)
        df = pd.DataFrame.from_records(records)
        if verbose:
            print(f"Downloaded {len(df):,} records from Socrata")
        # Socrata returns strings — cast numeric columns where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    if verbose:
        print(f"SSL raw: {len(df):,} rows, {len(df.columns)} columns")

    # --- normalise column names to lowercase_underscore -------------------
    # Socrata API returns lowercase_underscore names; local CSV uses
    # UPPERCASE WITH SPACES.  Normalise to a single convention.
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # NOTE: The 7 'predictor_rat_*' columns are excluded because they are the
    # sub-scores that the CPD algorithm sums to compute ssl_score. Including
    # them as features when predicting ssl_score >= threshold constitutes
    # structural data leakage: the target is a near-deterministic function of
    # those inputs.  Only raw individual characteristics independent of the
    # CPD formula are used.

    # ── 1. Ordinal age ranges → 0-6 ────────────────────────────────────
    df['age_at_arrest_ord'] = (
        df['predictor_rat_age_at_latest_arrest']
        .astype(str).str.strip().str.lower()
        .map(_AGE_RANGE_MAP)
    )
    df['age_curr_ord'] = (
        df['age_curr']
        .astype(str).str.strip().str.lower()
        .map(_AGE_RANGE_MAP)
    )

    # ── 2. Binary flags (Y/N → 1/0) ────────────────────────────────────
    for flag_col in ['weapon_i', 'drug_i', 'cpd_arrest_i']:
        df[flag_col] = (
            df[flag_col].astype(str).str.strip().str.upper() == 'Y'
        ).astype(int)

    # ── 3. Sex → binary ────────────────────────────────────────────────
    df['sex_male'] = (
        df['sex_code_cd'].astype(str).str.strip().str.upper() == 'M'
    ).astype(int)

    # ── 4. Temporal: is latest arrest recent? ──────────────────────────
    _latest = df['latest_date']
    if _latest.dtype == 'object':
        _latest = pd.to_datetime(_latest, errors='coerce').dt.year
    df['latest_year_recent'] = (_latest >= 2014).astype(int)

    # ── assemble feature matrix (7 raw individual characteristics) ─────
    feature_cols = [
        'age_at_arrest_ord', 'age_curr_ord',
        'weapon_i', 'drug_i', 'cpd_arrest_i',
        'sex_male', 'latest_year_recent',
    ]
    X = df[feature_cols].copy()

    # ── target ─────────────────────────────────────────────────────────
    y = (df['ssl_score'] >= score_threshold).astype(int)

    # ── protected attribute: race ──────────────────────────────────────
    protected = (
        df['race_code_cd'].astype(str).str.strip().str.upper() == 'BLK'
    ).astype(int)

    # ── drop rows with any NaN left ────────────────────────────────────
    valid = X.notna().all(axis=1)
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    protected = protected[valid].reset_index(drop=True)

    if verbose:
        print(f"SSL after cleaning: {len(X):,} samples, {X.shape[1]} features")
        print(f"High-risk rate (score >= {score_threshold}): {y.mean():.2%}")
        print(f"Black (BLK) rate: {protected.mean():.2%}")

    # ── rename columns to short snake_case for downstream convenience ──
    rename_map = {
        'weapon_i':      'weapon_flag',
        'drug_i':        'drug_flag',
        'cpd_arrest_i':  'cpd_arrest_flag',
    }
    X = X.rename(columns=rename_map)
    feature_names = X.columns.tolist()

    # ── train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected,
        test_size=test_size, random_state=random_state, stratify=y
    )

    # ── standardise ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'protected_train': prot_train,
        'protected_test': prot_test,
        'feature_names': feature_names,
        'protected_attribute': 'race',
        'dataset_info': {
            'name': 'Chicago SSL',
            'n_samples': int(len(X_train) + len(X_test)),
            'n_features': len(feature_names),
            'base_rate': float(y.mean()),
            'base_rate_group_0': float(y[protected == 0].mean()) if (protected == 0).sum() > 0 else 0.0,
            'base_rate_group_1': float(y[protected == 1].mean()) if (protected == 1).sum() > 0 else 0.0,
            'group_0_name': 'Non-Black',
            'group_1_name': 'Black',
            'score_threshold': score_threshold,
        }
    }


# =============================================================================
# Universal Loader
# =============================================================================

def load_dataset(name: str, data_path: Optional[str] = None, **kwargs) -> Dict:
    """
    Universal dataset loader.
    
    Parameters
    ----------
    name : str
        Dataset name: 'compas', 'communities_crime', 'ssl'
    data_path : str, optional
        Path to local data file
    **kwargs
        Additional arguments passed to specific loader
    """
    loaders = {
        'compas': load_compas,
        'communities_crime': load_communities_crime,
        'communities': load_communities_crime,
        'ssl': load_ssl,
        'chicago_ssl': load_ssl
    }
    
    name_lower = name.lower().replace(' ', '_').replace('-', '_')
    
    if name_lower not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    return loaders[name_lower](data_path=data_path, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing data loaders...\n")
    
    # Test COMPAS
    print("=" * 50)
    print("COMPAS Dataset")
    print("=" * 50)
    try:
        data = load_compas()
        info = data['dataset_info']
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}")
        print(f"Base rate: {info['base_rate']:.3f}")
        print(f"Base rate {info['group_0_name']}: {info['base_rate_group_0']:.3f}")
        print(f"Base rate {info['group_1_name']}: {info['base_rate_group_1']:.3f}")
        print(f"Train size: {len(data['X_train'])}")
        print(f"Test size: {len(data['X_test'])}")
        print("✓ COMPAS loaded successfully")
    except Exception as e:
        print(f"✗ Error loading COMPAS: {e}")
    
    # Test Communities
    print("\n" + "=" * 50)
    print("Communities & Crime Dataset")
    print("=" * 50)
    try:
        data = load_communities_crime()
        info = data['dataset_info']
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}")
        print(f"Base rate: {info['base_rate']:.3f}")
        print(f"Base rate {info['group_0_name']}: {info['base_rate_group_0']:.3f}")
        print(f"Base rate {info['group_1_name']}: {info['base_rate_group_1']:.3f}")
        print("✓ Communities & Crime loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Communities: {e}")

    # Test SSL (uses default path next to this module)
    print("\n" + "=" * 50)
    print("Chicago SSL Dataset")
    print("=" * 50)
    try:
        data = load_ssl()
        info = data['dataset_info']
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}  →  {data['feature_names']}")
        print(f"Base rate (high risk): {info['base_rate']:.3f}")
        print(f"Base rate {info['group_0_name']}: {info['base_rate_group_0']:.3f}")
        print(f"Base rate {info['group_1_name']}: {info['base_rate_group_1']:.3f}")
        print(f"Train size: {len(data['X_train'])}")
        print(f"Test size: {len(data['X_test'])}")
        print("✓ Chicago SSL loaded successfully")
    except Exception as e:
        print(f"✗ Error loading SSL: {e}")
