"""
preprocessing.py

Schema-agnostic preprocessing for the REAL house_prices.csv.

Rather than hardcoding column names (which we don't know until the real
file is inspected), this builds a sklearn ColumnTransformer dynamically:
- numeric columns -> median imputation + StandardScaler
- categorical columns -> most-frequent imputation + OneHotEncoder

This should work directly on the real Kaggle data. The only manual input
required is which column is the target (label) and which columns (if any)
should be excluded (e.g. an ID column) -- set these in CONFIG after
inspecting the real schema with data_loader.inspect_schema().
"""

from dataclasses import dataclass, field
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class PreprocessConfig:
    target_column: str                     # e.g. "price" -- CONFIRM after inspecting real CSV
    id_columns: list = field(default_factory=list)   # columns to drop entirely (e.g. "Id")
    location_column: str | None = None     # used later for locations.json, not dropped here


def split_features_target(df: pd.DataFrame, config: PreprocessConfig):
    if config.target_column not in df.columns:
        raise KeyError(
            f"Target column '{config.target_column}' not found in real dataset. "
            f"Available columns: {list(df.columns)}. Update CONFIG.target_column."
        )
    drop_cols = [c for c in config.id_columns if c in df.columns]
    X = df.drop(columns=[config.target_column] + drop_cols)
    y = df[config.target_column]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Auto-detects numeric vs categorical columns from the REAL data and
    builds the corresponding transformers. No column names are hardcoded.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    print(f"Detected {len(numeric_cols)} numeric columns: {numeric_cols}")
    print(f"Detected {len(categorical_cols)} categorical columns: {categorical_cols}")

    return preprocessor
