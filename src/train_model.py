"""
train_model.py

Trains TWO regression models on the REAL house_prices.csv, evaluates both
honestly on the same held-out test set, builds a comparison table, and
saves the winning model as models/house_price.pkl (compressed to stay
under the project guide's 50MB GitHub limit) plus outputs/locations.json.

Models compared:
  1. RandomForestRegressor  -- non-linear, handles mixed feature interactions
  2. LinearRegression        -- simple linear baseline

Both share the exact same train/test split, the exact same preprocessing
(ColumnTransformer fit only on training data), the exact same outlier
removal, and the exact same log1p/expm1 target transform, so the
comparison is apples-to-apples.

OUTLIER REMOVAL (project guide requirement) -- methodology and why it's
ordered this way:
  1. clean_real_dataset() parses the raw CSV (per-row logic only, no
     leakage risk).
  2. train_test_split() happens BEFORE any outlier statistic is computed.
  3. compute_price_per_sqft_bounds() computes the 1st/99th percentile of
     real price-per-sqft using ONLY the training split's price and area.
  4. filter_price_per_sqft_outliers() applies those train-derived bounds
     to BOTH the training split (removing outliers from what the model
     learns from) AND the test split (removing outliers from what the
     model is evaluated on) -- using the same bounds on both, never
     refitting them on test data. This mirrors how any other fitted
     statistic (e.g. an imputer's median) is handled: fit on train,
     applied to test, never fit on test. This keeps the held-out test
     evaluation honest -- no information about which test rows are
     outliers leaks into the threshold calculation.

TOP-N LOCATION GROUPING (project guide requirement) -- same pattern:
  1. compute_top_n_locations() computes the 50 most frequent locations
     using ONLY the (outlier-filtered) training split.
  2. apply_location_grouping() maps every location not in that top-50
     set -- in EITHER split, and at inference time -- to a literal
     "other" category, using the same train-derived set throughout.
  This makes "other" a real, learned category (the model sees real
  training rows labeled "other" and learns an actual price effect for
  it), not just a handle_unknown="ignore" safety net at inference. The
  backend applies the identical mapping using the same locations.json
  (see backend/app/services/preprocessing.py).

This script REFUSES to run if the real dataset file is not present --
see data_loader.load_real_dataset(). It will not train on synthetic data
under any circumstances.

Usage:
    python src/train_model.py
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
)

from data_loader import load_real_dataset, inspect_schema
from preprocessing import PreprocessConfig, split_features_target, build_preprocessor
from feature_engineering import (
    clean_real_dataset,
    compute_price_per_sqft_bounds,
    filter_price_per_sqft_outliers,
    compute_top_n_locations,
    apply_location_grouping,
    OTHER_LOCATION_LABEL,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price.pkl"
LOCATIONS_PATH = ROOT / "outputs" / "locations.json"

CONFIG = PreprocessConfig(
    target_column="price_rupees",
    id_columns=[],
    location_column="location",
)

OUTLIER_LOWER_QUANTILE = 0.01
OUTLIER_UPPER_QUANTILE = 0.99
TOP_N_LOCATIONS = 50

# RandomForest sized specifically to stay well under the 50MB GitHub
# limit once joblib-compressed, while keeping accuracy close to an
# unconstrained forest (verified previously: unconstrained 150-tree/
# depth-20 forest -> 231MB; this constrained version -> ~28MB).
RF_PARAMS = dict(
    n_estimators=150,
    max_depth=16,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1,
)
JOBLIB_COMPRESS_LEVEL = 3


def build_model_candidates():
    """
    Both candidates are wrapped in TransformedTargetRegressor with the
    same log1p/expm1 transform: real prices are heavily right-skewed,
    so fitting on log(price) and inverting predictions back to rupees is
    applied identically to both models for a fair comparison -- this is
    a preprocessing choice, not a metric-improving trick, and it's
    applied uniformly.
    """
    rf = TransformedTargetRegressor(
        regressor=RandomForestRegressor(**RF_PARAMS),
        func=np.log1p, inverse_func=np.expm1,
    )
    lr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p, inverse_func=np.expm1,
    )
    return {
        "RandomForest": rf,
        "LinearRegression": lr,
    }


def evaluate(y_true, preds):
    """
    Computes MAE / RMSE / R^2 on the raw rupee scale (as the project guide
    requires), plus two additional, more robust views that are reported
    alongside -- never in place of -- the raw metrics:

      - R^2 (log scale): R^2 computed on log1p(price), far less sensitive
        to any remaining extreme values.
      - R^2 (99th pct trimmed): R^2 on raw rupees after excluding the
        top 1% most expensive real test listings.

    These stay clearly labeled and separate from the raw-scale metrics
    throughout -- they are additional context, not a replacement.
    """
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds) ** 0.5
    r2 = r2_score(y_true, preds)
    medae = median_absolute_error(y_true, preds)
    r2_log = r2_score(np.log1p(y_true), np.log1p(np.clip(preds, 0, None)))
    p99 = np.quantile(y_true, 0.99)
    mask = np.asarray(y_true) <= p99
    r2_trimmed = r2_score(np.asarray(y_true)[mask], np.asarray(preds)[mask])
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "MedAE": medae, "R2_log": r2_log, "R2_trimmed_99pct": r2_trimmed,
    }


def train_and_evaluate():
    print("Loading real dataset...")
    df = load_real_dataset()
    schema = inspect_schema(df)
    print(f"Loaded real raw data with shape {schema['shape']}")

    df = clean_real_dataset(df)
    rows_before_outlier_removal = df.shape[0]
    print(f"After cleaning (before outlier removal): "
          f"{rows_before_outlier_removal} usable rows, {df.shape[1]} columns")

    if CONFIG.target_column not in df.columns:
        raise KeyError(
            f"CONFIG.target_column='{CONFIG.target_column}' missing after cleaning. "
            f"Available columns: {list(df.columns)}."
        )

    # --- Split BEFORE computing any outlier statistic ---
    X, y = split_features_target(df, CONFIG)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Outlier removal (price-per-sqft based, project guide requirement) ---
    # Bounds computed from the TRAINING split's real price/area only --
    # never from X_test/y_test -- so no test-set information leaks into
    # the threshold.
    lower_bound, upper_bound = compute_price_per_sqft_bounds(
        y_train, X_train["area_sqft"],
        lower_q=OUTLIER_LOWER_QUANTILE, upper_q=OUTLIER_UPPER_QUANTILE,
    )
    print(f"\nPrice-per-sqft outlier bounds (from TRAIN only, "
          f"{OUTLIER_LOWER_QUANTILE:.0%}/{OUTLIER_UPPER_QUANTILE:.0%}): "
          f"[{lower_bound:,.0f}, {upper_bound:,.0f}] rupees/sqft")

    train_df = X_train.copy()
    train_df[CONFIG.target_column] = y_train
    train_df = filter_price_per_sqft_outliers(
        train_df, CONFIG.target_column, "area_sqft", lower_bound, upper_bound
    )
    X_train = train_df.drop(columns=[CONFIG.target_column])
    y_train = train_df[CONFIG.target_column]

    test_df = X_test.copy()
    test_df[CONFIG.target_column] = y_test
    test_df = filter_price_per_sqft_outliers(
        test_df, CONFIG.target_column, "area_sqft", lower_bound, upper_bound
    )
    X_test = test_df.drop(columns=[CONFIG.target_column])
    y_test = test_df[CONFIG.target_column]

    rows_after_outlier_removal = len(X_train) + len(X_test)
    print(f"\nRows before outlier removal (post-cleaning, both splits): "
          f"{rows_before_outlier_removal}")
    print(f"Rows after outlier removal (both splits combined): "
          f"{rows_after_outlier_removal}")
    print(f"Train: {len(y_train)} rows | Test: {len(y_test)} rows")

    # --- Top-N location grouping (project guide requirement) ---
    # Top-50 locations computed from the TRAINING split only (post
    # outlier-removal, i.e. the actual final training data), never from
    # X_test. Every other location -- including any location never seen
    # at all -- is mapped to a literal "other" category in BOTH splits,
    # using this same train-derived top-50 set.
    top_locations = compute_top_n_locations(X_train["location"], n=TOP_N_LOCATIONS)
    print(f"\nTop-{TOP_N_LOCATIONS} locations computed from TRAIN only "
          f"(of {X_train['location'].nunique()} distinct real locations in train).")

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["location"] = apply_location_grouping(X_train["location"], top_locations)
    X_test["location"] = apply_location_grouping(X_test["location"], top_locations)

    train_other_count = int((X_train["location"] == OTHER_LOCATION_LABEL).sum())
    test_other_count = int((X_test["location"] == OTHER_LOCATION_LABEL).sum())
    print(f"Grouped into '{OTHER_LOCATION_LABEL}': {train_other_count} train rows, "
          f"{test_other_count} test rows.")
    print(f"Final location categories used by the model: "
          f"{X_train['location'].nunique()} (top {TOP_N_LOCATIONS} + '{OTHER_LOCATION_LABEL}').")

    # --- Train + evaluate both models on the outlier-filtered, split ---
    results = {}
    fitted_pipelines = {}

    for name, model in build_model_candidates().items():
        print(f"\nTraining {name} on REAL, outlier-filtered, location-grouped data...")
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(f"  MAE={metrics['MAE']:,.0f}  RMSE={metrics['RMSE']:,.0f}  "
              f"R2(raw)={metrics['R2']:.4f}  R2(log)={metrics['R2_log']:.4f}")

    # --- Comparison table (raw-scale metrics and log/trimmed metrics
    # kept in clearly separate, clearly labeled columns) ---
    comparison = pd.DataFrame(results).T[
        ["MAE", "RMSE", "R2", "MedAE", "R2_log", "R2_trimmed_99pct"]
    ]
    print("\n=== Model comparison (real held-out, outlier-filtered test set) ===")
    print(comparison.to_string(float_format=lambda v: f"{v:,.4f}"))

    # --- Winner selection ---
    # Raw-scale R^2 alone is not a reliable ranking criterion here: it is
    # dominated by a handful of extreme-outlier listings and can even go
    # negative for an otherwise reasonable model. We select the winner
    # using MAE (rupee-scale, robust to outliers, and the most directly
    # interpretable "typical error" number) as the primary criterion,
    # with R2_log as a tie-breaker/consistency check. This is decided
    # BEFORE looking at which model 'wins' on raw R^2, to avoid
    # cherry-picking a metric that flatters one model.
    winner_name = comparison["MAE"].astype(float).idxmin()
    winner_pipeline = fitted_pipelines[winner_name]
    print(f"\nWinner (lowest MAE): {winner_name}")
    print(
        "Justification: MAE is reported in real rupees and is not distorted "
        "by any remaining extreme-value listings the way RMSE/raw-R^2 are, "
        "making it the more honest primary criterion for 'typical' "
        "prediction error. R2_log (also robust to those outliers) is "
        "checked as a consistency tie-breaker."
    )

    # --- Save winning model (compressed to respect the 50MB guide limit) ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipeline, MODEL_PATH, compress=JOBLIB_COMPRESS_LEVEL)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved winning model ({winner_name}) to {MODEL_PATH}  "
          f"[{size_mb:.2f} MB, compress={JOBLIB_COMPRESS_LEVEL}]")
    if size_mb >= 50:
        print("WARNING: model still exceeds the 50MB guide limit.")

    # --- Verify the compressed file loads back and predicts identically ---
    reloaded = joblib.load(MODEL_PATH)
    reloaded_preds = reloaded.predict(X_test)
    identical = np.allclose(reloaded_preds, winner_pipeline.predict(X_test))
    print(f"Reloaded model prediction check (matches in-memory model): {identical}")

    # --- Save locations.json: the Top-N real locations the model was
    # actually trained on (NOT the full 81 -- "other" is a model
    # category but isn't a real place, so it's not offered as a
    # frontend dropdown choice) ---
    if CONFIG.location_column:
        LOCATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCATIONS_PATH, "w") as f:
            json.dump(sorted(top_locations), f, indent=2)
        print(f"Saved {len(top_locations)} real Top-{TOP_N_LOCATIONS} locations to {LOCATIONS_PATH}")

    return {
        "rows_before_outlier_removal": rows_before_outlier_removal,
        "rows_after_outlier_removal": rows_after_outlier_removal,
        "outlier_bounds": (lower_bound, upper_bound),
        "top_locations": top_locations,
        "comparison": comparison,
        "winner": winner_name,
        "model_path": str(MODEL_PATH),
        "model_size_mb": size_mb,
        "locations_path": str(LOCATIONS_PATH) if CONFIG.location_column else None,
    }


if __name__ == "__main__":
    train_and_evaluate()

