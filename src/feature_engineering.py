"""
feature_engineering.py

Cleaning / parsing logic for the REAL house_prices.csv (Juhi Bhojani,
Kaggle - Indian real-estate listings). This replaces the earlier
placeholder version now that the actual schema has been inspected.

Every parser here was written against real observed values in the
dataset, e.g.:
  - "Amount(in rupees)": "42 Lac", "1.40 Cr", "Call for Price"
  - "Carpet Area": "500 sqft"
  - "Bathroom"/"Balcony": "1".."10", "> 10"
  - "Car Parking": "1 Open", "66 Covered", "1 Covered,"
  - "Floor": "10 out of 11", "Ground out of 7", "Lower Basement out of 2",
             sometimes just "2" (no "out of" part)
  - "overlooking": comma-separated subset of
             {"Garden/Park", "Pool", "Main Road", "Not Available"}

No values are invented -- rows/fields that can't be parsed become NaN
and are handled by imputation in preprocessing.py, except the target
(`Amount(in rupees)` -> price_rupees), where unparseable rows (e.g.
"Call for Price") are dropped since we cannot fabricate a price.

Outlier removal (required by the project guide): see
compute_price_per_sqft_bounds() and filter_price_per_sqft_outliers()
below. This is intentionally NOT run inside clean_real_dataset() --
outlier thresholds are percentile statistics, and computing them before
the train/test split (or on the test split) would leak information
about the held-out data into training. train_model.py calls
clean_real_dataset() first, THEN splits, THEN computes bounds from the
training portion only and applies them to both splits. See
train_model.py's docstring for the full rationale.

High-cardinality location handling (required by the project guide): see
compute_top_n_locations() and apply_location_grouping() below. The real
cleaned `location` column has 81 distinct values; per the guide, the
top 50 (by frequency, computed from the TRAINING split only, after
outlier removal) are kept as their own categories and every other
location -- including any location never seen at all -- is mapped to a
literal "other" category. This is a genuine training-time feature (the
model learns an actual "other location" effect from real training rows
grouped that way), not just an inference-time safety net -- OneHotEncoder's
handle_unknown="ignore" is a separate, additional safety net for values
that are still unseen after grouping (there shouldn't be any, since
"other" itself is always a known category, but it's a defensive backstop).
"""

import re
import numpy as np
import pandas as pd

LAC = 1e5
CR = 1e7


def parse_amount_to_rupees(series: pd.Series) -> pd.Series:
    """
    Parses the real 'Amount(in rupees)' column into a numeric rupee value.
    "42 Lac" -> 4,200,000 ; "1.40 Cr" -> 14,000,000
    "Call for Price" (no numeric value given) -> NaN (dropped later, not fabricated).
    """
    s = series.astype(str).str.strip()

    def _parse_one(v: str):
        m = re.match(r'^([\d.]+)\s*(Lac|Cr)$', v)
        if not m:
            return np.nan
        num, unit = float(m.group(1)), m.group(2)
        return num * (CR if unit == "Cr" else LAC)

    return s.apply(_parse_one)


def parse_area_sqft(series: pd.Series) -> pd.Series:
    """'500 sqft' -> 500.0 ; anything else unparseable -> NaN."""
    s = series.astype(str).str.strip()
    extracted = s.str.extract(r'^([\d.]+)\s*sqft$')[0]
    return pd.to_numeric(extracted, errors="coerce")


def parse_count_field(series: pd.Series) -> pd.Series:
    """Bathroom / Balcony: '1'..'10' -> numeric, '> 10' -> 11 (capped), NaN stays NaN."""
    s = series.astype(str).str.strip()
    s = s.replace({"> 10": "11"})
    return pd.to_numeric(s, errors="coerce")


def parse_car_parking(series: pd.Series) -> pd.DataFrame:
    """
    '1 Open' -> count=1, type='Open'
    '66 Covered' -> count=66, type='Covered'
    '1 Covered,' -> count=1, type='Covered'  (trailing comma stripped)
    Missing -> count=NaN, type=NaN
    """
    s = series.astype(str).str.strip().str.rstrip(",").str.strip()
    counts = pd.to_numeric(s.str.extract(r'^(\d+)')[0], errors="coerce")
    types = s.str.extract(r'(Covered|Open)$')[0]
    return pd.DataFrame({"car_parking_count": counts, "car_parking_type": types})


def parse_floor(series: pd.Series) -> pd.DataFrame:
    """
    '10 out of 11' -> floor_num=10, total_floors=11
    'Ground out of 7' -> floor_num=0, total_floors=7
    'Lower Basement out of 2' -> floor_num=-1, total_floors=2
    'Upper Basement out of 2' -> floor_num=-0.5 (below ground, above lower basement), total_floors=2
    '2' (no 'out of' part -- total floors unknown) -> floor_num=2, total_floors=NaN
    """
    def _floor_token_to_num(tok: str):
        tok = tok.strip()
        if tok == "Ground":
            return 0.0
        if tok == "Lower Basement":
            return -1.0
        if tok == "Upper Basement":
            return -0.5
        try:
            return float(tok)
        except ValueError:
            return np.nan

    floor_nums, total_floors = [], []
    for raw in series:
        if pd.isna(raw):
            floor_nums.append(np.nan)
            total_floors.append(np.nan)
            continue
        v = str(raw).strip()
        m = re.match(r'^(.*?)\s+out of\s+(\d+)$', v)
        if m:
            floor_nums.append(_floor_token_to_num(m.group(1)))
            total_floors.append(float(m.group(2)))
        else:
            floor_nums.append(_floor_token_to_num(v))
            total_floors.append(np.nan)

    return pd.DataFrame({"floor_num": floor_nums, "total_floors": total_floors})


def parse_overlooking_flags(series: pd.Series) -> pd.DataFrame:
    """
    Multi-label 'overlooking' -> three binary flags based on the real
    observed token set: {'Garden/Park', 'Pool', 'Main Road', 'Not Available'}.
    """
    s = series.fillna("").astype(str)
    return pd.DataFrame({
        "overlooks_garden_park": s.str.contains("Garden/Park", regex=False).astype(int),
        "overlooks_pool": s.str.contains("Pool", regex=False).astype(int),
        "overlooks_main_road": s.str.contains("Main Road", regex=False).astype(int),
    })


def clean_real_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the real house_prices.csv.
    Returns a new dataframe with:
      - a numeric target column 'price_rupees'
      - parsed numeric/categorical features
      - unusable columns (all-null, constant, free text, ultra-high-cardinality) dropped
    Rows where the target can't be recovered ('Call for Price') are dropped --
    never imputed or invented.
    """
    df = df.copy()

    # --- Target ---
    df["price_rupees"] = parse_amount_to_rupees(df["Amount(in rupees)"])
    before = len(df)
    df = df[df["price_rupees"].notna()].copy()
    dropped = before - len(df)
    print(f"Dropped {dropped} rows with unparseable/missing price "
          f"(e.g. 'Call for Price') out of {before}.")

    # --- Areas ---
    carpet = parse_area_sqft(df["Carpet Area"])
    super_ = parse_area_sqft(df["Super Area"])
    # Coalesce: prefer carpet area, fall back to super area, since either
    # represents real usable floor area and both are frequently missing.
    # Keep only the coalesced column (+ a flag for which source was used)
    # rather than all three, to avoid feeding the model near-duplicate,
    # highly collinear area columns.
    df["area_sqft"] = carpet.fillna(super_)
    df["area_is_carpet_area"] = carpet.notna().astype(int)
    df["area_missing"] = df["area_sqft"].isna().astype(int)

    # --- Counts ---
    df["bathroom_num"] = parse_count_field(df["Bathroom"])
    df["balcony_num"] = parse_count_field(df["Balcony"])

    # --- Car parking ---
    car = parse_car_parking(df["Car Parking"])
    df["car_parking_count"] = car["car_parking_count"]
    df["car_parking_type"] = car["car_parking_type"]

    # --- Floor ---
    floor = parse_floor(df["Floor"])
    df["floor_num"] = floor["floor_num"]
    df["total_floors"] = floor["total_floors"]

    # --- Overlooking (multi-label -> binary flags) ---
    overlook = parse_overlooking_flags(df["overlooking"])
    df = pd.concat([df, overlook], axis=1)

    # --- Society: too high-cardinality / too sparse to one-hot; keep only
    # a boolean signal of whether the listing belongs to a named society ---
    df["has_society"] = df["Society"].notna().astype(int)

    # --- Drop columns that are unusable or already re-encoded above ---
    drop_cols = [
        "Index", "Title", "Description",           # free text / identifiers
        "Dimensions", "Plot Area",                  # 100% missing in real data
        "Status",                                   # constant, no signal
        "Amount(in rupees)", "Price (in rupees)",   # replaced by price_rupees;
                                                     # 'Price (in rupees)' is a
                                                     # per-sqft-style figure derived
                                                     # from price itself -> would leak
        "Carpet Area", "Super Area",                # replaced by area_sqft + flags
        "Bathroom", "Balcony",                      # replaced by *_num numeric
        "Car Parking",                               # replaced by parsed fields
        "Floor",                                     # replaced by floor_num/total_floors
        "overlooking",                               # replaced by binary flags
        "Society",                                   # replaced by has_society
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def compute_price_per_sqft_bounds(
    price: pd.Series, area_sqft: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99
) -> tuple[float, float]:
    """
    Computes price-per-sqft outlier bounds (1st/99th percentile by default)
    from the given price/area pair ONLY -- callers must pass the TRAINING
    portion here, never the full dataset or the test set, so the
    thresholds are never informed by held-out data (avoiding leakage).

    Rows with missing or non-positive area are excluded from the
    percentile calculation itself (a price-per-sqft ratio is undefined
    for them), but are NOT removed by this function -- see
    filter_price_per_sqft_outliers() for how they're handled.
    """
    area = area_sqft.copy()
    valid = area.notna() & (area > 0) & price.notna()
    ratio = price[valid] / area[valid]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if ratio.empty:
        raise ValueError("No valid price/area pairs available to compute outlier bounds.")

    lower = float(ratio.quantile(lower_q))
    upper = float(ratio.quantile(upper_q))
    return lower, upper


def filter_price_per_sqft_outliers(
    df: pd.DataFrame,
    price_col: str,
    area_col: str,
    lower_bound: float,
    upper_bound: float,
) -> pd.DataFrame:
    """
    Removes listings whose real price-per-sqft ratio falls below
    `lower_bound` or above `upper_bound` (per the project guide's
    required outlier-removal step: absurd price-per-sqft below the 1st
    percentile or above the 99th percentile).

    `lower_bound`/`upper_bound` must come from
    compute_price_per_sqft_bounds() called on the TRAINING split only
    (see train_model.py) -- this function just applies them.

    Rows where area is missing or non-positive are KEPT (not removed):
    a price-per-sqft ratio can't be computed for them, and removing them
    based on a ratio we can't calculate would mean discarding real,
    otherwise-usable listings on a criterion we have no evidence for --
    the opposite of "safe handling of missing/invalid area". Missing
    area is instead already flagged via `area_missing` and handled by
    imputation downstream.
    """
    area = df[area_col]
    price = df[price_col]

    has_valid_area = area.notna() & (area > 0)
    ratio = pd.Series(np.nan, index=df.index)
    ratio.loc[has_valid_area] = price[has_valid_area] / area[has_valid_area]

    is_outlier = has_valid_area & ((ratio < lower_bound) | (ratio > upper_bound))

    kept = df.loc[~is_outlier].copy()
    removed = int(is_outlier.sum())
    print(
        f"Removed {removed} price-per-sqft outlier rows "
        f"(ratio outside [{lower_bound:,.0f}, {upper_bound:,.0f}] rupees/sqft) "
        f"out of {len(df)}; {int((~has_valid_area).sum())} rows with missing/invalid "
        f"area were kept (ratio undefined for them, not removed)."
    )
    return kept


OTHER_LOCATION_LABEL = "other"


def compute_top_n_locations(location_series: pd.Series, n: int = 50) -> list[str]:
    """
    Returns the N most frequent real location values, computed from
    whatever series is passed in -- callers MUST pass the TRAINING
    portion only (post outlier-removal), never the full dataset or the
    test split, so this never uses test-set information to decide which
    locations count as "top".
    """
    counts = location_series.dropna().value_counts()
    return counts.head(n).index.tolist()


def apply_location_grouping(
    location_series: pd.Series, top_locations: list[str] | set[str],
    other_label: str = OTHER_LOCATION_LABEL,
) -> pd.Series:
    """
    Maps every location value NOT in `top_locations` to `other_label`
    (including locations never seen at all, e.g. at inference time) --
    the same treatment for "not in the top 50" and "completely unseen",
    since both cases mean the model has no reliable per-location signal
    for that value. `top_locations` must come from
    compute_top_n_locations() called on the training split only; this
    function just applies the mapping and is safe to call on train,
    test, or a single inference row using the identical `top_locations`
    set.
    """
    allowed = set(top_locations)
    return location_series.where(location_series.isin(allowed), other_label)
