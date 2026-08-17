"""
Revenue Model — Phase 2
Predicts annual STR revenue for a never-listed property using Ridge regression
with market normalization. Trained on AirROI data across 15 markets.

Key concept: model predicts revenue_ratio (how much above/below market median)
not raw revenue. Market median anchors the dollar amount; model adjusts for
property-specific features.

Usage:
  python revenue_model.py               ← train + evaluate
  python revenue_model.py --predict     ← predict for a specific property
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

# ── City centers + key landmarks ──────────────────────────────────────────────

CITY_CENTERS = {
    "indianapolis":   (39.7684, -86.1581),
    "los_angeles":    (34.0522, -118.2437),
    "las_vegas":      (36.1699, -115.1398),
    "anaheim":        (33.8366, -117.9143),
    "austin":         (30.2672, -97.7431),
    "four_corners":   (36.9990, -109.0452),
    "houston":        (29.7604, -95.3698),
    "kissimmee":      (28.2919, -81.4076),
    "miami":          (25.7617, -80.1918),
    "portland":       (45.5051, -122.6750),
    "san_diego":      (32.7157, -117.1611),
    "san_francisco":  (37.7749, -122.4194),
    "seattle":        (47.6062, -122.3321),
    "south_lake_tahoe": (38.9399, -119.9772),
    "washington":     (38.9072, -77.0369),
}

# Key attraction for each market — second distance feature
KEY_ATTRACTIONS = {
    "indianapolis":   (39.7640, -86.1555),   # Lucas Oil Stadium
    "los_angeles":    (34.0195, -118.4912),   # Santa Monica Pier
    "las_vegas":      (36.1147, -115.1728),   # Las Vegas Strip center
    "anaheim":        (33.8121, -117.9190),   # Disneyland
    "austin":         (30.2849, -97.7341),    # 6th Street
    "four_corners":   (36.9990, -109.0452),   # Monument
    "houston":        (29.6905, -95.4097),    # NRG Stadium
    "kissimmee":      (28.3747, -81.5494),    # Walt Disney World
    "miami":          (25.7900, -80.1300),    # South Beach
    "portland":       (45.5231, -122.6765),   # Pioneer Square
    "san_diego":      (32.7076, -117.1576),   # Gaslamp Quarter
    "san_francisco":  (37.8024, -122.4058),   # Fisherman's Wharf
    "seattle":        (47.6205, -122.3493),   # Space Needle
    "south_lake_tahoe": (38.9399, -119.9772), # Lake Tahoe
    "washington":     (38.8895, -77.0353),    # National Mall
}


def haversine(lat1, lon1, lat2, lon2):
    """Distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Target markets — model predictions will be anchored to these medians
TARGET_MARKETS = ["indianapolis", "los_angeles", "las_vegas"]

# Numeric features
NUMERIC_FEATURES = [
    "bedrooms",
    "baths",
    "guests",
    "amenities_count",
    "cleaning_fee",
    "photos_count",             # quality signal — more photos = better listing
    "min_nights",
    "superhost",
    "instant_book",
    "professional_management",
    "guest_favorite",
    "rating_overall",
    "rating_cleanliness",
    "rating_location",
    "rating_value",
    "num_reviews",
    "ttm_avg_length_of_stay",
    "dist_to_center",
    "dist_to_attraction",
    "permit_required",          # regulatory features
    "owner_occupancy",
    "has_night_cap",
    "night_cap",
]

# Categorical features — one-hot encoded
CATEGORICAL_FEATURES = ["market"]

# Model target — monthly revenue ratio
TARGET = "monthly_revenue"

# ── Load and prepare data ─────────────────────────────────────────────────────

def load_training_data():
    path = os.path.join(PROCESSED_DIR, "all_listings.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} listings across {df['market'].nunique()} markets")
    return df


def add_market_median(df):
    """Add market median MONTHLY revenue per bedroom count as a column."""
    if "monthly_revenue" not in df.columns:
        df["monthly_revenue"] = df["ttm_revenue"] / 12

    df["market_median_monthly"] = df.groupby(
        ["market", "bedrooms"]
    )["monthly_revenue"].transform("median")

    df["revenue_ratio"] = df["monthly_revenue"] / df["market_median_monthly"]

    df = df[df["market_median_monthly"].notna() & df["revenue_ratio"].notna()]
    df = df[df["revenue_ratio"].between(0.1, 5.0)]
    return df


def add_location_features(df):
    """Add distance to city center and key attraction per listing."""
    dist_center     = []
    dist_attraction = []

    for _, row in df.iterrows():
        market = row["market"]
        lat    = row.get("latitude",  None)
        lon    = row.get("longitude", None)

        if pd.notna(lat) and pd.notna(lon) and market in CITY_CENTERS:
            cc = CITY_CENTERS[market]
            ka = KEY_ATTRACTIONS[market]
            dist_center.append(haversine(lat, lon, cc[0], cc[1]))
            dist_attraction.append(haversine(lat, lon, ka[0], ka[1]))
        else:
            dist_center.append(np.nan)
            dist_attraction.append(np.nan)

    df["dist_to_center"]     = dist_center
    df["dist_to_attraction"] = dist_attraction

    # Fill missing with market median distance
    df["dist_to_center"]     = df.groupby("market")["dist_to_center"].transform(
        lambda x: x.fillna(x.median())
    )
    df["dist_to_attraction"] = df.groupby("market")["dist_to_attraction"].transform(
        lambda x: x.fillna(x.median())
    )
    return df


def prepare_features(df):
    """Clean and prepare feature matrix."""
    df = add_location_features(df)

    for col in ["rating_overall", "rating_cleanliness", "rating_location", "rating_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df["min_nights"]             = pd.to_numeric(df["min_nights"],             errors="coerce").fillna(1)
    df["cleaning_fee"]           = pd.to_numeric(df["cleaning_fee"],           errors="coerce").fillna(0)
    df["ttm_avg_length_of_stay"] = pd.to_numeric(df["ttm_avg_length_of_stay"], errors="coerce").fillna(3.5)
    df["num_reviews"]            = pd.to_numeric(df["num_reviews"],            errors="coerce").fillna(0)
    df["photos_count"]           = pd.to_numeric(df.get("photos_count", 10),  errors="coerce").fillna(10)

    # Regulatory features — fill from STR_REGULATIONS if missing
    for col in ["permit_required", "owner_occupancy", "has_night_cap"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "night_cap" not in df.columns:
        df["night_cap"] = 365
    df["night_cap"] = pd.to_numeric(df["night_cap"], errors="coerce").fillna(365)

    for col in ["superhost", "instant_book", "professional_management", "guest_favorite"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    missing_num   = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing_num:
        print(f"  ⚠ Missing features: {missing_num}")

    return df, available_num


# ── Recency factor ────────────────────────────────────────────────────────────

def compute_recency_factor(df):
    """
    Compare last 90 days annualized vs TTM monthly to detect market trend.
    """
    if "l90d_revenue" not in df.columns:
        return 1.0
    l90d_monthly = df["l90d_revenue"] / 3      # 90 days = ~3 months
    ttm_monthly  = df["ttm_revenue"]  / 12
    ratio = (l90d_monthly / ttm_monthly).median()
    return round(float(ratio), 3)


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(df):
    df = add_market_median(df)
    print(f"  After normalization: {len(df)} usable listings")
    print(f"  Monthly rev ratio range:  {df['revenue_ratio'].min():.2f} — {df['revenue_ratio'].max():.2f}")
    print(f"  Monthly rev ratio median: {df['revenue_ratio'].median():.2f}")

    df, available_num = prepare_features(df)
    y = df["revenue_ratio"]
    X = df[available_num + CATEGORICAL_FEATURES]

    # ColumnTransformer: scale numerics, one-hot encode market
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), available_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("ridge",        Ridge(alpha=5.0)),
    ])
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_r2  =  cross_val_score(pipeline, X, y, cv=kf, scoring="r2")

    print(f"\n  Cross-validation results (5-fold):")
    print(f"  MAE on ratio:  {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")
    print(f"  R²:            {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"\n  Interpreting MAE: predictions typically within ±{cv_mae.mean()*100:.0f}% of market median")

    # Fit on full dataset
    pipeline.fit(X, y)

    # Feature importance from Ridge coefficients
    ridge  = pipeline.named_steps["ridge"]
    ohe    = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = available_num + cat_names
    coef_pairs = sorted(zip(all_names, ridge.coef_), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Feature importance (top 12):")
    print(f"  {'Feature':<32} {'Coefficient':>12}")
    print(f"  {'─'*46}")
    for feat, coef in coef_pairs[:12]:
        direction = "↑" if coef > 0 else "↓"
        print(f"  {feat:<32} {coef:>+10.3f}  {direction}")

    # Save
    model_data = {
        "pipeline":          pipeline,
        "numeric_features":  available_num,
        "cat_features":      CATEGORICAL_FEATURES,
        "all_feature_names": all_names,
        "markets":           df["market"].unique().tolist(),
    }
    with open(os.path.join(MODEL_DIR, "revenue_model.pkl"), "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  ✓ Model saved → data/models/revenue_model.pkl")

    return pipeline, available_num, df


# ── Market summary loader ─────────────────────────────────────────────────────

def load_market_summary():
    path = os.path.join(PROCESSED_DIR, "market_summary.csv")
    df   = pd.read_csv(path)
    # Build lookup: market + bedrooms → median stats
    lookup = {}
    for _, row in df.iterrows():
        key = (row["market"], int(row["bedrooms"]))
        lookup[key] = {
            "median_annual_revenue": row["median_annual_revenue"],
            "median_adr":            row["median_adr"],
            "median_occupancy":      row["median_occupancy"],
            "p25_revenue":           row["p25_revenue"],
            "p75_revenue":           row["p75_revenue"],
            "sample_size":           row["sample_size"],
        }
    return lookup


def load_seasonality():
    path = os.path.join(PROCESSED_DIR, "seasonality.csv")
    df   = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        market = row["market"]
        month  = int(row["month"])
        if market not in lookup:
            lookup[market] = {}
        lookup[market][month] = {
            "revenue_index":   row["revenue_index"],
            "occupancy_index": row["occupancy_index"],
            "rate_index":      row["rate_index"],
        }
    return lookup


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_revenue(
    market_key,
    bedrooms,
    baths,
    guests,
    amenities_count,
    cleaning_fee,
    latitude,
    longitude,
    photos_count        = 15,
    min_nights          = 2,
    superhost           = 0,
    instant_book        = 1,
    professional_mgmt   = 0,
    guest_favorite      = 0,
    rating_overall      = 4.8,
    rating_cleanliness  = 4.8,
    rating_location     = 4.8,
    rating_value        = 4.7,
    num_reviews         = 0,
    avg_stay_length     = 3.5,
    target_month        = None,
    recency_factor      = 1.0,
):
    model_path = os.path.join(MODEL_DIR, "revenue_model.pkl")
    if not os.path.exists(model_path):
        print("  ✗ Model not found. Train first: python revenue_model.py")
        return None

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    pipeline         = model_data["pipeline"]
    numeric_features = model_data["numeric_features"]
    cat_features     = model_data["cat_features"]

    market_summary = load_market_summary()
    seasonality    = load_seasonality()

    key = (market_key, int(bedrooms))
    if key not in market_summary:
        available_brs = [k[1] for k in market_summary if k[0] == market_key]
        nearest_br    = min(available_brs, key=lambda x: abs(x - bedrooms))
        key           = (market_key, nearest_br)
        print(f"  ⚠ No data for {bedrooms}BR in {market_key}, using {nearest_br}BR median")

    market_data          = market_summary[key]
    market_median_monthly= market_data.get("median_monthly_revenue",
                           market_data["median_annual_revenue"] / 12)
    p25                  = market_data.get("p25_monthly", market_data["p25_revenue"] / 12)
    p75                  = market_data.get("p75_monthly", market_data["p75_revenue"] / 12)
    sample_size          = market_data["sample_size"]

    # Regulatory features for this market
    reg             = STR_REGULATIONS.get(market_key, {})
    permit_required = reg.get("permit_required", 0)
    owner_occupancy = reg.get("owner_occupancy", 0)
    has_night_cap   = 1 if reg.get("cap_nights") else 0
    night_cap       = reg.get("cap_nights") or 365

    cc              = CITY_CENTERS.get(market_key, (latitude, longitude))
    ka              = KEY_ATTRACTIONS.get(market_key, (latitude, longitude))
    dist_to_center  = haversine(latitude, longitude, cc[0], cc[1])
    dist_to_attract = haversine(latitude, longitude, ka[0], ka[1])

    feature_map = {
        "bedrooms":                bedrooms,
        "baths":                   baths,
        "guests":                  guests,
        "amenities_count":         amenities_count,
        "cleaning_fee":            cleaning_fee,
        "photos_count":            photos_count,
        "min_nights":              min_nights,
        "superhost":               superhost,
        "instant_book":            instant_book,
        "professional_management": professional_mgmt,
        "guest_favorite":          guest_favorite,
        "rating_overall":          rating_overall,
        "rating_cleanliness":      rating_cleanliness,
        "rating_location":         rating_location,
        "rating_value":            rating_value,
        "num_reviews":             num_reviews,
        "ttm_avg_length_of_stay":  avg_stay_length,
        "dist_to_center":          dist_to_center,
        "dist_to_attraction":      dist_to_attract,
        "permit_required":         permit_required,
        "owner_occupancy":         owner_occupancy,
        "has_night_cap":           has_night_cap,
        "night_cap":               night_cap,
        "market":                  market_key,
    }

    num_vals = [feature_map[c] for c in numeric_features]
    cat_vals = [feature_map[c] for c in cat_features]
    X = pd.DataFrame([num_vals + cat_vals], columns=numeric_features + cat_features)

    ratio = float(pipeline.predict(X)[0])
    ratio = max(0.1, min(ratio, 5.0))
    ratio *= recency_factor

    predicted_monthly = market_median_monthly * ratio
    ci_low            = p25 * ratio
    ci_high           = p75 * ratio
    predicted_annual  = predicted_monthly * 12

    seas_index = 1.0
    if target_month and market_key in seasonality:
        seas_index        = seasonality[market_key][target_month]["revenue_index"]
        predicted_month_s = predicted_monthly * seas_index
    else:
        predicted_month_s = predicted_monthly

    market_label = market_key.replace("_", " ").title()
    divider      = "─" * 52

    print(f"\n{'═'*52}")
    print(f"  REVENUE PREDICTION  |  {market_label}")
    print(f"{'═'*52}")
    print(f"\n  PROPERTY")
    print(f"  {divider}")
    print(f"  Bedrooms            {bedrooms:>20}")
    print(f"  Bathrooms           {baths:>20.1f}")
    print(f"  Guests              {guests:>20}")
    print(f"  Amenities           {amenities_count:>20}")
    print(f"  Photos              {photos_count:>20}")
    print(f"  Cleaning fee        ${cleaning_fee:>19,.0f}")
    print(f"  Dist to center      {dist_to_center:>18.1f}mi")
    print(f"  Dist to attraction  {dist_to_attract:>18.1f}mi")
    print(f"  Superhost           {'Yes' if superhost else 'No':>20}")

    print(f"\n  REGULATORY  ({market_key})")
    print(f"  {divider}")
    print(f"  Permit required     {'Yes' if permit_required else 'No':>20}")
    print(f"  Owner occupancy req {'Yes' if owner_occupancy else 'No':>20}")
    print(f"  Night cap           {str(night_cap) + ' nights' if has_night_cap else 'None':>20}")

    print(f"\n  MARKET ANCHOR  (based on {int(sample_size)} comps)")
    print(f"  {divider}")
    print(f"  {int(bedrooms)}BR monthly median    ${market_median_monthly:>18,.0f}/mo")
    print(f"  Revenue ratio        {ratio:>19.2f}x")
    print(f"  Recency factor       {recency_factor:>19.2f}x")

    print(f"\n  PREDICTION")
    print(f"  {divider}")
    print(f"  Monthly revenue     ${predicted_month_s:>19,.0f}/mo")
    print(f"  Annual revenue      ${predicted_annual:>19,.0f}/yr")
    print(f"  Monthly CI range    ${ci_low:>8,.0f} — ${ci_high:,.0f}/mo")
    if target_month:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        print(f"  Seasonality ({months[target_month-1]})   {seas_index:>19.2f}x avg")
    print(f"  ⚠ Sample size: {int(sample_size)} comps — {'reliable' if sample_size >= 10 else 'low confidence'}")

    implied_occ = market_data["median_occupancy"] * ratio
    print(f"\n  IMPLIED METRICS")
    print(f"  {divider}")
    print(f"  Implied occupancy   {min(implied_occ, 0.95)*100:>19.1f}%")
    print(f"  Implied ADR         ${market_data['median_adr']:>19,.0f}/night")
    print(f"{'═'*52}\n")

    return {
        "predicted_monthly_revenue": round(predicted_month_s),
        "predicted_annual_revenue":  round(predicted_annual),
        "ci_low_monthly":  round(ci_low),
        "ci_high_monthly": round(ci_high),
        "revenue_ratio":   round(ratio, 3),
        "market_median_monthly": round(market_median_monthly),
        "implied_occupancy": round(min(implied_occ, 0.95), 3),
        "implied_adr": round(market_data["median_adr"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="Run prediction for a sample property")
    args = parser.parse_args()

    if not args.predict:
        print(f"\n{'═'*52}")
        print(f"  REVENUE MODEL — TRAINING")
        print(f"{'═'*52}\n")

        df = load_training_data()

        # Compute recency factors per target market
        print("\n  Recency factors (l90d vs TTM trend):")
        for market in TARGET_MARKETS:
            mdf    = df[df["market"] == market]
            factor = compute_recency_factor(mdf)
            trend  = "↑ heating up" if factor > 1.05 else "↓ cooling" if factor < 0.95 else "→ stable"
            print(f"  {market:<20} {factor:.2f}x  {trend}")

        print(f"\n  Training model...")
        pipeline, feature_cols, df_processed = train_model(df)

        # Save recency factors
        recency = {}
        for market in TARGET_MARKETS:
            mdf = df[df["market"] == market]
            recency[market] = compute_recency_factor(mdf)
        with open(os.path.join(MODEL_DIR, "recency_factors.json"), "w") as f:
            json.dump(recency, f, indent=2)
        print(f"  ✓ Recency factors saved → data/models/recency_factors.json")

    else:
        # Load recency factors
        recency_path = os.path.join(MODEL_DIR, "recency_factors.json")
        recency = json.load(open(recency_path)) if os.path.exists(recency_path) else {}

        # ── Example predictions — edit these to match a real property ─────
        print("\n  Running sample predictions...\n")

        # Indianapolis — 2BR house near downtown
        predict_revenue(
            market_key      = "indianapolis",
            bedrooms        = 2,
            baths           = 1.0,
            guests          = 4,
            amenities_count = 18,
            cleaning_fee    = 95,
            photos_count    = 20,
            latitude        = 39.7850,
            longitude       = -86.1527,
            min_nights      = 2,
            superhost       = 0,
            instant_book    = 1,
            rating_overall  = 4.7,
            recency_factor  = recency.get("indianapolis", 1.0),
        )

        # Las Vegas — 3BR house near Strip
        predict_revenue(
            market_key      = "las_vegas",
            bedrooms        = 3,
            baths           = 2.0,
            guests          = 6,
            amenities_count = 22,
            cleaning_fee    = 130,
            photos_count    = 25,
            latitude        = 36.1200,
            longitude       = -115.1728,
            min_nights      = 2,
            superhost       = 0,
            instant_book    = 1,
            rating_overall  = 4.8,
            recency_factor  = recency.get("las_vegas", 1.0),
            target_month    = 10,
        )

        # Los Angeles — 1BR apartment near beach
        predict_revenue(
            market_key      = "los_angeles",
            bedrooms        = 1,
            baths           = 1.0,
            guests          = 2,
            amenities_count = 14,
            cleaning_fee    = 75,
            photos_count    = 18,
            latitude        = 34.0195,
            longitude       = -118.4912,
            min_nights      = 1,
            superhost       = 1,
            instant_book    = 1,
            rating_overall  = 4.9,
            recency_factor  = recency.get("los_angeles", 1.0),
        )


if __name__ == "__main__":
    main()