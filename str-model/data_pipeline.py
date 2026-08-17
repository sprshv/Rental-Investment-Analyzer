"""
Data Pipeline — Phase 2
Loads and cleans AirROI listings + calendar data for all markets.
Outputs:
  - cleaned listings DataFrame per market
  - market summary lookup table (feeds financial model)
  - seasonality curves per market (from calendar data)
  - saved to data/processed/ as CSVs
Usage: python data_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────

MARKETS = {
    "indianapolis":     "data/raw/indianapolis",
    "los_angeles":      "data/raw/los_angeles",
    "las_vegas":        "data/raw/las_vegas",
    "anaheim":          "data/raw/anaheim",
    "austin":           "data/raw/austin",
    "four_corners":     "data/raw/four_corners",
    "houston":          "data/raw/houston",
    "kissimmee":        "data/raw/kissimmee",
    "miami":            "data/raw/miami",
    "portland":         "data/raw/portland",
    "san_diego":        "data/raw/san_diego",
    "san_francisco":    "data/raw/san_francisco",
    "seattle":          "data/raw/seattle",
    "south_lake_tahoe": "data/raw/south_lake_tahoe",
    "washington":       "data/raw/washington",
}

# ── Regulatory config ─────────────────────────────────────────────────────────
# Used to add regulatory features to each listing so the model learns
# that cap markets systematically earn less per month.

STR_REGULATIONS = {
    "los_angeles":      {"cap_nights": 120, "permit_required": 1, "owner_occupancy": 1},
    "san_francisco":    {"cap_nights": 90,  "permit_required": 1, "owner_occupancy": 1},
    "seattle":          {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "portland":         {"cap_nights": None,"permit_required": 1, "owner_occupancy": 1},
    "washington":       {"cap_nights": 90,  "permit_required": 1, "owner_occupancy": 1},
    "austin":           {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "indianapolis":     {"cap_nights": None,"permit_required": 0, "owner_occupancy": 0},
    "las_vegas":        {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "miami":            {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "houston":          {"cap_nights": None,"permit_required": 0, "owner_occupancy": 0},
    "anaheim":          {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "kissimmee":        {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "san_diego":        {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "south_lake_tahoe": {"cap_nights": None,"permit_required": 1, "owner_occupancy": 0},
    "four_corners":     {"cap_nights": None,"permit_required": 0, "owner_occupancy": 0},
}

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Columns we actually need from listings
LISTING_COLS = [
    "listing_id", "listing_type", "room_type",
    "latitude", "longitude",
    "guests", "bedrooms", "beds", "baths",
    "superhost", "instant_book", "professional_management",
    "guest_favorite", "min_nights",
    "amenities", "cleaning_fee", "extra_guest_fee",
    "photos_count",                              # quality signal
    "num_reviews", "rating_overall", "rating_cleanliness",
    "rating_location", "rating_value",
    # TTM (trailing 12 months) — main targets
    "ttm_revenue", "ttm_avg_rate", "ttm_occupancy",
    "ttm_adjusted_occupancy", "ttm_avg_length_of_stay",
    "ttm_reserved_days", "ttm_revpar",
    # L90D (last 90 days) — recency signal
    "l90d_revenue", "l90d_avg_rate", "l90d_occupancy",
]

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_listings(market_key):
    path = os.path.join(MARKETS[market_key], "listings.csv")
    # doublequote=True handles description/amenities fields that contain
    # commas, newlines, and HTML — without it the parser shifts columns
    df = pd.read_csv(
        path,
        encoding="utf-8",
        quotechar='"',
        doublequote=True,
        on_bad_lines="skip",
        engine="python",
    )

    # Keep only columns we need (drop any missing ones gracefully)
    cols = [c for c in LISTING_COLS if c in df.columns]
    df = df[cols].copy()

    df["market"] = market_key
    return df


def load_calendar(market_key):
    path = os.path.join(MARKETS[market_key], "past_calendar_rates.csv")
    df = pd.read_csv(path)
    df["market"] = market_key
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_listings(df, market_key):
    # Entire home only — you're buying whole properties
    if "room_type" in df.columns:
        df = df[df["room_type"].str.lower().str.contains("entire", na=False)]

    # Drop listings with no revenue data — inactive or new
    df = df[df["ttm_revenue"].notna() & (df["ttm_revenue"] > 0)]

    # Drop obvious outliers
    df = df[df["ttm_avg_rate"].between(30, 2000)]
    df = df[df["ttm_occupancy"].between(0.01, 0.99)]

    # Bedrooms: fill missing with 1, cap at 6
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(1).clip(1, 6)
    df["bedrooms"] = df["bedrooms"].astype(int)

    # Baths and guests
    df["baths"]  = pd.to_numeric(df["baths"],  errors="coerce").fillna(1)
    df["guests"] = pd.to_numeric(df["guests"], errors="coerce").fillna(2)

    # Cleaning fee — fill missing with 0
    df["cleaning_fee"] = pd.to_numeric(df["cleaning_fee"], errors="coerce").fillna(0)

    # Photos count — quality signal, fill missing with median
    if "photos_count" in df.columns:
        df["photos_count"] = pd.to_numeric(df["photos_count"], errors="coerce")
        df["photos_count"] = df["photos_count"].fillna(df["photos_count"].median())
    else:
        df["photos_count"] = 10  # fallback default

    # Boolean fields
    for col in ["superhost", "instant_book", "professional_management", "guest_favorite"]:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "t": 1, "f": 0}).fillna(0).astype(int)

    # Amenities count
    if "amenities" in df.columns:
        df["amenities_count"] = df["amenities"].fillna("").apply(
            lambda x: len(str(x).split(",")) if x else 0
        )

    # Avg stay length — fill missing with market median
    df["ttm_avg_length_of_stay"] = pd.to_numeric(
        df["ttm_avg_length_of_stay"], errors="coerce"
    )
    df["ttm_avg_length_of_stay"] = df["ttm_avg_length_of_stay"].fillna(
        df["ttm_avg_length_of_stay"].median()
    )

    # Rating — fill missing with median
    df["rating_overall"] = pd.to_numeric(df["rating_overall"], errors="coerce")
    df["rating_overall"] = df["rating_overall"].fillna(df["rating_overall"].median())

    # ── Monthly revenue — the model target ───────────────────────────────
    # Use ttm_reserved_days * ttm_avg_rate / 12 for cleaner monthly figure
    # Falls back to ttm_revenue / 12 if reserved_days missing
    if "ttm_reserved_days" in df.columns and "ttm_avg_rate" in df.columns:
        df["monthly_revenue"] = (
            pd.to_numeric(df["ttm_reserved_days"], errors="coerce") *
            pd.to_numeric(df["ttm_avg_rate"],      errors="coerce") / 12
        ).fillna(df["ttm_revenue"] / 12)
    else:
        df["monthly_revenue"] = df["ttm_revenue"] / 12

    # Drop rows where monthly revenue is clearly wrong
    df = df[df["monthly_revenue"] > 0]

    # ── Regulatory features ───────────────────────────────────────────────
    reg = STR_REGULATIONS.get(market_key, {"cap_nights": None, "permit_required": 0, "owner_occupancy": 0})
    df["permit_required"]    = reg["permit_required"]
    df["owner_occupancy"]    = reg["owner_occupancy"]
    df["has_night_cap"]      = 1 if reg["cap_nights"] else 0
    df["night_cap"]          = reg["cap_nights"] if reg["cap_nights"] else 365

    df = df.reset_index(drop=True)
    return df


# ── Market summary ────────────────────────────────────────────────────────────

def build_market_summary(df, market_key):
    """Build lookup table that feeds the financial model."""
    summary = df.groupby("bedrooms").agg(
        median_adr             = ("ttm_avg_rate",       "median"),
        median_occupancy       = ("ttm_occupancy",      "median"),
        median_adj_occupancy   = ("ttm_adjusted_occupancy", "median"),
        median_annual_revenue  = ("ttm_revenue",        "median"),
        median_monthly_revenue = ("monthly_revenue",    "median"),
        median_avg_stay        = ("ttm_avg_length_of_stay", "median"),
        median_cleaning_fee    = ("cleaning_fee",       "median"),
        p25_revenue            = ("ttm_revenue", lambda x: x.quantile(0.25)),
        p75_revenue            = ("ttm_revenue", lambda x: x.quantile(0.75)),
        p25_monthly            = ("monthly_revenue", lambda x: x.quantile(0.25)),
        p75_monthly            = ("monthly_revenue", lambda x: x.quantile(0.75)),
        sample_size            = ("listing_id",         "count"),
    ).round(2)

    summary["market"]         = market_key
    reg = STR_REGULATIONS.get(market_key, {})
    summary["permit_required"] = reg.get("permit_required", 0)
    summary["owner_occupancy"] = reg.get("owner_occupancy", 0)
    summary["has_night_cap"]   = 1 if reg.get("cap_nights") else 0
    summary["night_cap"]       = reg.get("cap_nights") or 365
    return summary


# ── Seasonality ───────────────────────────────────────────────────────────────

def build_seasonality(cal_df, market_key):
    """
    Build a monthly seasonality index from calendar data.
    Index of 1.0 = average month. 1.3 = 30% above average. 0.7 = 30% below.
    """
    monthly = cal_df.groupby("month").agg(
        avg_revenue     = ("revenue",   "mean"),
        avg_occupancy   = ("occupancy", "mean"),
        avg_rate        = ("rate_avg",  "mean"),
    ).round(2)

    # Normalize to index
    monthly["revenue_index"]   = monthly["avg_revenue"]   / monthly["avg_revenue"].mean()
    monthly["occupancy_index"] = monthly["avg_occupancy"] / monthly["avg_occupancy"].mean()
    monthly["rate_index"]      = monthly["avg_rate"]      / monthly["avg_rate"].mean()
    monthly["market"]          = market_key

    return monthly.round(3)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    all_listings   = []
    all_summaries  = []
    all_seasonality= []

    for market_key in MARKETS:
        print(f"\n{'─'*50}")
        print(f"  Processing {market_key}...")
        print(f"{'─'*50}")

        # ── Listings ──────────────────────────────────────────────────────
        try:
            raw = load_listings(market_key)
            print(f"  Raw listings loaded:    {len(raw):>5} rows")

            cleaned = clean_listings(raw, market_key)
            print(f"  After cleaning:         {len(cleaned):>5} rows")

            # Save cleaned listings
            out_path = os.path.join(PROCESSED_DIR, f"{market_key}_listings.csv")
            cleaned.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}")

            # Build and print summary
            summary = build_market_summary(cleaned, market_key)
            all_summaries.append(summary.reset_index())

            reg = STR_REGULATIONS.get(market_key, {})
            cap = reg.get("cap_nights")
            reg_str = f"cap={cap}nts" if cap else "no cap"
            print(f"\n  Market summary by bedrooms  [{reg_str}, permit={'yes' if reg.get('permit_required') else 'no'}]:")
            print(f"  {'BR':<4} {'Median ADR':>10} {'Occupancy':>10} {'Monthly Rev':>12} {'Ann Revenue':>12} {'N':>5}")
            print(f"  {'─'*57}")
            for br, row in summary.iterrows():
                print(f"  {br:<4} ${row['median_adr']:>9,.0f} {row['median_occupancy']*100:>9.1f}% ${row['median_monthly_revenue']:>11,.0f} ${row['median_annual_revenue']:>11,.0f} {row['sample_size']:>5.0f}")

            all_listings.append(cleaned)

        except FileNotFoundError:
            print(f"  ⚠ listings.csv not found for {market_key} — skipping")

        # ── Calendar ──────────────────────────────────────────────────────
        try:
            cal = load_calendar(market_key)
            print(f"\n  Calendar rows loaded:   {len(cal):>5}")

            seasonality = build_seasonality(cal, market_key)
            all_seasonality.append(seasonality.reset_index())

            print(f"\n  Seasonality index by month:")
            print(f"  {'Mo':<4} {'Rev Index':>10} {'Occ Index':>10} {'Rate Index':>10}")
            print(f"  {'─'*38}")
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            for mo, row in seasonality.iterrows():
                print(f"  {months[mo-1]:<4} {row['revenue_index']:>10.2f} {row['occupancy_index']:>10.2f} {row['rate_index']:>10.2f}")

        except FileNotFoundError:
            print(f"  ⚠ past_calendar_rates.csv not found for {market_key} — skipping")

    # ── Save combined outputs ─────────────────────────────────────────────
    if all_listings:
        combined = pd.concat(all_listings, ignore_index=True)
        combined.to_csv(os.path.join(PROCESSED_DIR, "all_listings.csv"), index=False)
        print(f"\n\n  ✓ Combined listings saved → data/processed/all_listings.csv  ({len(combined)} rows)")

    if all_summaries:
        summary_df = pd.concat(all_summaries, ignore_index=True)
        summary_df.to_csv(os.path.join(PROCESSED_DIR, "market_summary.csv"), index=False)
        print(f"  ✓ Market summary saved  → data/processed/market_summary.csv")

    if all_seasonality:
        seas_df = pd.concat(all_seasonality, ignore_index=True)
        seas_df.to_csv(os.path.join(PROCESSED_DIR, "seasonality.csv"), index=False)
        print(f"  ✓ Seasonality saved     → data/processed/seasonality.csv")

    print(f"\n  Pipeline complete.\n")
    return combined if all_listings else None


if __name__ == "__main__":
    run_pipeline()