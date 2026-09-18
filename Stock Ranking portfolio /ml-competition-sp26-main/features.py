"""
Feature engineering for the CSI500 stock-selection baseline.

A small set of classic technical features + cross-sectional ranks.  Students are
encouraged to extend this (add fundamentals, industry dummies, alternative data,
better cross-sectional normalization, etc.).

The target is the 5-trading-day forward return on the forward-adjusted close,
i.e. what the portfolio earns if you hold a $1 position from close(t) to close(t+5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# columns used downstream by the baseline
FEATURE_COLUMNS = [
    "ret_1d", 
    "ret_5d",
    "ret_10d", 
    "ret_20d",
    "ret_60d",
    "ret_5d_acc", 
    "vol_20d",
    "close_over_ma20",
    "close_over_ma60",
    "volume_z_20d", 
    "turnover_ma_20d", 
    "trend_diff",
    "rsi_dev",
    "momentum_accel",
    "vol_spike",
    "trend_strength",
    "ret_5d_rank",
    "ret_20d_rank",
    "vol_20d_rank", 
    "reversal_5d", 
    "reversal_10d", 
    "risk_adj_mom", 
    "vol_price",
]

###
#TARGET_COLUMN = "target_5d"
TARGET_COLUMN = "target_5d_excess"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)


    df["ret_3d"] = close.pct_change(3)
    df["ret_5d_acc"] = df["ret_5d"] - df["ret_3d"]

    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    df["trend_strength"] = df["ret_20d"] / (df["vol_20d"] + 1e-6)

    df["momentum_accel"] = df["ret_5d"] - df["ret_20d"]
    df["vol_spike"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_change"] = df["vol_20d"] - df["vol_20d"].rolling(10).mean()

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    df["trend_diff"] = df["close_over_ma20"] - df["close_over_ma60"]

    # --- NEW FEATURES ---

    # Mean reversion
    df["reversal_5d"] = -df["ret_5d"]
    df["reversal_10d"] = -df["ret_10d"]
    
    # Risk-adjusted momentum
    df["risk_adj_mom"] = df["ret_20d"] / (df["vol_20d"] + 1e-6)
    
    # Volume-price interaction
    df["vol_price"] = df["ret_5d"] * df["volume_z_20d"]

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df["rsi_dev"] = df["rsi_14"] - 50
    
    ###
    #df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    df["target_5d"] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank of selected features (values in [0, 1])."""
    for base in ["ret_5d", "ret_20d", "vol_20d"]:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )

    # # 2. Industry-Neutral Alpha (The Extension)
    # # Check if industry_code exists to prevent errors
    # if "industry_code" in panel.columns:
    #     # Step A: Calculate how much a stock beats its industry average
    #     panel["ind_rel_ret"] = panel.groupby(["date", "industry_code"])["ret_5d"].transform(
    #         lambda x: x - x.mean()
    #     )
    #     # Step B: Rank that relative return (Optional but recommended for XGBoost)
    #     panel["ind_rel_ret_rank"] = (
    #         panel.groupby("date")["ind_rel_ret"].rank(method="average", pct=True)
    #     )
        
    return panel

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target.

    Parameters
    ----------
    prices : DataFrame with columns [date, stock_code, open, close, high, low,
             volume, amount, turnover?]

    Returns
    -------
    DataFrame with FEATURE_COLUMNS and TARGET_COLUMN populated.  Rows where any
    feature is NaN (typically the first ~60 days per stock) are kept so callers
    can decide how to handle them.
    """
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(['stock_code', 'date'])
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )
    panel = _cross_sectional_ranks(panel)
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    """Rows usable for supervised training: all features present AND target present.

    The target for date t uses close(t+5), so rows within the last 5 trading
    days of the panel are dropped automatically (target is NaN there).
    """
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date (defaults to the latest date)."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df
