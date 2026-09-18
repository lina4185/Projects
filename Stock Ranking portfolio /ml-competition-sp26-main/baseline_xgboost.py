"""
XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load data/prices.parquet
2. Build features + 5-day forward target (features.py)
3. Train XGBoost on all but the last `EMBARGO_DAYS` training rows
4. Validate on those held-out rows (reports rank IC as sanity check)
5. Predict on the most recent date
6. Build a portfolio: top-K names, score-weighted with the 10% cap

Usage
-----
  python baseline_xgboost.py                       # predict from latest data
  python baseline_xgboost.py --as-of 20260503      # predict as of a given date
  python baseline_xgboost.py --top-k 50 --out submissions/week1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 15               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df = train_df.sort_values("date")
    val_df = val_df.sort_values("date")

    train_group = train_df.groupby("date").size().to_list()
    val_group = val_df.groupby("date").size().to_list()

    assert sum(train_group) == len(train_df)
    assert sum(val_group) == len(val_df)


    model = xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="rmse",
    
        n_estimators=300,
        learning_rate=0.03,
    
        max_depth=4,
        min_child_weight=20,
    
        subsample=0.7,
        colsample_bytree=0.7,
    
        reg_alpha=0.5,
        reg_lambda=5.0,
    
        gamma=0.1,
    
        random_state=42,
        n_jobs=-1,

        early_stopping_rounds=50
    )

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        group=train_group,                   
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        eval_group=[val_group],              
        verbose=False,
    )

    return model
    
    # model = xgb.XGBRegressor(
    #     n_estimators=300,
    #     max_depth=4,
    #     learning_rate=0.05,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     min_child_weight=20,
    #     reg_lambda=1.0,
    #     tree_method="hist",
    #     n_jobs=-1,
    #     early_stopping_rounds=30,
    #     random_state=42
    # )
    
    # model.fit(
    #     train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
    #     eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
    #     verbose=False,
    # )
    # return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")

def build_portfolio(scores: pd.Series, top_k: int = 35) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k)

    s = chosen - chosen.min() + 1e-6
    s = np.sqrt(s)   

    score_w = s / s.sum()
    equal_w = pd.Series(1/len(chosen), index=chosen.index)

    alpha = 0.6
    w = alpha * score_w + (1 - alpha) * equal_w

    # cap at 10%
    for _ in range(50):
        over = w > 0.10
        if not over.any():
            break
    
        excess = (w[over] - 0.10).sum()
        w[over] = 0.10
    
        free = ~over
        if not free.any():
            break 
        
        ranks = np.arange(free.sum(), 0, -1)
        rank_weights = pd.Series(ranks, index=w[free].index)
        rank_weights = rank_weights / rank_weights.sum()
    
        w[free] += excess * rank_weights

    return w

##
def compute_portfolio_returns(weights_df, panel, horizon=5):

    # --- shift weights forward ---
    trading_dates = np.sort(panel["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    weights_df = weights_df.copy()
    weights_df["date_idx"] = weights_df["date"].map(date_to_idx)
    weights_df["future_idx"] = weights_df["date_idx"] + horizon

    # map to actual future date
    weights_df["future_date"] = weights_df["future_idx"].map(
        lambda x: trading_dates[x] if x < len(trading_dates) else pd.NaT
    )

    weights_df = weights_df.dropna(subset=["future_date"])

    # --- merge with realized returns ---
    df = weights_df.merge(
        panel[["date", "stock_code", "target_5d"]],
        left_on=["date", "stock_code"],
        right_on=["date", "stock_code"],
        how="inner"
    )

    df["weighted_return"] = df["weight"] * df["target_5d"]

    portfolio_returns = (
        df.groupby("date")["weighted_return"]
        .sum()
        .reset_index(name="portfolio_return")
    )

    return portfolio_returns

def compute_total_return(portfolio_returns):
    """
    portfolio_returns: DataFrame with [date, portfolio_return]
    """
    return (1 + portfolio_returns["portfolio_return"]).prod() - 1

def compute_index_returns(index_df, horizon=5):
    index_df = index_df.sort_values("date").copy()
    
    index_df["index_return"] = (
        index_df["close"].shift(-horizon) / index_df["close"] - 1
    )
    
    return index_df[["date", "index_return"]].dropna()
    
def compute_excess_return(portfolio_returns, index_returns):
    df = portfolio_returns.merge(index_returns, on="date")

    df["excess"] = df["portfolio_return"] - df["index_return"]

    total_portfolio = (1 + df["portfolio_return"]).prod() - 1
    total_index = (1 + df["index_return"]).prod() - 1
    total_excess = total_portfolio - total_index

    return {
        "portfolio_total_return": total_portfolio,
        "index_total_return": total_index,
        "excess_return": total_excess
    }

def evaluate_strategy(weights_df, panel, index_df):
    port_ret = compute_portfolio_returns(weights_df, panel)
    idx_ret = compute_index_returns(index_df)

    results = compute_excess_return(port_ret, idx_ret)

    print(f"Portfolio Return: {results['portfolio_total_return']:.4f}")
    print(f"Index Return:     {results['index_total_return']:.4f}")
    print(f"Excess Return:    {results['excess_return']:.4f}")

    return results
####
def backtest(model, panel, dates, top_k):
    all_weights = []

    for d in dates:
        pred_df = prediction_frame(panel, as_of=d)
        if pred_df.empty:
            continue

        pred_df = pred_df.copy()
        pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])

        scores = pred_df.set_index("stock_code")["score"]

        scores = scores.clip(-3, 3)
        scores = (scores - scores.mean()) / scores.std()

        weights = build_portfolio(scores, top_k=top_k)

        tmp = pd.DataFrame({
            "date": d,
            "stock_code": weights.index,
            "weight": weights.values
        })

        all_weights.append(tmp)

    weights_df = pd.concat(all_weights, ignore_index=True)
    return weights_df
####

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    panel = build_features(prices)

    #####
    # --- Load index and compute returns ---
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df = compute_index_returns(index_df)
    
    # --- Merge into panel ---
    panel = panel.merge(index_df, on="date", how="left")
    
    # --- Create excess return target ---
    panel["target_5d_excess"] = panel["target_5d"] - panel["index_return"]

    ####


    # Bound training data so backtesting with --as-of doesn't leak future rows.
    # Training uses features from date t with target = close(t+FORWARD_HORIZON),
    # so we cap training dates at as_of - FORWARD_HORIZON trading days.

    #as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    
    trading_dates = np.sort(panel["date"].unique())

    if args.as_of:
        as_of_ts = pd.Timestamp(args.as_of)
    else:
        as_of_ts = pd.Timestamp(trading_dates[-(FORWARD_HORIZON + 1)])
        
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    # Time-based split with embargo:
    #   [ ... train ... | embargo (discarded) | val (last VAL_DAYS) ]
    # The embargo prevents training labels (5-day forward) from reaching into
    # dates whose prices also feed the validation features.
    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")
    
    print(">> Training XGBoost")
    model = train_model(train_df, val_df)

    val_pred = model.predict(val_df[FEATURE_COLUMNS])

    val_df = val_df.copy()
    val_df["pred"] = val_pred

    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")

    for k in [10, 20, 30, 50]:
        top_k = []
        for d in val_df["date"].unique():
            daily = val_df[val_df["date"] == d]
    
            if len(daily) < k:
                continue
    
            top = daily.nlargest(k, "pred")
            top_k.append(top[TARGET_COLUMN].mean())
    
        print(f"   Top{k} avg return: {np.mean(top_k):.4f}")
####
    # print(">> Predicting portfolio")
    # # pred_df = prediction_frame(panel, as_of=args.as_of)
    # pred_df = prediction_frame(panel, as_of=as_of_ts)
    # if pred_df.empty:
    #     raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    # pred_date = pred_df["date"].iloc[0]
    # print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    # pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    # scores = pred_df.set_index("stock_code")["score"]
    # scores = scores.clip(-3, 3)
    # scores = (scores - scores.mean()) / scores.std()
    # weights = build_portfolio(scores, top_k=args.top_k)

    print(">> Running backtest on validation period")
    
    val_dates = np.sort(val_df["date"].unique())
    
    weights_df = backtest(model, panel, val_dates, top_k=args.top_k)


    port_ret = compute_portfolio_returns(weights_df, panel)

    print("\nBacktest summary:")
    print("Mean return:", port_ret["portfolio_return"].mean())
    print("Std:", port_ret["portfolio_return"].std())
    print("Num periods:", len(port_ret))

    # print("\nTop 10 scores:")
    # print(scores.sort_values(ascending=False).head(10))

    # print("\nTop 10 weights:")
    # print(weights.sort_values(ascending=False).head(10))

    # out_path = Path(args.out)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    # out.to_csv(out_path, index=False)
    # print(f">> Wrote {len(out)} names to {out_path}")
    # print(f"   weight summary: min={out['weight'].min():.4f} "
    #       f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    # #### --- Build weights_df for this single date ---
    # weights_df = pd.DataFrame({
    #     "date": pred_date,
    #     "stock_code": weights.index,
    #     "weight": weights.values
    # })

    weights_df = backtest(model, panel, val_dates, top_k=args.top_k)

    print("\n>> Evaluating backtest")
    
    port_ret = compute_portfolio_returns(weights_df, panel)
    idx_ret = index_df.copy()
    
    df_eval = port_ret.merge(idx_ret, on="date")
    df_eval["excess"] = df_eval["portfolio_return"] - df_eval["index_return"]
    
    print("\nBacktest performance:")
    print("Mean portfolio return:", df_eval["portfolio_return"].mean())
    print("Mean index return:    ", df_eval["index_return"].mean())
    print("Mean excess return:   ", df_eval["excess"].mean())
    
    print("\nCumulative return:")
    port_total = (1 + df_eval["portfolio_return"]).prod() - 1
    idx_total = (1 + df_eval["index_return"]).prod() - 1
    
    print("Portfolio:", port_total)
    print("Index:    ", idx_total)
    print("Excess:   ", port_total - idx_total)

    print("\n>> Generating submission for latest date")

    pred_df = prediction_frame(panel, as_of=as_of_ts)
    
    pred_df = pred_df.copy()
    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])
    
    scores = pred_df.set_index("stock_code")["score"]
    scores = scores.clip(-3, 3)
    scores = (scores - scores.mean()) / scores.std()
    
    weights = build_portfolio(scores, top_k=args.top_k)
    
    print("\nTop 10 weights (submission):")
    print(weights.sort_values(ascending=False).head(10))
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out = pd.DataFrame({
        "stock_code": weights.index,
        "weight": weights.values
    })
    
    out.to_csv(out_path, index=False)
    
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    # print("\nSample portfolio (first validation date):")
    # first_date = weights_df["date"].min()
    # sample = weights_df[weights_df["date"] == first_date]
    
    # print(sample.sort_values("weight", ascending=False).head(10))


    # # --- Load index ---
    # index_df = pd.read_parquet(DATA_DIR / "index.parquet")

    # # --- DEBUG: check date alignment ---
    # print("\n=== DEBUG ===")
    # print("Prediction date:", pred_date)
    
    # valid_target_dates = panel.dropna(subset=["target_5d"])["date"]
    # print("Last valid target date:", valid_target_dates.max())
    # print("====================\n")
    
    
    # # --- Evaluate ---
    # results = evaluate_strategy(weights_df, panel, index_df)

    # print("\n=== Evaluation (single period) ===")
    # print(f"Portfolio Return: {results['portfolio_total_return']:.4f}")
    # print(f"Index Return:     {results['index_total_return']:.4f}")
    # print(f"Excess Return:    {results['excess_return']:.4f}")

###


if __name__ == "__main__":
    main()
