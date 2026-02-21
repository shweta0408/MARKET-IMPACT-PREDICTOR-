"""
main.py — Run the full pipeline demo
=====================================
  python main.py
"""

import sys, os
sys.path.insert(0, "/home/claude")

import json
import numpy as np
import pandas as pd
from financial_sentiment.analyzer import SentimentPredictor
from financial_sentiment.data.pipeline import fetch_market_data, compute_returns


# ── Sample headlines ──────────────────────────────────────────────────────────
SAMPLE_HEADLINES = [
    "Apple beats Q3 earnings by 15%, iPhone sales surge to record high",
    "Federal Reserve raises interest rates by 75bps, markets tumble",
    "Tesla reports Q2 delivery numbers in line with analyst estimates",
    "Nvidia revenue skyrockets 220% on AI chip demand, stock soars",
    "Amazon lays off 18,000 employees amid economic slowdown",
    "Microsoft acquires gaming company for $68.7B in landmark deal",
    "GDP growth contracts 0.9% in Q2, recession fears mount",
    "Goldman Sachs upgrades S&P 500 target, bullish market outlook",
    "ExxonMobil profits collapse as oil prices plummet 30%",
    "FDA approves Pfizer's new drug, biotech stocks rally",
    "JPMorgan warns of coming credit crisis as defaults rise",
    "Google announces $70B buyback program and dividend increase",
    "Netflix subscriber growth disappoints, stock plunges 25%",
    "Consumer confidence hits 10-year high on strong jobs market",
    "Walmart reports record quarterly earnings, raises full-year guidance",
]


def run():
    print("\n🚀 Financial News Sentiment: Market Impact Predictor")
    print("="*60)

    # ── 1. Initialize and train ────────────────────────────────────────────
    sp = SentimentPredictor()
    metrics = sp.train(n_per_class=500, verbose=True)

    # ── 2. Analyze headlines ───────────────────────────────────────────────
    print("\n🔍 Analyzing sample financial headlines...")
    results = sp.analyze(SAMPLE_HEADLINES)
    sp.print_analysis(results)

    # ── 3. Get market data (synthetic) ────────────────────────────────────
    print("📈 Generating market data for dashboard...")
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA"]
    market_data = {}
    for t in tickers:
        df = sp.get_market_data(t, days=90)
        market_data[t] = {
            "dates": [str(d.date()) for d in df.index],
            "close": df["Close"].round(2).tolist(),
            "volume": df["Volume"].tolist(),
            "daily_return": df["daily_return"].round(4).tolist(),
            "rsi": df["rsi"].round(2).tolist(),
            "sma_20": df["sma_20"].round(2).tolist(),
        }

    # ── 4. Feature importance ──────────────────────────────────────────────
    feat_imp = sp.predictor.feature_importance()
    feat_imp_data = feat_imp.head(10).to_dict("records")

    # ── 5. Serialize results for dashboard ────────────────────────────────
    export = {
        "headlines": [
            {
                "headline": r["headline"],
                "sentiment": r["sentiment"],
                "compound": r["compound_score"],
                "market_impact": r["market_impact"],
                "confidence": r["confidence"],
                "bullish_prob": r["bullish_prob"],
                "neutral_prob": r["neutral_prob"],
                "bearish_prob": r["bearish_prob"],
            }
            for r in results
        ],
        "market_data": market_data,
        "feature_importance": feat_imp_data,
        "model_metrics": {
            "lr_accuracy": round(metrics["lr_report"]["accuracy"], 4),
            "rf_accuracy": round(metrics["rf_report"]["accuracy"], 4),
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
        },
        "sentiment_summary": {
            "Positive": sum(1 for r in results if r["sentiment"] == "Positive"),
            "Neutral": sum(1 for r in results if r["sentiment"] == "Neutral"),
            "Negative": sum(1 for r in results if r["sentiment"] == "Negative"),
        },
        "impact_summary": {
            "Bullish": sum(1 for r in results if r["market_impact"] == "Bullish"),
            "Neutral": sum(1 for r in results if r["market_impact"] == "Neutral"),
            "Bearish": sum(1 for r in results if r["market_impact"] == "Bearish"),
        },
    }

    out_path = "/home/claude/dashboard_data.json"
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n✅ Dashboard data exported → {out_path}")
    return export


if __name__ == "__main__":
    run()
