"""
Financial News Sentiment: Market Impact Predictor
==================================================
Main orchestrator — ties sentiment engine, feature engineering, and ML models together.

Usage:
    from financial_sentiment.analyzer import SentimentPredictor
    sp = SentimentPredictor()
    sp.train()
    results = sp.analyze(["Apple beats Q3 earnings by 12%, stock surges"])
    for r in results:
        print(r)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from financial_sentiment.utils.sentiment_engine import FinancialSentimentAnalyzer
from financial_sentiment.utils.feature_engineering import FeatureEngineer
from financial_sentiment.models.predictor import MarketImpactPredictor, LABEL_MAP
from financial_sentiment.data.pipeline import (
    generate_synthetic_headlines, fetch_market_data, compute_returns
)


class SentimentPredictor:
    """End-to-end pipeline from raw headline → analysis result."""

    def __init__(self):
        self.sentiment = FinancialSentimentAnalyzer()
        self.features = FeatureEngineer()
        self.predictor = MarketImpactPredictor()
        self._trained = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, n_per_class: int = 600, verbose: bool = True) -> Dict:
        """Train on synthetic data (replace with real labeled data for production)."""
        if verbose:
            print("📊 Generating synthetic training data...")

        df_raw = generate_synthetic_headlines(n_per_class)

        # Score each headline
        scores_list = [self.sentiment.polarity_scores(h) for h in df_raw["headline"]]

        # Engineer features
        feat_df = self.features.headlines_to_dataframe(df_raw["headline"].tolist(), scores_list)

        # Add label column
        label_map = {"Bullish": 2, "Neutral": 1, "Bearish": 0}
        y = df_raw["true_label"].map(label_map).values

        if verbose:
            print(f"   Training on {len(feat_df)} examples across 3 classes...")

        metrics = self.predictor.fit(feat_df, y)
        self._trained = True

        if verbose:
            lr_acc = metrics["lr_report"]["accuracy"]
            rf_acc = metrics["rf_report"]["accuracy"]
            print(f"✅ Training complete! LR accuracy: {lr_acc:.1%}  |  RF accuracy: {rf_acc:.1%}")

        return metrics

    # ── Inference ──────────────────────────────────────────────────────────────

    def analyze(self, headlines: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a list of headlines.
        Returns list of result dicts, one per headline.
        """
        results = []
        for headline in headlines:
            scores = self.sentiment.polarity_scores(headline)
            sentiment_label = self.sentiment.classify(scores["compound"])

            feat = self.features.extract(headline, scores)
            feat_df = pd.DataFrame([feat])

            prediction = self.predictor.predict(feat_df)[0]

            results.append({
                "headline": headline,
                # Sentiment
                "sentiment": sentiment_label,
                "compound_score": scores["compound"],
                "pos_score": scores["pos"],
                "neg_score": scores["neg"],
                "neu_score": scores["neu"],
                # Market Impact
                "market_impact": prediction["label"],
                "confidence": prediction["confidence"],
                "confidence_pct": f"{prediction['confidence']:.1%}",
                "bullish_prob": prediction["ensemble_proba"][2],
                "neutral_prob": prediction["ensemble_proba"][1],
                "bearish_prob": prediction["ensemble_proba"][0],
                # Debug
                "features": feat,
                "_raw_prediction": prediction,
            })

        return results

    def analyze_single(self, headline: str) -> Dict[str, Any]:
        return self.analyze([headline])[0]

    # ── Market data ────────────────────────────────────────────────────────────

    def get_market_data(self, ticker: str = "AAPL", days: int = 90) -> pd.DataFrame:
        df = fetch_market_data(ticker, days)
        return compute_returns(df)

    # ── Report ─────────────────────────────────────────────────────────────────

    def print_analysis(self, results: List[Dict]):
        print("\n" + "="*80)
        print("  FINANCIAL NEWS SENTIMENT: MARKET IMPACT PREDICTOR")
        print("="*80)
        for r in results:
            icon = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(r["sentiment"], "⚪")
            impact_icon = {"Bullish": "📈", "Bearish": "📉", "Neutral": "⚖️"}.get(r["market_impact"], "")
            print(f"\n📰 {r['headline']}")
            print(f"   Sentiment : {icon} {r['sentiment']} (compound: {r['compound_score']:+.3f})")
            print(f"   Impact    : {impact_icon} {r['market_impact']}  |  Confidence: {r['confidence_pct']}")
            print(f"   Probabilities → 📈 Bullish: {r['bullish_prob']:.1%}  "
                  f"⚖️ Neutral: {r['neutral_prob']:.1%}  "
                  f"📉 Bearish: {r['bearish_prob']:.1%}")
        print("="*80 + "\n")
