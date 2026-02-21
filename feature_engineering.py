"""
Feature Engineering
-------------------
Converts raw headlines + sentiment scores into ML-ready feature vectors.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any


# ── Financial entity patterns ─────────────────────────────────────────────────
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
PERCENTAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DOLLAR_RE = re.compile(r"\$(\d+(?:\.\d+)?)\s*([BMK]?)")

MAGNITUDE_WORDS = {
    "billion": 1e9, "million": 1e6, "trillion": 1e12,
    "b": 1e9, "m": 1e6, "k": 1e3,
}

SECTOR_KEYWORDS = {
    "tech": ["apple", "microsoft", "google", "meta", "nvidia", "semiconductor",
             "software", "ai", "cloud", "chip", "tech", "technology"],
    "finance": ["bank", "fed", "federal reserve", "interest rate", "treasury",
                "goldman", "jpmorgan", "lending", "credit", "mortgage"],
    "energy": ["oil", "gas", "opec", "exxon", "chevron", "renewable", "solar",
               "wind", "energy", "petroleum"],
    "health": ["fda", "drug", "pharma", "vaccine", "clinical", "biotech",
               "hospital", "healthcare", "medical"],
    "macro": ["gdp", "inflation", "cpi", "unemployment", "recession",
              "economy", "growth", "policy", "tariff", "trade"],
}

URGENCY_WORDS = ["breaking", "urgent", "alert", "flash", "just in",
                 "developing", "exclusive", "first quarter", "q1", "q2", "q3", "q4"]


class FeatureEngineer:
    def __init__(self):
        pass

    def extract(self, headline: str, scores: Dict[str, float]) -> Dict[str, Any]:
        """Return a flat feature dict for a single headline."""
        text_lower = headline.lower()
        features = {}

        # ── Sentiment features ────────────────────────────────────────────────
        features["compound"] = scores["compound"]
        features["pos"] = scores["pos"]
        features["neg"] = scores["neg"]
        features["neu"] = scores["neu"]
        features["sentiment_spread"] = scores["pos"] - scores["neg"]
        features["sentiment_abs"] = abs(scores["compound"])

        # ── Text structural features ──────────────────────────────────────────
        words = text_lower.split()
        features["word_count"] = len(words)
        features["char_count"] = len(headline)
        features["has_question"] = int("?" in headline)
        features["has_exclamation"] = int("!" in headline)
        features["caps_ratio"] = sum(1 for c in headline if c.isupper()) / max(len(headline), 1)

        # ── Financial numeric features ────────────────────────────────────────
        pcts = PERCENTAGE_RE.findall(text_lower)
        features["pct_mentioned"] = int(bool(pcts))
        features["max_pct"] = max((float(p) for p in pcts), default=0.0)

        features["dollar_mentioned"] = int(bool(DOLLAR_RE.search(headline)))
        features["has_ticker"] = int(bool(TICKER_RE.search(headline)))

        # ── Urgency & recency ─────────────────────────────────────────────────
        features["urgency"] = int(any(w in text_lower for w in URGENCY_WORDS))

        # ── Sector encoding (one-hot) ─────────────────────────────────────────
        for sector, kws in SECTOR_KEYWORDS.items():
            features[f"sector_{sector}"] = int(any(kw in text_lower for kw in kws))

        # ── Keyword category counts ───────────────────────────────────────────
        bullish_kws = ["beat", "surge", "soar", "record", "profit", "growth",
                       "upgrade", "rally", "boom", "outperform", "bullish",
                       "buyback", "dividend", "exceed", "strong earnings"]
        bearish_kws = ["miss", "crash", "plunge", "loss", "decline", "downgrade",
                       "layoff", "bankruptcy", "fraud", "warning", "weak",
                       "disappoint", "recall", "investigation", "bearish"]

        features["bullish_kw_count"] = sum(kw in text_lower for kw in bullish_kws)
        features["bearish_kw_count"] = sum(kw in text_lower for kw in bearish_kws)
        features["kw_ratio"] = (
            (features["bullish_kw_count"] - features["bearish_kw_count"])
            / max(features["bullish_kw_count"] + features["bearish_kw_count"], 1)
        )

        return features

    def headlines_to_dataframe(
        self, headlines: List[str], scores_list: List[Dict]
    ) -> pd.DataFrame:
        rows = [self.extract(h, s) for h, s in zip(headlines, scores_list)]
        return pd.DataFrame(rows)

    @property
    def feature_columns(self) -> List[str]:
        dummy = self.extract("Test headline", {"compound": 0, "pos": 0, "neg": 0, "neu": 1})
        return list(dummy.keys())
