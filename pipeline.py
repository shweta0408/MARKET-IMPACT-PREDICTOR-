"""
Data Pipeline
-------------
• Synthetic labeled headline generator for training
• yfinance market data fetcher (when network available)
• News–price correlation builder
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


# ─── Synthetic headline templates ────────────────────────────────────────────

BULLISH_TEMPLATES = [
    "{company} beats Q{q} earnings by {pct}%, stock surges",
    "{company} revenue surges {pct}% year-over-year, analysts upgrade",
    "{company} announces record quarterly profit of ${amount}B",
    "Fed signals rate cuts as inflation eases, markets rally",
    "{company} secures ${amount}B contract, shares soar",
    "{company} AI breakthrough drives {pct}% stock surge",
    "Strong jobs report boosts market confidence, indices climb",
    "{company} dividend increased by {pct}%, bullish signal",
    "{company} buyback program of ${amount}B announced",
    "GDP growth exceeds expectations at {pct}%, economy booming",
    "{company} acquires rival for ${amount}B, synergies expected",
    "Oil prices rally {pct}% on OPEC production cut",
    "{company} FDA approval granted for blockbuster drug",
    "Consumer confidence index hits {val}-year high",
]

BEARISH_TEMPLATES = [
    "{company} misses Q{q} earnings, shares plunge {pct}%",
    "{company} lays off {val}% of workforce amid restructuring",
    "{company} faces ${amount}B fraud investigation, stock crashes",
    "Fed raises rates by {pct}bps, markets tumble",
    "{company} recalls {val}M products over safety concerns",
    "{company} Q{q} revenue disappoints, downgraded to sell",
    "Recession fears mount as GDP contracts {pct}%",
    "{company} loses ${amount}B lawsuit, shares collapse",
    "Inflation surges to {val}-year high, bearish outlook",
    "{company} guidance cut, stock plummets {pct}%",
    "Banking crisis deepens, {company} shares crash",
    "Trade war escalates, tariffs on ${amount}B of goods",
    "{company} CEO resigns amid scandal, investors flee",
    "Oil prices collapse {pct}% on demand concerns",
]

NEUTRAL_TEMPLATES = [
    "{company} announces Q{q} earnings in line with expectations",
    "{company} reports flat revenue growth, guidance maintained",
    "Federal Reserve holds rates steady, markets unmoved",
    "{company} completes merger review process",
    "Market trading range-bound ahead of jobs report",
    "{company} updates product roadmap at annual investor day",
    "Q{q} GDP growth meets consensus estimate of {pct}%",
    "{company} appoints new CFO, transition expected to be smooth",
    "Oil prices stable as supply and demand remain balanced",
    "{company} files quarterly 10-Q with no material changes",
    "Sector rotation continues as investors reassess portfolios",
    "{company} maintains annual guidance despite macro headwinds",
]

COMPANIES = ["Apple", "Microsoft", "Tesla", "Amazon", "Google", "Meta",
             "Nvidia", "JPMorgan", "Goldman Sachs", "ExxonMobil",
             "Pfizer", "Boeing", "Netflix", "Walmart", "Berkshire"]


def _fill(template: str) -> str:
    return template.format(
        company=random.choice(COMPANIES),
        q=random.randint(1, 4),
        pct=round(random.uniform(2, 45), 1),
        amount=round(random.uniform(0.5, 50), 1),
        val=random.randint(2, 30),
    )


def generate_synthetic_headlines(n_per_class: int = 400) -> pd.DataFrame:
    """Generate balanced synthetic training dataset."""
    rows = []
    for _ in range(n_per_class):
        rows.append({"headline": _fill(random.choice(BULLISH_TEMPLATES)), "true_label": "Bullish"})
    for _ in range(n_per_class):
        rows.append({"headline": _fill(random.choice(BEARISH_TEMPLATES)), "true_label": "Bearish"})
    for _ in range(n_per_class):
        rows.append({"headline": _fill(random.choice(NEUTRAL_TEMPLATES)), "true_label": "Neutral"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─── Market data (yfinance or synthetic fallback) ──────────────────────────

def fetch_market_data(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data. Returns None if yfinance unavailable."""
    try:
        import yfinance as yf
        end = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return _synthetic_market_data(ticker, days)


def _synthetic_market_data(ticker: str, days: int) -> pd.DataFrame:
    """Generate realistic-looking synthetic price series via GBM."""
    np.random.seed(abs(hash(ticker)) % 2**31)
    n = days
    price = 150.0
    mu, sigma = 0.0003, 0.018
    prices = [price]
    for _ in range(n - 1):
        ret = np.random.normal(mu, sigma)
        prices.append(prices[-1] * (1 + ret))

    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    closes = np.array(prices[:len(dates)])
    opens = closes * (1 + np.random.normal(0, 0.005, len(dates)))
    highs = np.maximum(closes, opens) * (1 + np.abs(np.random.normal(0, 0.008, len(dates))))
    lows = np.minimum(closes, opens) * (1 - np.abs(np.random.normal(0, 0.008, len(dates))))
    volume = np.random.randint(20_000_000, 120_000_000, len(dates))

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volume,
    }, index=dates)


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return/volatility columns to price dataframe."""
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_5d"] = df["daily_return"].rolling(5).std()
    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["above_sma20"] = (df["Close"] > df["sma_20"]).astype(int)
    df["above_sma50"] = (df["Close"] > df["sma_50"]).astype(int)
    df["rsi"] = _rsi(df["Close"])
    return df.dropna()


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
