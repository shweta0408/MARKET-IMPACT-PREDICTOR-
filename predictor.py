"""
Market Impact Predictor
-----------------------
Logistic Regression + Random Forest ensemble.
Labels:  0=Bearish  1=Neutral  2=Bullish
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any
import json, os, pickle


LABEL_MAP = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
LABEL_INV = {"Bearish": 0, "Neutral": 1, "Bullish": 2}


def _compound_to_label(compound: float, threshold: float = 0.08) -> int:
    """Heuristic: map VADER compound → market impact label for synthetic training."""
    if compound >= threshold:
        return 2  # Bullish
    elif compound <= -threshold:
        return 0  # Bearish
    return 1  # Neutral


class MarketImpactPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42
        )
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        self._fitted = False
        self._feature_names: List[str] = []

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train both models. Returns evaluation metrics."""
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.lr.fit(X_train_s, y_train)
        self.rf.fit(X_train, y_train)  # RF doesn't need scaling

        self._fitted = True

        lr_pred = self.lr.predict(X_test_s)
        rf_pred = self.rf.predict(X_test)

        return {
            "lr_report": classification_report(
                y_test, lr_pred, target_names=["Bearish", "Neutral", "Bullish"],
                output_dict=True
            ),
            "rf_report": classification_report(
                y_test, rf_pred, target_names=["Bearish", "Neutral", "Bullish"],
                output_dict=True
            ),
            "lr_confusion": confusion_matrix(y_test, lr_pred).tolist(),
            "rf_confusion": confusion_matrix(y_test, rf_pred).tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def fit_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convenience wrapper: df must have 'compound' col and all feature cols."""
        y = df["compound"].apply(_compound_to_label).values
        feature_cols = [c for c in df.columns if c != "compound" and not c.startswith("_")]
        return self.fit(df[feature_cols], y)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns list of dicts per row:
        {
          label: str, label_id: int,
          confidence: float,
          lr_proba: list, rf_proba: list,
          ensemble_proba: list
        }
        """
        if not self._fitted:
            # Return heuristic predictions when model not trained
            return [self._heuristic_predict(row) for _, row in X.iterrows()]

        X_arr = X[self._feature_names].values.astype(float)
        X_scaled = self.scaler.transform(X_arr)

        lr_proba = self.lr.predict_proba(X_scaled)
        rf_proba = self.rf.predict_proba(X_arr)
        ensemble = (0.4 * lr_proba + 0.6 * rf_proba)

        results = []
        for i in range(len(X_arr)):
            label_id = int(np.argmax(ensemble[i]))
            confidence = float(ensemble[i][label_id])
            results.append({
                "label": LABEL_MAP[label_id],
                "label_id": label_id,
                "confidence": round(confidence, 4),
                "lr_proba": [round(p, 4) for p in lr_proba[i]],
                "rf_proba": [round(p, 4) for p in rf_proba[i]],
                "ensemble_proba": [round(p, 4) for p in ensemble[i]],
            })
        return results

    def predict_single(self, feature_dict: Dict) -> Dict[str, Any]:
        df = pd.DataFrame([feature_dict])
        return self.predict(df)[0]

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_predict(row) -> Dict[str, Any]:
        compound = float(row.get("compound", 0))
        kw = float(row.get("kw_ratio", 0))
        combined = 0.7 * compound + 0.3 * kw
        if combined >= 0.08:
            label_id = 2
        elif combined <= -0.08:
            label_id = 0
        else:
            label_id = 1
        prob = abs(combined)
        confidence = min(0.95, 0.5 + prob * 1.5)
        proba = [0.0, 0.0, 0.0]
        proba[label_id] = confidence
        proba[(label_id + 1) % 3] = (1 - confidence) * 0.6
        proba[(label_id + 2) % 3] = (1 - confidence) * 0.4
        return {
            "label": LABEL_MAP[label_id],
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "lr_proba": proba[:],
            "rf_proba": proba[:],
            "ensemble_proba": proba[:],
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump({"lr": self.lr, "rf": self.rf, "scaler": self.scaler,
                         "features": self._feature_names, "fitted": self._fitted}, f)

    def load(self, path: str):
        with open(os.path.join(path, "model.pkl"), "rb") as f:
            d = pickle.load(f)
        self.lr = d["lr"]; self.rf = d["rf"]
        self.scaler = d["scaler"]; self._feature_names = d["features"]
        self._fitted = d["fitted"]

    # ── Feature importance ─────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        if not self._fitted:
            return pd.DataFrame()
        imp = self.rf.feature_importances_
        return pd.DataFrame(
            {"feature": self._feature_names, "importance": imp}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
