import json
from pathlib import Path
from typing import Optional

import pandas as pd
import xgboost as xgb
from nfstream import NFPlugin


class XGBoostPredictions(NFPlugin):
    """NFStream plugin for making predictions using XGBoost model."""

    def __init__(self, model_path: Optional[Path] = None, **kwargs):
        """Initialize the XGBoost predictions plugin.

        Args:
            model_path: Path to the model directory containing model.json and metadata.json.
                       If None, defaults to 'models/development'.
            **kwargs: Additional keyword arguments passed to NFPlugin.
        """
        super().__init__(**kwargs)

        if model_path is None:
            model_path = Path("models/development")
            # Find the latest version if multiple versions exist
            versions = list(model_path.glob("**/model.json"))
            if not versions:
                raise FileNotFoundError("No model.json found in models/development")
            model_path = versions[-1].parent.parent

        self.model_dir = Path(model_path)
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.feature_names = self.metadata.get("feature_names", [])

        if not self.feature_names:
            raise ValueError("No feature names found in metadata")

    def _load_model(self) -> xgb.XGBClassifier:
        """Load the XGBoost model from model.json."""
        model_path = self.model_dir / "model" / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    def _load_metadata(self) -> dict:
        """Load model metadata from metadata.json."""
        metadata_path = self.model_dir / "metrics" / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def on_init(self, packet, flow):
        """Initialize flow enrichments and predictions."""
        flow.udps.enrichments = (
            flow.udps.enrichments if hasattr(flow.udps, "enrichments") else {}
        )
        flow.udps.enrichments["xgboost"] = {
            "prediction": None,
            "probability": None,
            "error": None,
        }
        self.on_update(packet, flow)

    def on_expire(self, flow):
        """Make predictions when flow expires."""
        try:
            # Create DataFrame from flow features
            df = pd.DataFrame(flow.values(), index=flow.keys()).transpose().astype({feature: "float64" for feature in self.feature_names})

            # Keep only features used by the model
            available_features = set(df.columns) & set(self.feature_names)
            missing_features = set(self.feature_names) - available_features

            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            X = df[self.feature_names]

            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0].max()

            flow.udps.enrichments["xgboost"].update(
                {
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "error": None,
                }
            )
            if prediction == 1 and probability > 0.8:
                flow.udps.detections += 1

        except Exception as e:
            flow.udps.enrichments["xgboost"].update(
                {"prediction": None, "probability": None, "error": str(e)}
            )
