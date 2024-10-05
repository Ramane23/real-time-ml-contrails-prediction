from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger
import pandas as pd
from typing import Tuple, Optional
import numpy as np

# Import the existing FeaturesEngineering class
from features_engineering import FeaturesEngineering

# Custom Transformer for Encoding Categorical Features
class CategoricalFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer to apply `LabelEncoder` on categorical columns.
    """
    def __init__(self, columns: list):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.label_encoders.items():
            X[col] = le.transform(X[col])
        return X

# Main Model Class for XGBoost
class XGBoostPipelineModel:
    def __init__(self, target_column: str = 'contrail_formation', tune_hyper_params: Optional[bool] = False):
        self.target_column = target_column
        self.tune_hyper_params = tune_hyper_params
        self.features_engineering = FeaturesEngineering()

    def fit_xgboost_pipeline(self, df: pd.DataFrame) -> XGBClassifier:
        """
        Prepare the data using feature engineering and fit the XGBoost classifier.
        """
        logger.info("Applying feature engineering transformations...")

        # Apply feature engineering to get transformed X and y
        X_transformed, y_transformed = self.features_engineering.apply_feature_engineering(df)

        # Encode categorical features
        categorical_columns = ['route', 'flight_phase']
        categorical_encoder = CategoricalFeatureEncoder(columns=categorical_columns)
        X_transformed = categorical_encoder.fit_transform(X_transformed)

        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)

        # Check for consistency in number of rows
        if X_scaled.shape[0] != y_transformed.shape[0]:
            logger.error(f"Row mismatch after transformation: X_transformed has {X_scaled.shape[0]} rows, but y has {y_transformed.shape[0]} rows.")
            raise ValueError(f"Row mismatch after transformation: X_transformed ({X_scaled.shape[0]} rows) != y ({y_transformed.shape[0]} rows)")

        logger.info(f"Fitting XGBoost classifier on {X_scaled.shape[0]} samples...")

        # Fit the XGBoost classifier
        xgb_classifier = XGBClassifier()
        #breakpoint()
        xgb_classifier.fit(X_scaled, y_transformed)

        logger.info("XGBoost classifier fitting complete.")
        return xgb_classifier

# Main Model Class for LightGBM
class LightGBMPipelineModel:
    def __init__(self, target_column: str = 'contrail_formation', tune_hyper_params: Optional[bool] = False):
        self.target_column = target_column
        self.tune_hyper_params = tune_hyper_params
        self.features_engineering = FeaturesEngineering()

    def fit_lightgbm_pipeline(self, df: pd.DataFrame) -> LGBMClassifier:
        """
        Prepare the data using feature engineering and fit the LightGBM classifier.
        """
        logger.info("Applying feature engineering transformations...")

        # Apply feature engineering to get transformed X and y
        X_transformed, y_transformed = self.features_engineering.apply_feature_engineering(df)

        # Encode categorical features
        categorical_columns = ['route', 'flight_phase']
        categorical_encoder = CategoricalFeatureEncoder(columns=categorical_columns)
        X_transformed = categorical_encoder.fit_transform(X_transformed)

        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)

        # Check for consistency in number of rows
        if X_scaled.shape[0] != y_transformed.shape[0]:
            logger.error(f"Row mismatch after transformation: X_transformed has {X_scaled.shape[0]} rows, but y has {y_transformed.shape[0]} rows.")
            raise ValueError(f"Row mismatch after transformation: X_transformed ({X_scaled.shape[0]} rows) != y ({y_transformed.shape[0]} rows)")

        logger.info(f"Fitting LightGBM classifier on {X_scaled.shape[0]} samples...")

        # Fit the LightGBM classifier
        lgbm_classifier = LGBMClassifier()
        lgbm_classifier.fit(X_scaled, y_transformed)

        logger.info("LightGBM classifier fitting complete.")
        return lgbm_classifier
