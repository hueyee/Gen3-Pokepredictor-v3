"""Custom one-hot encoder for Pokemon feature encoding."""
import numpy as np


class CustomOneHotEncoder:
    """A custom one-hot encoder that handles unknown values gracefully."""

    def __init__(self):
        self.encoders = {}
        self.feature_names = []
        self.n_features = 0

    def fit(self, X, categorical_features):
        """Fit the encoder on the training data.

        Args:
            X: DataFrame with features
            categorical_features: List of categorical feature column names

        Returns:
            self
        """
        self.encoders = {}
        self.feature_names = []
        self.n_features = 0
        for feature in categorical_features:
            unique_values = X[feature].unique()
            self.encoders[feature] = {value: i for i, value in enumerate(unique_values)}
            for value in unique_values:
                self.feature_names.append(f"{feature}_{value}")
            self.n_features += len(unique_values)
        return self

    def transform(self, X, categorical_features):
        """Transform data using the fitted encoder.

        Args:
            X: DataFrame with features
            categorical_features: List of categorical feature column names

        Returns:
            Encoded numpy array
        """
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features))
        current_idx = 0
        for feature in categorical_features:
            if feature not in self.encoders:
                current_idx += 0
                continue
            encoder = self.encoders[feature]
            for i, value in enumerate(X[feature]):
                if value in encoder:
                    encoded[i, current_idx + encoder[value]] = 1
            current_idx += len(encoder)
        return encoded

    def get_feature_names(self):
        """Get the names of all encoded features.

        Returns:
            List of feature names
        """
        return self.feature_names
