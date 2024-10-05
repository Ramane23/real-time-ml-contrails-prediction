import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from loguru import logger
from features_engineering import FeaturesEngineering  


class BaselineModel:
    
    # Initialize the FeaturesEngineering class
    features_engineering = FeaturesEngineering()
    
    def __init__(self, df: pd.DataFrame, target_column: str, timestamp_split: str, features_engineering):
        """
        Initialize the BaselineModel class and split the dataset.

        Args:
            df (pd.DataFrame): The original dataframe with all necessary features.
            target_column (str): The target column for prediction.
            timestamp_split (str): Timestamp for splitting the training and testing data.
            features_engineering: An instance of the FeaturesEngineering class to apply feature transformations.
        """
        logger.info("Initializing the BaselineModel class...")

        # Step 1: Apply feature engineering and get transformed data
        logger.info("Applying feature engineering to get transformed features and target...")
        X_transformed, y_transformed = features_engineering.apply_feature_engineering(df)

        # Step 2: Concatenate `X_transformed` and `y_transformed` into a single dataframe
        df_transformed = pd.concat([X_transformed, y_transformed.rename(target_column)], axis=1)
        logger.debug(f"Combined dataframe shape after concatenation: {df_transformed.shape}")

        # Step 3: Convert the index to datetime and store it
        df_transformed.index = pd.to_datetime(df_transformed.index)

        # Step 4: Set up the target column and cutoff timestamp
        self.target_column = target_column
        self.cutoff_timestamp = pd.Timestamp(timestamp_split)

        # Step 5: Split the concatenated dataframe into train and test sets based on the timestamp
        logger.info("Splitting the transformed data into train and test sets...")
        self.train_data = df_transformed.loc[df_transformed.index < self.cutoff_timestamp].copy()
        self.test_data = df_transformed.loc[df_transformed.index >= self.cutoff_timestamp].copy()

        # Ensure monotonicity in the index
        if not self.train_data.index.is_monotonic_increasing:
            self.train_data = self.train_data.sort_index()
        if not self.test_data.index.is_monotonic_increasing:
            self.test_data = self.test_data.sort_index()

    def weighted_random_classifier(self):
        """Use the training data to build a weighted random classifier."""
        logger.info("Making predictions based on the weighted random classifier...")
        positive_class_proba = self.train_data[self.target_column].mean()
        
        np.random.seed(42)  # Ensure reproducibility
        self.test_data['random_pred'] = np.random.choice(
            [0, 1], size=len(self.test_data), p=[1 - positive_class_proba, positive_class_proba]
        )

    def evaluate_naive_models(self):
        """Evaluate the weighted random classifier and display its performance metrics."""
        logger.info("Evaluating the performance of the weighted random classifier...")
        evaluation_results = {}

        # Calculate metrics for Weighted Random Classifier
        random_metrics, cm = self.calculate_metrics(self.test_data[self.target_column], self.test_data['random_pred'])
        if pd.isna(random_metrics['Accuracy']):
            logger.warning("No positive class samples found in the test data. Metrics will be set to zero.")
            random_metrics = {metric: 0.0 for metric in random_metrics}
            cm = [[0, 0], [0, 0]]  # Default confusion matrix for empty class scenario

        evaluation_results['Weighted Random Classifier'] = random_metrics
        evaluation_results['Confusion Matrix'] = cm

        # Display the results
        print("\n=== Naive Model Evaluation Results ===")
        for model_name, metrics in evaluation_results.items():
            if model_name == 'Confusion Matrix':
                print(f"{model_name}: \n{metrics}")
            else:
                print(f"\n{model_name}:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")

        return evaluation_results

    @staticmethod
    def calculate_metrics(true_values, predicted_values):
        """Helper function to calculate classification metrics."""
        try:
            metrics = {
                'Accuracy': accuracy_score(true_values, predicted_values),
                'Precision': precision_score(true_values, predicted_values, zero_division=0),
                'Recall': recall_score(true_values, predicted_values, zero_division=0),
                'F1-Score': f1_score(true_values, predicted_values, zero_division=0)
            }
            cm = confusion_matrix(true_values, predicted_values).tolist()  # Ensure the confusion matrix is a list
        except ValueError as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0}
            cm = [[0, 0], [0, 0]]  # Default confusion matrix for error scenarios
        return metrics, cm


# Example usage:
if __name__ == "__main__":
    from features_engineering import FeaturesEngineering  # Assuming this is defined elsewhere

    # Load the dataset and initialize the BaselineModel class
    df = pd.read_csv("./files/flights_data_with_features.csv", index_col=0)
    
    # Instantiate the feature engineering class
    features_engineering = FeaturesEngineering()
    
    # Initialize the baseline model with the complete original dataframe
    baseline_model = BaselineModel(df, target_column='contrail_formation', timestamp_split='2024-09-14 12:00:00', features_engineering=features_engineering)

    # Use the weighted random classifier to make predictions
    baseline_model.weighted_random_classifier()

    # Evaluate the model
    baseline_model.evaluate_naive_models()
