import os
import pandas as pd
from loguru import logger
from typing import Tuple, Optional
from comet_ml import Experiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from model_factory import XGBoostPipelineModel, LightGBMPipelineModel  # Import the pipeline model classes

from data_preprocessing import DataPreprocessing
from features_engineering import FeaturesEngineering
from baseline_model import BaselineModel
from model_factory import CategoricalFeatureEncoder
from config import config

#display the content of the config file
logger.debug('training service is starting...')
logger.debug(f'Config: {config.model_dump()}')

# Instantiate the DataPreprocessing class
data_preprocessing = DataPreprocessing()
# Instantiate the FeaturesEngineering class
features_engineering = FeaturesEngineering()

def train(
    target_column: str,
    timestamp_split: str,
    comet_ml_api_key: str,
    comet_ml_project_name: str,
    comet_ml_workspace: str,
):
    """
    This function trains the contrail prediction model using XGBoost and LightGBM,
    logs the results to CometML, and evaluates model performance.

    Args:
        input_data_path (str): Path to the input flight data.
        target_column (str): The name of the target column.
        timestamp_split (str): Timestamp for splitting the data into train and test sets.
        comet_ml_api_key (str): API key for CometML.
        comet_ml_project_name (str): Project name for CometML.
        comet_ml_workspace (str): Workspace for CometML.

    Returns:
        None.
    """
    # Create an experiment to log metadata to CometML
    experiment = Experiment(
        api_key=comet_ml_api_key,
        project_name=comet_ml_project_name,
        workspace=comet_ml_workspace,
    )

    # Step 1: Load and preprocess the dataset
    logger.info(f"Loading and preprocessing data from the hopsworks feature store...")
    #df = data_preprocessing.preprocess_data()
    df = pd.read_csv('./files/flights_data_preprocessed.csv').head(400000)
        
    #reduce the size of the dataset to avoi memory error*
    #df = df.head(300000)
    
    #set the index to the timestamp column
    df.set_index('current_flight_time', inplace=True)
    
    #convert the index to datetime
    df.index = pd.to_datetime(df.index)
    
    # log a dataset hash to track the data
    experiment.log_dataset_hash(df)
    #breakpoint()
    
    # Step 2: Split the data into train and test sets based on the timestamp
    logger.info(f"Splitting the data into train and test sets at {timestamp_split}...")
    train_data, test_data = split_train_test(df, timestamp_split)
    logger.debug(f"Train data shape: {train_data.shape} and Test data shape: {test_data.shape}")
    #breakpoint()

    # Log the number of rows in train and test datasets
    experiment.log_metric('n_rows_train', train_data.shape[0])
    experiment.log_metric('n_rows_test', test_data.shape[0])
    
    #step 3 : buil a baseline model
    logger.info("Building a baseline model...")
    baseline_model = BaselineModel(df, target_column, timestamp_split, features_engineering)
    
    #the first model is a time based frequency model
    #baseline_model.time_based_frequency_model()
    
    #the second model is a weighted random classifier
    baseline_model.weighted_random_classifier()
    
    #evaluate the two baseline models
    evaluation_results = baseline_model.evaluate_naive_models()
    
    #get the metrics and confusion matrix for the time based frequency model
    #time_based_baseline_metrics = evaluation_results['Time-Based Frequency Model']
    #time_based_baseline_confusion_matrix = evaluation_results['Confusion Matrix']
    #logger.info(f"Time based frequency model Baseline Model Metrics: {time_based_baseline_metrics}, {time_based_baseline_confusion_matrix}")
    #experiment.log_metrics("Time based frequency model Baseline Model Metrics",time_based_baseline_metrics, time_based_baseline_confusion_matrix)
    
    #get the metrics and confusion matrix for the weighted random classifier
    weighted_baseline_metrics = evaluation_results['Weighted Random Classifier']
    weighted_baseline_confusion_matrix = evaluation_results['Confusion Matrix']
    weighted_baseline_accuracy = weighted_baseline_metrics['Accuracy']
    logger.info(f"Weighted random classifier Baseline Model Metrics: {weighted_baseline_metrics}, {weighted_baseline_confusion_matrix}")
    experiment.log_metrics(weighted_baseline_metrics)
    experiment.log_confusion_matrix(matrix=weighted_baseline_confusion_matrix)
    
    #get the best baseline model based on the accuracy
    #baseline_test_accuracy = time_based_baseline_metrics['Accuracy'] if time_based_baseline_metrics['Accuracy'] > weighted_baseline_metrics['Accuracy'] else weighted_baseline_metrics['Accuracy']
    
    #Step 4: Train XGBoost classifier using the custom pipeline
    logger.info("Training XGBoost classifier using the custom pipeline...")
    xgb_pipeline_model = XGBoostPipelineModel(target_column=target_column, tune_hyper_params=False)
    xgb_pipeline = xgb_pipeline_model.fit_xgboost_pipeline(train_data)
    xgb_metrics, xgb_confusion_matrix = evaluate_model(xgb_pipeline, test_data, target_column, description="XGBoost Model on Test Data")
    experiment.log_metrics(xgb_metrics)
    experiment.log_confusion_matrix(matrix=xgb_confusion_matrix)

    # Step 5: Train LightGBM classifier using the custom pipeline
    logger.info("Training LightGBM classifier using the custom pipeline...")
    lgbm_pipeline_model = LightGBMPipelineModel(target_column=target_column, tune_hyper_params=False)
    lgbm_pipeline = lgbm_pipeline_model.fit_lightgbm_pipeline(train_data)
    lgb_metrics, lgb_confusion_matrix = evaluate_model(lgbm_pipeline, test_data, target_column, description="LightGBM Model on Test Data")
    experiment.log_metrics(lgb_metrics)
    experiment.log_confusion_matrix(matrix=lgb_confusion_matrix)

    # Step 6: Save the best model based on the accuracy
    best_model = xgb_pipeline if xgb_metrics['Accuracy'] > lgb_metrics['Accuracy'] else lgbm_pipeline
    save_model(best_model, './best_model.pkl')
    
    if best_model == xgb_pipeline:
        model_name = f"best_xgboost_{target_column}_model"
        #log the model 
        experiment.log_model(name=model_name, file_or_folder='./best_model.pkl')
        #push the model to the model registry if it performs better than the baseline model
        if xgb_metrics['Accuracy'] > weighted_baseline_accuracy:
            experiment.register_model(
                model_name=model_name,
            )
        logger.info(f"Best model is XGBoost with accuracy: {xgb_metrics['Accuracy']}")
        logger.info("model saved to the model registry")
    else:
        model_name = f"best_lightgbm_{target_column}_model"
        #log the model 
        experiment.log_model(name=model_name, file_or_folder='./best_model.pkl')
        #push the model to the model registry if it performs better than the baseline model
        if lgb_metrics['Accuracy'] > weighted_baseline_accuracy:
            experiment.register_model(
                model_name=model_name,
            )
        logger.info(f"Best model is LightGBM with accuracy: {lgb_metrics['Accuracy']}")
        logger.info("model saved to the model registry")


def split_train_test(
    df: pd.DataFrame,
    timestamp_split: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing based on a given timestamp.

    Args:
        df (pd.DataFrame): The input dataframe.
        timestamp_split (str): The timestamp to split the data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    # Convert the index to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.info("Converting index to datetime format...")
        df.index = pd.to_datetime(df.index)

    # Convert the timestamp_split to Timestamp object
    cutoff_timestamp = pd.Timestamp(timestamp_split)
    
    # Perform the split using the datetime index
    train_data = df[df.index < cutoff_timestamp]
    test_data = df[df.index >= cutoff_timestamp]
    #breakpoint()
    return train_data, test_data

def evaluate_model(
    model,
    test_data: pd.DataFrame,
    target_column: str,
    description: Optional[str] = 'Model Evaluation'
) -> dict:
    """
    Evaluates the model using accuracy, precision, recall, and F1-score.

    Args:
        model: The trained pipeline model.
        test_data (pd.DataFrame): Test dataframe containing features and target.
        target_column (str): The target column in the dataframe.
        description (str): Description of the evaluation.

    Returns:
        dict: Evaluation metrics.
    """
    logger.info(f'**** {description} ****')
    
    #first we need to apply the same feature engineering to the test data
    X_test, y_test  = features_engineering.apply_feature_engineering(test_data)
    
    # Encode categorical features
    categorical_columns = ['route', 'flight_phase']
    categorical_encoder = CategoricalFeatureEncoder(columns=categorical_columns)
    X_test = categorical_encoder.fit_transform(X_test)
    
    #Scale features for better performance
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and log metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    logger.info(f"Metrics: {metrics}")
    
    #get the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix: {cm}")
    
    return metrics, cm


def save_model(model, filename: str):
    """
    Save the model as a pickle file.

    Args:
        model: The model to save.
        filename (str): Path to save the model file.
    """
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    logger.info(f"Model saved to {filename}")


if __name__ == '__main__':
    # Run the training pipeline
    train(
        target_column='contrail_formation',
        timestamp_split='2024-09-14 12:00:00',
        comet_ml_api_key=config.comet_ml_api_key,
        comet_ml_project_name=config.comet_ml_project_name,
        comet_ml_workspace=config.comet_ml_workspace,
    )
