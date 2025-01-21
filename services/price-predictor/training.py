from sklearn.metrics import mean_absolute_error
from feature_reader import FeatureReader
from loguru import logger
import pandas as pd
from dummy import DummyModel


def train_test_split(
    data: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing sets.
    The data is sorted by timestamp_ms and then split into training and testing sets.
    The testing set is the last 20% of the data.
    """
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    return train_data, test_data


def train(
    hopsworks_project_name: str,
    hopsworks_api_key: str,
    feature_view_name: str,
    feature_view_version: int,
    pair_to_predict: str,
    candle_seconds: int,
    technical_indicators_as_features: list[str],
    pairs_as_features: list[str],
    prediction_seconds: int,
    llm_model_name_news_signals: str,
    days_back: int,
):
    """
    Does the following:
    1. Reads feature data from the Feature Store
    2. Splits the data into training and testing sets
    3. Trains a model on the training set
    4. Evaluates the model on the testing set
    5. Saves the model to the model registry

    Everything is instrumented with CometML.


    """
    logger.info("Hello from the ML model training job...")

    # 1. Read feature data from the Feature Store
    feature_reader = FeatureReader(
        hopsworks_project_name,
        hopsworks_api_key,
        feature_view_name,
        feature_view_version,
        pair_to_predict,
        candle_seconds,
        pairs_as_features,
        technical_indicators_as_features,
        prediction_seconds,
        llm_model_name_news_signals,
    )
    logger.info(f"Reading feature data for {days_back} days back...")
    features_and_targets = feature_reader.get_training_data(days_back=days_back)
    logger.info(f"Got {len(features_and_targets)} rows")

    # 2. Split the data into training and testing sets
    train_df, test_df = train_test_split(features_and_targets, test_size=0.2)

    # 3. Split into features and targets
    # X_train = train_df.drop(columns=["target"])
    # y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # 4. Build a dummy baseline based on current close price
    dummy_close = DummyModel(from_feature="close")

    y_pred_close = dummy_close.predict(X_test)
    mae_dummy_model = mean_absolute_error(y_true=y_test, y_pred=y_pred_close)
    logger.info(f"MAE of dummy model based on close price: {mae_dummy_model}")

    # Dummy model based on sma_7
    if "sma_7" in technical_indicators_as_features:
        dummy_sma7 = DummyModel(from_feature="sma_7")
        y_pred_sma7 = dummy_sma7.predict(X_test)
        mae_sma7 = mean_absolute_error(y_test, y_pred_sma7)
        logger.info(f"MAE of dummy model based on sma_7: {mae_sma7}")

    # Dummy model based on sma_14
    if "sma_14" in technical_indicators_as_features:
        dummy_sma14 = DummyModel(from_feature="sma_14")
        y_pred_sma14 = dummy_sma14.predict(X_test)
        mae_sma14 = mean_absolute_error(y_test, y_pred_sma14)
        logger.info(f"MAE of dummy model based on sma_14: {mae_sma14}")

    # 5. Save the model to the model registry


if __name__ == "__main__":
    from config import hopsworks_credentials, training_config

    train(
        hopsworks_project_name=hopsworks_credentials.project_name,
        hopsworks_api_key=hopsworks_credentials.api_key,
        feature_view_name=training_config.feature_view_name,
        feature_view_version=training_config.feature_view_version,
        pair_to_predict=training_config.pair_to_predict,
        candle_seconds=training_config.candle_seconds,
        technical_indicators_as_features=training_config.technical_indicators_as_features,
        pairs_as_features=training_config.pairs_as_features,
        prediction_seconds=training_config.prediction_seconds,
        llm_model_name_news_signals=training_config.llm_model_name_news_signals,
        days_back=training_config.days_back,
    )
