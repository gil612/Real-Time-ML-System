import comet_ml
import joblib
from sklearn.metrics import mean_absolute_error
from feature_reader import FeatureReader
from loguru import logger
import pandas as pd
from models.dummy_model import DummyModel
from models.xgboost_model import XGBoostModel


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
    hyperparameters_tuning: bool,
    comet_api_key: str,
    comet_project_name: str,
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
    # https://www.comet.com/docs/v2/guides/quickstart/
    experiment = comet_ml.start(api_key=comet_api_key, project_name=comet_project_name)

    experiment.log_parameters(
        {
            # super important view name and version
            "feature_view_name": feature_view_name,
            "feature_view_version": feature_view_version,
            "pair_to_predict": pair_to_predict,
            "candle_seconds": candle_seconds,
            "technical_indicators_as_features": technical_indicators_as_features,
            "pairs_as_features": pairs_as_features,
            "prediction_seconds": prediction_seconds,
            "llm_model_name_news_signals": llm_model_name_news_signals,
            "days_back": days_back,
            "hyperparameters_tuning": hyperparameters_tuning,
        }
    )

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
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    experiment.log_parameters(
        {
            "X_train": X_train.shape,
            "y_train": y_train.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
        }
    )

    # 4. Build a dummy baseline based on current close price
    dummy_close = DummyModel(from_feature="close")

    # dummy model based on close price
    y_pred_close = dummy_close.predict(X_test)
    mae_dummy_model = mean_absolute_error(y_true=y_test, y_pred=y_pred_close)
    logger.info(f"MAE of dummy model based on close price: {mae_dummy_model}")
    experiment.log_metric("mae_dummy_model", mae_dummy_model)
    # Dummy model based on sma_7
    if "sma_7" in technical_indicators_as_features:
        dummy_sma7 = DummyModel(from_feature="sma_7")
        y_pred_sma7 = dummy_sma7.predict(X_test)
        mae_sma7 = mean_absolute_error(y_test, y_pred_sma7)
        logger.info(f"MAE of dummy model based on sma_7: {mae_sma7}")
        experiment.log_metric("mae_dummy_model_sma7", mae_sma7)

    # Dummy model based on sma_14
    if "sma_14" in technical_indicators_as_features:
        dummy_sma14 = DummyModel(from_feature="sma_14")
        y_pred_sma14 = dummy_sma14.predict(X_test)
        mae_sma14 = mean_absolute_error(y_test, y_pred_sma14)
        logger.info(f"MAE of dummy model based on sma_14: {mae_sma14}")
        experiment.log_metric("mae_dummy_model_sma14", mae_sma14)
    # Fit an ML modelon the training set
    model = XGBoostModel()
    model.fit(X_train, y_train, hyperparameters_tuning=hyperparameters_tuning)

    y_test_pred = model.predict(X_test)
    mae_xgboost = mean_absolute_error(y_test, y_test_pred)
    logger.info(f"MAE of XGBoost model: {mae_xgboost}")
    experiment.log_metric("mae", mae_xgboost)

    # https://www.comet.com/docs/v2/guides/model-registry/quickstart/

    # Save the model to local filepath
    model_filepath = "xgboost_model.joblib"
    joblib.dump(model.get_model_object(), model_filepath)

    # Log the model to Comet
    experiment.log_model(
        name="xgboost_model",
        file_or_folder=model_filepath,
    )


if __name__ == "__main__":
    from config import hopsworks_credentials, training_config, comet_credentials

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
        hyperparameters_tuning=training_config.hyperparameters_tuning,
        comet_api_key=comet_credentials.api_key,
        comet_project_name=comet_credentials.project_name,
    )
