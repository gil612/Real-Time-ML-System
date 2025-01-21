from datetime import datetime, timedelta
from typing import Optional

import hopsworks
import pandas as pd
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
from loguru import logger


class FeatureReader:
    """
    Reads features from our 2 features groups
    - technical_indicators
    - news_signals
    and preprocess it so that it has the format (features, target) we need for
    training and for inference.
    """

    def __init__(
        self,
        hopsworks_project_name: str,
        hopsworks_api_key: str,
        feature_view_name: str,
        feature_view_version: int,
        pair_to_predict: str,
        candle_seconds: int,
        pairs_as_features: list[str],
        technical_indicators_as_features: list[str],
        prediction_seconds: int,
        llm_model_name_news_signals: str,
        # Optional. Only required if the feature view above does not exist and needs
        # to be created
        technical_indicators_feature_group_name: Optional[str] = None,
        technical_indicators_feature_group_version: Optional[int] = None,
        news_signals_feature_group_name: Optional[str] = None,
        news_signals_feature_group_version: Optional[int] = None,
    ):
        """ """

        self.pair_to_predict = pair_to_predict
        self.candle_seconds = candle_seconds
        self.pairs_as_features = pairs_as_features
        self.technical_indicators_as_features = technical_indicators_as_features
        self.prediction_seconds = prediction_seconds
        self.llm_model_name_news_signals = llm_model_name_news_signals

        # connect to the Hopsworks Feature Store
        self._feature_store = self._get_feature_store(
            hopsworks_project_name,
            hopsworks_api_key,
        )

        if technical_indicators_feature_group_name is not None:
            logger.info(
                f"Attempt to create the feature view {feature_view_name}-{feature_view_version}"
            )
            self._feature_view = self._create_feature_view(
                feature_view_name,
                feature_view_version,
                technical_indicators_feature_group_name,
                technical_indicators_feature_group_version,
                news_signals_feature_group_name,
                news_signals_feature_group_version,
            )
        else:
            self._feature_view = self._get_feature_view(
                feature_view_name,
                feature_view_version,
            )

    def _get_feature_group(self, name: str, version: int) -> FeatureGroup:
        """
        Returns a feature group object given its name and version.
        """
        return self._feature_store.get_feature_group(
            name=name,
            version=version,
        )

    def _create_feature_view(
        self,
        feature_view_name: str,
        feature_view_version: int,
        technical_indicators_feature_group_name: str,
        technical_indicators_feature_group_version: int,
        news_signals_feature_group_name: str,
        news_signals_feature_group_version: int,
    ) -> FeatureView:
        """
        Creates a feature view by joining the technical_indicators and news_signals
        feature groups.

        Args:
            feature_view_name: The name of the feature view to create.
            feature_view_version: The version of the feature view to create.
            technical_indicators_feature_group_name: The name of the technical_indicators
                feature group.
            technical_indicators_feature_group_version: The version of the technical_indicators
                feature group.
            news_signals_feature_group_name: The name of the news_signals feature group.
            news_signals_feature_group_version: The version of the news_signals feature group.

        Returns:
            The feature view object.
        """

        # we get the 2 features groups we need to join
        technical_indicators_fg = self._get_feature_group(
            technical_indicators_feature_group_name,
            technical_indicators_feature_group_version,
        )
        news_signals_fg = self._get_feature_group(
            news_signals_feature_group_name,
            news_signals_feature_group_version,
        )

        # # Query in 3-steps
        # # Step 1. Filter rows from news_signals_fg for the model_name we need and drop the model_name column
        # news_signal_query = news_signals_fg \
        #     .select_all() \
        #     .filter(news_signals_fg.model_name == self.llm_model_name_news_signals)
        # # query = news_signal_query

        # # # Step 2. Filter rows from technical_indicators_fg for the candle_seconds we need
        # technical_indicators_query = technical_indicators_fg \
        #     .select_all() \
        #     .filter(technical_indicators_fg.candle_seconds == self.candle_seconds) \

        # # Step 3. Join the 2 queries on the `coin` column
        # query = technical_indicators_query.join(
        #     news_signal_query,
        #     on=["coin"],
        #     join_type="left",
        #     prefix='news_signals_',
        # )

        # Attempt to create the feature view in one query
        query = (
            technical_indicators_fg.select_all()
            .join(
                news_signals_fg.select_all(),
                on=["coin"],
                join_type="left",
                prefix="news_signals_",
            )
            .filter(
                (technical_indicators_fg.candle_seconds == self.candle_seconds)
                & (news_signals_fg.model_name == self.llm_model_name_news_signals)
            )
        )

        # attempt to create the feature view
        feature_view = self._feature_store.create_feature_view(
            name=feature_view_name,
            version=feature_view_version,
            query=query,
            # This seemingly innocent flag makes this crash
            # logging_enabled=True,
        )
        logger.info(f"Feature view {feature_view_name}-{feature_view_version} created")

        return feature_view

    def _get_feature_view(
        self, feature_view_name: str, feature_view_version: int
    ) -> FeatureView:
        """
        Returns a feature view object given its name and version.
        """
        # raise NotImplementedError('Feature view creation is not supported yet')
        return self._feature_store.get_feature_view(
            name=feature_view_name,
            version=feature_view_version,
        )

    def _get_feature_store(self, project_name: str, api_key: str) -> FeatureStore:
        """
        Returns a feature store object.
        """
        logger.info("Getting feature store")
        project = hopsworks.login(project=project_name, api_key_value=api_key)
        fs = project.get_feature_store()
        return fs

    def get_training_data(self, days_back: int):
        """
        Use the self._feature_view to get the training data going back `days_back` days.
        """
        # get raw features from the Feature Store
        logger.info(f"Getting training data going back {days_back} days")
        raw_features = self._feature_view.get_batch_data(
            start_time=datetime.now() - timedelta(days=days_back),
            end_time=datetime.now(),
        )

        # horizontally stack the features for each pair
        # we want the outpu to be a daframe with (features, target)
        features = self._preprocess_raw_features_into_features_and_target(
            raw_features,
            add_target_column=True,
        )

        return features

    def _preprocess_raw_features_into_features_and_target(
        self,
        data: pd.DataFrame,
        add_target_column: bool,
    ) -> pd.DataFrame:
        """
        Preprocess the features into features and possibly targets.
        Horizontally stack the features for each pair, matching the timestamps.
        """
        if self.pair_to_predict != self.pairs_as_features[0]:
            raise ValueError(
                f"Pair {self.pair_to_predict} not found as the first feature in pairs_as_features"
            )

        df_all = None
        for pair in self.pairs_as_features:
            logger.info(f"Horizontally stacking features for pair {pair}")

            # Filter rows for this pair
            df = data[data["pair"] == pair]

            # Define base columns we want
            base_columns = ["pair", "window_end_ms", "open", "close"]

            # Add technical indicators if they exist
            available_indicators = [
                col
                for col in self.technical_indicators_as_features
                if col in df.columns
            ]

            # Add news signals if they exist
            news_columns = (
                ["news_signals_signal"] if "news_signals_signal" in df.columns else []
            )

            # Combine all available columns
            columns_to_keep = base_columns + available_indicators + news_columns

            # Keep only the columns we need
            df = df[columns_to_keep]

            if df_all is not None:
                df_all = df_all.merge(
                    df,
                    on="window_end_ms",
                    how="left",
                    suffixes=("", f"_{pair}"),
                )
            else:
                df_all = df

        # Add target column after all pairs are merged
        if add_target_column:
            logger.info("Adding target column to the dataset")
            # Get close price for BTC/USD (first pair) for target
            df_target = df_all[
                ["window_end_ms", "close"]
            ].copy()  # Use copy to avoid SettingWithCopyWarning
            df_target["window_end_ms"] = (
                df_target["window_end_ms"] - self.prediction_seconds * 1000
            )
            df_all = df_all.merge(
                df_target,
                on="window_end_ms",
                how="left",
                suffixes=("", "_target"),
            )
            df_all = df_all[df_all["close_target"].notna()]
            df_all.rename(columns={"close_target": "target"}, inplace=True)

        # rename the window_end_ms column to timestamp_ms and sort by it
        df_all.rename(columns={"window_end_ms": "timestamp_ms"}, inplace=True)
        df_all.sort_values(by="timestamp_ms", inplace=True)

        return df_all

    def get_inference_data(self):
        pass


if __name__ == "__main__":
    from config import hopsworks_credentials

    feature_reader = FeatureReader(
        hopsworks_project_name=hopsworks_credentials.project_name,
        hopsworks_api_key=hopsworks_credentials.api_key,
        feature_view_name="price_predictor",
        feature_view_version=4,
        pair_to_predict="BTC/USD",
        candle_seconds=60,
        pairs_as_features=["BTC/USD", "ETH/USD", "XRP/USD"],
        technical_indicators_as_features=[
            "rsi_9",
            "rsi_14",
            "rsi_21",
            "macd",
            "macd_signal",
            "macd_hist",
            "bbands_upper",
            "bbands_middle",
            "bbands_lower",
            "stochrsi_fastk",
            "stochrsi_fastd",
            "adx",
            "volume_ema",
            "ichimoku_conv",
            "ichimoku_base",
            "ichimoku_span_a",
            "ichimoku_span_b",
            "mfi",
            "atr",
            "price_roc",
            "sma_7",
            "sma_14",
            "sma_21",
        ],
        prediction_seconds=60 * 5,
        llm_model_name_news_signals="dummy",
    )

    training_data = feature_reader.get_training_data(days_back=90)
    print(training_data)
