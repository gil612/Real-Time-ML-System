from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file="training.settings.env")

    feature_view_name: str = Field(description="The name of the feature view")
    feature_view_version: int = Field(description="The version of the feature view")
    pair_to_predict: str = Field(description="The pair to train the model on")
    candle_seconds: int = Field(description="The number of seconds per candle")
    prediction_seconds: int = Field(
        description="The number of seconds into the future to predict"
    )
    pairs_as_features: list[str] = Field(
        description="The pairs to use for the features"
    )
    technical_indicators_as_features: list[str] = Field(
        description="The technical indicators to use for from the technical_indicators feature group"
    )
    llm_model_name_news_signals: str = Field(
        description="The name of the LLM model to use for the news signals"
    )
    days_back: int = Field(
        description="The number of days to consider for the historical data"
    )
    hyperparameters_tuning: bool = Field(
        description="Whether to tune the hyperparameters of the model"
    )


training_config = TrainingConfig()


class HopsworksCredentials(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="hopsworks_credentials.env",
        extra="allow",
        env_prefix="",  # Don't use any prefix for env vars
    )

    # Keep field names matching how they're used in the code
    api_key: str = Field(alias="hopsworks_api_key")
    project_name: str = Field(alias="hopsworks_project_name")


hopsworks_credentials = HopsworksCredentials()


class CometCredentials(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="comet_credentials.env",
        extra="allow",
        env_prefix="",  # Don't use any prefix for env vars
    )

    # Keep field names matching how they're used in the code
    api_key: str = Field(alias="comet_api_key")
    project_name: str = Field(alias="comet_project_name")


comet_credentials = CometCredentials()
