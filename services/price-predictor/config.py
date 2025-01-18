from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file="training.settings.env")

    feature_view_name: str = Field(alias="FEATURE_VIEW_NAME")
    feature_view_version: int = Field(alias="FEATURE_VIEW_VERSION")
    pair_to_predict: str = Field(alias="PAIR_TO_PREDICT")
    candle_seconds: int = Field(alias="CANDLE_SECONDS")
    prediction_seconds: int = Field(alias="PREDICTION_SECONDS")
    pairs_as_features: list[str] = Field(alias="PAIRS_AS_FEATURES")
    days_back: int = Field(alias="DAYS_BACK")
    llm_model_name_news_signals: str = Field(alias="LLM_MODEL_NAME_NEWS_SIGNALS")


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
