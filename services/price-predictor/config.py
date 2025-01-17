from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class TrainingConfig(BaseModel):
    model_config = SettingsConfigDict(env_file=".env")

    target_pair: str = Field(description="The pair to train the model on")
    candle_seconds: int = Field(description="The number of seconds per candle")
    prediction_seconds: int = Field(description="The number of seconds to predict")

    feature_pairs: list[str] = Field(description="The pairs to use for features")
    days_back: int = Field(description="The number of days to use for features")


training_config = TrainingConfig()
