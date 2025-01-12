from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Configuration for the news service.
    """

    model_config = SettingsConfigDict(env_file="settings.env")
    kafka_broker_address: str
    kafka_topic: str
    data_source: Literal["live", "historical"]

    polling_interval_sec: Optional[int] = 10
    historical_data_source_url: Optional[str] = None


config = Config()


class CryptopanicConfig(BaseSettings):
    """
    Configuration for the Cryptopanic API.
    """

    model_config = SettingsConfigDict(env_file="cryptopanic_credentials.env")
    api_key: str


cryptopanic_config = CryptopanicConfig()
