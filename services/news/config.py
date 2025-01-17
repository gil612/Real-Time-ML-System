from typing import Literal, Optional
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Configuration for the news service.
    """

    kafka_broker_address: str
    kafka_topic: str
    data_source: Literal["live", "historical"]
    historical_data_source_url_rar_file: Optional[str] = None
    historical_days_back: Optional[int] = 180
    polling_interval_sec: Optional[int] = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug(f"Loaded config: {self.model_dump_json(indent=2)}")


config = Config()


class CryptopanicConfig(BaseSettings):
    """
    Configuration for the Cryptopanic API.
    """

    model_config = SettingsConfigDict(env_file="cryptopanic_credentials.env")
    api_key: str


cryptopanic_config = CryptopanicConfig()
