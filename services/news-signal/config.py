from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration settings for the news-signal service.

    Attributes:
        kafka_broker_address: The address of the Kafka broker.
        kafka_input_topic: The topic to consume news from.
        kafka_output_topic: The topic to produce signals to.
        kafka_consumer_group: The consumer group ID.
        model_provider: The LLM provider to use ('anthropic' or 'ollama').
    """

    model_config = SettingsConfigDict(env_file='settings.env')

    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str

    model_provider: Literal['anthropic', 'ollama']


config = Config()
