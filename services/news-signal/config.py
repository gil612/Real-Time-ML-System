from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str

    model_provider: Literal["anthropic", "ollama"]

    # Ollama settings
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_timeout: int = 300
    ollama_model_name: str = "llama3.2:3b"

    # Anthropic settings
    anthropic_api_key: str = ""
    anthropic_model_name: str = "claude-3-sonnet-20240229"

    model_config = SettingsConfigDict(
        env_file="settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(
            "settings_",
        ),  # This addresses the warning about model_ namespace
    )


config = Config()

if __name__ == "__main__":
    print(config.model_dump_json(indent=2))
