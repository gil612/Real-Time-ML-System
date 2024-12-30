import os
from typing import Any, ClassVar, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Base configuration class"""

    model_provider: str = 'ollama'
    model_config = SettingsConfigDict(
        env_file='settings.env', protected_namespaces=('settings_',)
    )


class AnthropicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='anthropic_credentials.env')
    model_name: str = 'claude-3-5-sonnet-20240620'
    api_key: str
    model_config: ClassVar[Dict[str, Any]] = {'protected_namespaces': ('settings_',)}


class OllamaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='ollama.env', env_prefix='OLLAMA_')
    model_name: str
    api_base: str = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:11434"
    model_config: ClassVar[Dict[str, Any]] = {'protected_namespaces': ('settings_',)}
