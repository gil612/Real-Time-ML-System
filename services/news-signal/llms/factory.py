from typing import Literal

from loguru import logger

from .base import BaseNewsSignalExtractor
from .claude import ClaudeNewsSignalExtractor
from .config import AnthropicConfig, Config, OllamaConfig
from .ollama import OllamaNewsSignalExtractor


def get_llm(model_provider: Literal['anthropic', 'ollama']) -> BaseNewsSignalExtractor:
    """
    Get the LLM instance based on the model provider.
    """
    if model_provider == 'anthropic':
        return ClaudeNewsSignalExtractor()
    elif model_provider == 'ollama':
        return OllamaNewsSignalExtractor()
    else:
        raise ValueError(f'Unknown model provider: {model_provider}')
