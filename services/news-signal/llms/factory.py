from typing import Literal

from .base import BaseNewsSignalExtractor
from .claude import ClaudeNewsSignalExtractor


def get_llm(model_provider: Literal['anthropic', 'ollama']) -> BaseNewsSignalExtractor:
    """
    Return the LLM we want for the news signal extractor

    Args: model_provider: Literal["anthropic", "ollama"]

    Returns:
        The LLM we want for the news signal extractor
    """
    if model_provider == 'anthropic':
        from .config import AnthropicConfig

        config = AnthropicConfig()
        return ClaudeNewsSignalExtractor(
            model_name=config.model_name,
            api_key=config.api_key,
        )
    # elif model_provider == "ollama":
    #     return OllamaNewsSignalExtractor()
    else:
        raise ValueError(f'Unsupported model provider: {model_provider}')
