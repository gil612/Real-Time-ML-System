from typing import Literal

import httpx

from .base import BaseNewsSignalExtractor, NewsSignal


class OllamaNewsSignalExtractor(BaseNewsSignalExtractor):
    """News signal extractor using Ollama"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = 'http://localhost:11434/api/generate'

    def get_signal(
        self, text: str, output_format: Literal['dict', 'NewsSignal'] = 'NewsSignal'
    ) -> NewsSignal | dict:
        """Extract trading signal from news text"""

        prompt = f"""
        You are an expert crypto financial analyst. Analyze this news and determine its impact on crypto prices:
        {text}
        Respond in JSON format with coin symbols and signals (1 for bullish, -1 for bearish).
        Only include coins that are directly impacted.
        """

        response = httpx.post(
            self.base_url,
            json={'model': self.model_name, 'prompt': prompt, 'stream': False},
        )

        result = NewsSignal.model_validate_json(response.json()['response'])

        if output_format == 'dict':
            return result.to_dict()
        return result
