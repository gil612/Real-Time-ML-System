import time
from typing import Literal, Optional

import httpx
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from loguru import logger

from .base import BaseLLM, BaseNewsSignalExtractor, NewsSignal


class OllamaNewsSignalExtractor(BaseNewsSignalExtractor):
    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = 0,
    ):
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
        )

        self.prompt_template = PromptTemplate(
            template="""
            You are a financial analyst.
            You are given a news article and you need to determine the impact of the news on the BTC and ETH price.

            You need to output the signal in the following format:
            {
                "btc_signal": 1,
                "eth_signal": 0
            }

            The signal is either 1, 0, or -1.
            1 means the price is expected to go up.
            0 means the price is expected to stay the same.
            -1 means the price is expected to go down.

            Here is the news article:
            {news_article}
            """
        )

        self.model_name = model_name

    def get_signal(
        self,
        text: str,
        output_format: Literal['dict', 'NewsSignal'] = 'dict',
    ) -> dict | NewsSignal:
        """
        Get the news signal from the given `text`

        Args:
            text: The news article to get the signal from
            output_format: The format of the output

        Returns:
            The news signal
        """
        response: NewsSignal = self.llm.structured_predict(
            NewsSignal,
            prompt=self.prompt_template,
            news_article=text,
        )

        if output_format == 'dict':
            return response.to_dict()
        else:
            return response


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 11434,
        model: str = 'llama2',
        timeout: int = 300,
    ):
        self.base_url = f'http://{host}:{port}'
        self.model = model
        self.timeout = timeout
        self._client = None
        self._init_client()

    def _init_client(self):
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout=self.timeout),
            headers={'Content-Type': 'application/json'},
        )

    def _wait_for_server(self, max_retries: int = 5, delay: int = 5) -> bool:
        for i in range(max_retries):
            try:
                response = self._client.get('/api/version')
                if response.status_code == 200:
                    logger.info(
                        f'Successfully connected to Ollama server: {response.json()}'
                    )
                    return True
            except Exception as e:
                logger.warning(
                    f'Attempt {i+1}/{max_retries} to connect to Ollama failed: {e}'
                )
                if i < max_retries - 1:
                    time.sleep(delay)
        return False

    async def generate(self, prompt: str) -> str:
        if not self._wait_for_server():
            raise ConnectionError('Could not connect to Ollama server')

        try:
            response = self._client.post(
                '/api/generate',
                json={'model': self.model, 'prompt': prompt, 'stream': False},
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f'Error generating response from Ollama: {e}')
            raise


if __name__ == '__main__':
    from .config import OllamaConfig

    config = OllamaConfig()

    llm = OllamaNewsSignalExtractor(
        model_name=config.model_name,
    )

    examples = [
        'Bitcoin ETF ads spotted on China’s Alipay payment app',
        'U.S. Supreme Court Lets Nvidia’s Crypto Lawsuit Move Forward',
        'Trump’s World Liberty Acquires ETH, LINK, and AAVE in $12M Crypto Shopping Spree',
    ]

    for example in examples:
        print(f'Example: {example}')
        response = llm.get_signal(example)
        print(response)

    """
    Example: Bitcoin ETF ads spotted on China’s Alipay payment app
    {
        "btc_signal": 1,
        "eth_signal": 0,
        'reasoning': "The news of Bitcoin ETF ads being spotted on China's Alipay payment
        app suggests a growing interest in Bitcoin and other cryptocurrencies among Chinese
        investors. This could lead to increased demand for BTC, causing its price to rise."
    }

    Example: U.S. Supreme Court Lets Nvidia’s Crypto Lawsuit Move Forward
    {
        'btc_signal': -1,
        'eth_signal': -1,
        'reasoning': "The US Supreme Court's decision allows Nvidia to pursue its crypto
        lawsuit, which could lead to increased regulatory uncertainty and potential
        restrictions on cryptocurrency mining. This could negatively impact the prices
        of both BTC and ETH."
    }

    Example: Trump’s World Liberty Acquires ETH, LINK, and AAVE in $12M Crypto Shopping Spree
    {
        'btc_signal': 0,
        'eth_signal': 1,
        'reasoning': "The acquisition of ETH by a major company like
        Trump's World Liberty suggests that there is increased demand for
        Ethereum, which could lead to an increase in its price."
    }
    """
