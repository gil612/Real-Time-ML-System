from typing import Any, Dict

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from loguru import logger

from .base import BaseNewsSignalExtractor, NewsSignal
from .config import OllamaConfig


class OllamaNewsSignalExtractor(BaseNewsSignalExtractor):
    def __init__(self):
        config = OllamaConfig()
        logger.info(f'Initializing Ollama with config: {config}')

        self.model_name = config.model_name
        self.llm = Ollama(
            model=config.model_name,
            base_url=config.api_base,
            temperature=0.1,
            request_timeout=30.0,
            additional_kwargs={
                'verify': False,
                'timeout': 30.0,
                'mirostat': 0,
                'num_ctx': 512,
                'num_thread': 4,
            },
        )
        logger.info(f'Created Ollama LLM with base_url: {config.api_base}')

        self.prompt_template = PromptTemplate(
            template=(
                'System: You are a financial analyst. Analyze the sentiment and topic of news titles '
                'and predict their impact on cryptocurrency prices. You must respond with valid JSON '
                'that matches this exact format, including all quotes and commas:\n\n'
                '{"btc_signal": NUMBER, "eth_signal": NUMBER, "reasoning": "YOUR ANALYSIS"}\n\n'
                'Where NUMBER must be -1 (bearish), 0 (neutral), or 1 (bullish).\n\n'
                'User: {text}\n\n'
                'Assistant: '
            )
        )

    def get_signal(self, title: str) -> NewsSignal:
        logger.info(f'Getting signal for title: {title}')
        try:
            response: NewsSignal = self.llm.structured_predict(
                NewsSignal, self.prompt_template, text=title
            )
            logger.info(f'Got response: {response}')
            return response
        except Exception as e:
            logger.error(f'Error getting signal: {e}')
            # Return neutral signals on error
            return NewsSignal(
                btc_signal=0,
                eth_signal=0,
                reasoning=f'Error processing: {str(e)[:100]}',  # Truncate long error messages
            )


if __name__ == '__main__':
    llm = OllamaNewsSignalExtractor()
    examples = [
        'Bitcoin Is Going Up FOREVER!',
        'Bitcoin Is Going Down FOREVER!',
        'Russia Weaponizing Bitcoin? New Law Allows Crypto to Bypass Western Sanctions',
    ]
    for example in examples:
        print(f'Example: {example}')
        response = llm.get_signal(example)
        print(response)


"""
Example: Bitcoin Is Going Up FOREVER!
{'btc_signal': 1, 'eth_signal': 0, 'reasoning': 'The statement suggests an extremely bullish sentiment towards Bitcoin, which could lead to increased investor confidence and a surge in price.'}
Example: Bitcoin Is Going Down FOREVER!
{'btc_signal': -1, 'eth_signal': 0, 'reasoning': "The statement 'Bitcoin Is Going Down Forever!' suggests a strong bearish sentiment, indicating that the price of BTC is expected to decline in the long term."}
Example: Russia Weaponizing Bitcoin? New Law Allows Crypto to Bypass Western Sanctions
{'btc_signal': 1, 'eth_signal': 0, 'reasoning': 'The new law in Russia could lead to an increase in BTC price as investors seek safe-haven assets. ETH might not be affected directly by this news.'}
Example: Solana co-founder Stephen Akridge accused of misappropriating ex-wife's crypto gains
{'btc_signal': -1, 'eth_signal': 0, 'reasoning': "The news about Stephen Akridge's alleged misappropriation of his ex-wife's crypto gains may lead to a decrease in investor confidence in the cryptocurrency market. This could result in a decline in BTC and ETH prices as investors become more cautious."}

"""
