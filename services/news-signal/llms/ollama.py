from typing import Literal, Optional

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama

from .base import BaseNewsSignalExtractor, NewsSignal


class OllamaNewsSignalExtractor(BaseNewsSignalExtractor):
    def __init__(self, model_name: str = 'llama3.2', temperature: Optional[float] = 0):
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
        )

        self.prompt_template = PromptTemplate(
            template="""
            You are a financial analyst.
            You are given a news item and you need to determine the impact of the news on the BTC and ETH price.
            You need to output the signal in the following format:
            {
                "btc_signal": 1,
                "eth_signal": 0
            }

            The signal is either 1, 0 or -1.
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
    ) -> NewsSignal | dict:
        """
        Get the news signa from the given @text

        Args:
            text: The news article to get signal form
            output_format: The format of the output. Can be "dict" or "NewsSignal"

        Returns:
            The news signal
        """
        response: NewsSignal = self.llm.structured_predict(
            NewsSignal,
            prompt=self.prompt_template,
            news_article=text,
        )
        self.output_format = output_format

        if output_format == 'dict':
            return response.to_dict()

        return response


if __name__ == '__main__':
    from .config import OllamaConfig

    config = OllamaConfig()

    llm = OllamaNewsSignalExtractor(
        model_name=config.model_name,
    )
    examples = [
        'Bitcoin Is Going Up FOREVER!',
        'Bitcoin Is Going Down FOREVER!',
        # "The Ghosts of Bitcoins Past",  # -> ValueError: Expected at least one tool call, but got 0 tool calls.
        'Russia Weaponizing Bitcoin? New Law Allows Crypto to Bypass Western Sanctions',
        'Solana co-founder Stephen Akridge accused of misappropriating ex-wife’s crypto gains',
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
Example: Solana co-founder Stephen Akridge accused of misappropriating ex-wife’s crypto gains
{'btc_signal': -1, 'eth_signal': 0, 'reasoning': "The news about Stephen Akridge's alleged misappropriation of his ex-wife's crypto gains may lead to a decrease in investor confidence in the cryptocurrency market. This could result in a decline in BTC and ETH prices as investors become more cautious."}

"""
