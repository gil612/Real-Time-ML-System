from typing import Literal, Optional

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.anthropic import Anthropic

from .base import NewsSignal


class ClaudeNewsSignalExtractor:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: Optional[float] = 0,
        output_format: Literal['dict', 'NewsSignal'] = 'dict',
    ):
        self.llm = Anthropic(model=model_name, api_key=api_key, temperature=temperature)

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

        self.output_format = output_format

        self.model_name = model_name

    def get_signal(self, text: str) -> NewsSignal | dict:
        response: NewsSignal = self.llm.structured_predict(
            NewsSignal,
            prompt=self.prompt_template,
            news_article=text,
        )

        return response


if __name__ == '__main__':
    from .config import AnthropicConfig

    config = AnthropicConfig()

    llm = ClaudeNewsSignalExtractor(
        model_name=config.model_name,
        api_key=config.api_key,
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
        print(response.model_dump_json())

        """
        {
            "btc_signal": 1,
            "eth_signal": 0,
            "reasoning": "The news article "Bitcoin Is Going Up FOREVER!" is explicitly bullish for
            Bitcoin, suggesting strong upward price movement. However, it makes no mention of Ethereum and
            doesn\'t contain any information that would directly impact ETH prices. While there might be some
            indirect positive sentiment spillover to the broader crypto market, the direct impact on ETH is neutral based on this specific headline"
        },

        {
            'btc_singnal': -1, 'eth_signal': -1, 'reasoning': 'The news headline "Bitcoin Is Going Down FOREVER!" presents an extremely bearish sentiment directly targeting Bitcoin. Such dramatic negative headlines typically cause fear in the market and lead to selling pressure.            While Ethereum isn\'t directly mentioned, the extreme negative sentiment about Bitcoin usually has a spillover effect on the entire cryptocurrency market, including Ethereum, though potentially to a lesser degree.'
        }


        { 'btc_singnal': 1,'eth_signal': 0, 'reasoning': "Russia's new law specifically targets Bitcoin for sanctions evasion, which is likely to increase BTC demand and usage in international transactions. This could drive up BTC prices in the short term due to increased adoption and utility. However, the news doesn't directly impact Ethereum's ecosystem or usage, so ETH is expected to remain relatively neutral to this specific news."}

        {
            "btc_singnal":0,
            "eth_signal":0,
            "reasoning":"The news about Solana co-founder Stephen Akridge's personal legal
            issues with his ex-wife regarding crypto gains is unlikely to affect BTC or ETH prices.
            This is a localized incident specific to Solana's leadership and doesn't impact the fundamental or technical aspects of Bitcoin or Ethereum. While it might negatively impact Solana's reputation,
            it's too isolated to cause significant market movements in the broader cryptocurrency market,
            particularly for the two largest cryptocurrencies."
        }
        """
