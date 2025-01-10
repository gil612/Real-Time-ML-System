from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


class NewsSignalOneCoin(BaseModel):
    coin: Literal[
        'BTC',
        'ETH',
        'SOL',
        'XRP',
        'DOGE',
        'ADA',
        'XLM',
        'LTC',
        'BCH',
        'DOT',
        'XMR',
        'EOS',
        'XEM',
        'ZEC',
        'ETC',
    ] = Field(description='The coin that the news is about')
    signal: Literal[1, 0, -1] = Field(
        description="""
    The signal of the news on the coin price.
    1 if the price is expected to go up
    0 if the price is expected to stay the same
    -1 if it is expected to go down.
    """
    )


class NewsSignal(BaseModel):
    news_signals: list[NewsSignalOneCoin]

    def to_dict(self) -> dict:
        """Convert NewsSignal to a dictionary format"""
        result = {}
        for signal in self.news_signals:
            result[f'{signal.coin.lower()}_signal'] = signal.signal
        return result


class BaseNewsSignalExtractor(ABC):
    def __init__(self, model_name: str):
        self._model_name = model_name

    @abstractmethod
    def get_signal(
        self, text: str, output_format: Literal['dict', 'NewsSignal'] = 'dict'
    ) -> dict | NewsSignal:
        pass

    @property
    def model_name(self) -> str:
        return self._model_name
