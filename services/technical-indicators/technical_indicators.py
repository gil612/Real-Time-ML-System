import numpy as np
from quixstreams import State
from talib import stream


def compute_indicators(
    candle: dict,
    state: State,
) -> dict:
    indicators = {}

    candles = state.get('candles', [])
    # extract the close price from the candle
    # open = np.array([candle['open'] for candle in candles])
    close = np.array([candle['close'] for candle in candles])
    high = np.array([candle['high'] for candle in candles])
    low = np.array([candle['low'] for candle in candles])
    volume = np.array([candle['volume'] for candle in candles])

    # Compute the technical indicators

    indicators['rsi_9'] = stream.RSI(close, timeperiod=9)
    indicators['rsi_14'] = stream.RSI(close, timeperiod=14)
    indicators['rsi_21'] = stream.RSI(close, timeperiod=21)

    # MACD
    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = (
        stream.MACD(close, fastperiod=10, slowperiod=24, signalperiod=9)
    )

    indicators['upper'], indicators['middle'], indicators['lower'] = stream.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    indicators['fastk'], indicators['fastd'] = stream.STOCHRSI(
        close, timeperiod=10, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    indicators['adx'] = stream.ADX(high, low, close, timeperiod=10)
    indicators['volume_ema'] = stream.EMA(volume, timeperiod=10)

    conversion = stream.EMA(close, timeperiod=9)
    base = stream.EMA(close, timeperiod=21)
    leading_span_a = (conversion + base) / 2
    leading_span_b = stream.EMA(close, timeperiod=40)
    indicators['ichimoku_base'] = conversion
    indicators['ichimoku_span_a'] = leading_span_a
    indicators['ichimoku_span_b'] = leading_span_b
    indicators['mfi'] = stream.MFI(high, low, close, volume, timeperiod=10)

    indicators['atr'] = stream.ATR(high, low, close, timeperiod=10)

    indicators['price_roc'] = stream.ROC(close, timeperiod=6)

    indicators['sma_7'] = stream.SMA(close, timeperiod=7)
    indicators['sma_14'] = stream.SMA(close, timeperiod=14)
    indicators['sma_21'] = stream.SMA(close, timeperiod=21)

    final_message = {**candle, **indicators}

    # breakpoint()

    return final_message
