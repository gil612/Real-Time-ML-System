from config import config
from loguru import logger
from quixstreams import State

MAX_CANDLES_IN_STATE = config.max_candles_in_state


def update_candles(candle: dict, state: State) -> dict:
    """
    Updates the list of cnadles we have in our state using the latest candle

    If the latest candle corresponds to a new window, we just append it to the list, total numer of candles is less than the candles we want to keep
    If the latest candle corresponds to the last window, we replace the candle in the list.

    Args:
        candle: The latest candle
        state: The state of the application
    Returns:
       None
    """
    # Get the candles from our state
    candles = state.get('candles', default=[])
    if not candles:
        candles.append(candle)

    # If the latest candle corresponds to a new window, we just append it to the list.
    elif same_window(candle, candles[-1]):
        candles[-1] = candle
    else:
        candles.append(candle)

    if len(candles) > MAX_CANDLES_IN_STATE:
        candles.pop(0)

    # TODO: we should check the candles have no missing windows
    # This can happen for low volume pairs. In this case, we could interpoalte the missing windows
    logger.debug(f'numer of candles in state: {len(candles)}')
    state.set('candles', candles)

    return candle


def same_window(candle_1: dict, candle_2: dict) -> bool:
    """
    Checks if the latest candle corresponds to the last window

    Args:
    candle: The current candle
    last_candle: The last candle
    Returns:
        True if the candles are in the same window, False otherwise
    """
    return (
        candle_1['window_start_ms'] == candle_2['window_start_ms']
        and candle_1['window_end_ms'] == candle_2['window_end_ms']
        and candle_1['pair'] == candle_2['pair']
    )
