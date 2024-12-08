from datetime import timedelta
from typing import Any, List, Optional, Tuple

from loguru import logger
from quixstreams.models import TimestampType


def custom_ts_extractor(
    value: Any,
    headers: Optional[List[Tuple[str, bytes]]],
    timestamp: float,
    timestamp_type: TimestampType,
) -> int:
    """
    Specifying a custom timestamp extractor to use the timestamp from the message payload
    instead of Kafka timestamp.
    """
    # breakpoint()
    return value['timestamp_ms']


def init_candle(trade: dict) -> dict:
    """
    Initialize a candle with the first trade
    """
    return {
        'open': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'close': trade['price'],
        'volume': trade['volume'],
    }


def update_candle(candle: dict, trade: dict) -> dict:
    """
    Update the candle with the latest trade
    """
    candle['close'] = trade['price']
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['volume'] += trade['volume']
    return candle


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    candle_seconds: int,
):
    """
    3 steps:
    1. Ingests trades from Kafka
    2. Generates candles using tumbling windows and
    3. Outputs candles to Kafka

    Args:
        kafka_broker_address (str): Kafka broker address
        kafka_input_topic (str): Kafka input topic
        kafka_output_topic (str): Kafka output topic
        kafka_consumer_group (str): Kafka consumer group
        candle_seconds (int): Candle seconds
    Returns:
        None
    """
    logger.info('Starting the candles service!')

    from quixstreams import Application

    # Initialize the Quix Streams application
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )
    # app.clear_state()

    # Define the input and output topics
    input_topic = app.topic(
        name=kafka_input_topic,
        value_deserializer='json',
        timestamp_extractor=custom_ts_extractor,
    )
    output_topic = app.topic(
        name=kafka_output_topic,
        value_serializer='json',
    )

    # Create a Streaming DataFrame from the input topic
    sdf = app.dataframe(topic=input_topic)

    # sdf = sdf.apply(lambda value: logger.info(f"Received trade: {value}"))

    # # Define the tumbling window
    # sdf = sdf.tumbling_window(timedelta(seconds=candle_seconds))

    # # Apply the reducer to update the candle, or initialize it with the first trade
    # sdf = sdf.reduce(reducer=update_candle, initializer=init_candle)

    # # Emit all intermediate candles to make the system more responsive
    # sdf = sdf.current()

    # # If you wanted to emit the final candle only, you could do this:
    # # sdf = sdf.final()

    # NOTE: you can pipe these operations together like in this example:
    # https://quix.io/docs/quix-streams/windowing.html#updating-window-definitions

    sdf = (
        # Define a tumbling window of 10 minutes
        sdf.tumbling_window(timedelta(seconds=candle_seconds))
        # Create a "reduce" aggregation with "reducer" and "initializer" functions
        .reduce(reducer=update_candle, initializer=init_candle)
        # Emit results only for closed windows
        .final()
    )
    # push the candle to the output topic
    sdf = sdf.to_topic(topic=output_topic)

    # Start the application
    app.run()


if __name__ == '__main__':
    from config import config

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        candle_seconds=config.candle_seconds,
    )
