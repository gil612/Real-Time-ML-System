from loguru import logger
from quixstreams import Application


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    num_candles_in_state: int,
):
    """
    3 stages:
    1. Ingests candles form the kafka input topic
    2. Computes technical indicators
    3. Sends the technical indicators to the kafka output topic
    Args:
        kafka_broker_address: The address of the kafka broker
        kafka_input_topic: The topic to ingest candles from
        kafka_output_topic: The topic to push technical indicators to
        kafka_consumer_group: The consumer group to use
        num_candles_in_state: The number of candles to use for the technical indicators
    Returns:
        None
    """
    logger.info('Hello from technical-indicators!')

    app = Application(
        broker_address=kafka_broker_address, consumer_group=kafka_consumer_group
    )

    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json')

    output_topic = app.topic(name=kafka_output_topic, value_serializer='json')

    sdf = app.dataframe(topic=input_topic)

    sdf = sdf.update(lambda value: logger.info(f'Candle: {value}'))

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
        num_candles_in_state=config.num_candles_in_state,
    )
