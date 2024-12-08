from loguru import logger


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    candle_seconds: int,
):
    """
    1. Ingests trades from Kafka
    2. Generates candles using tumbling window and
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
    logger.info('Starting candles service!')

    from quixstreams import Application

    # Initialize the Quix application
    app = Application(
        kafka_broker_address=kafka_broker_address,
        kafka_consumer_group=kafka_consumer_group,
    )

    # Define the input and output topic
    input_topic = app.get_topic(name=kafka_input_topic, value_deserializer='json')

    output_topic = app.get_topic(
        name=kafka_output_topic,
        value_serializer='json',
    )
    # Create a streaming DataFrame
    sdf = app.dataframe(topic=input_topic)


if __name__ == '__main__':
    main()
