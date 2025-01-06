# from llms.claude import ClaudeNewsSignalExtractor
from llms.base import BaseNewsSignalExtractor
from loguru import logger
from ollama._types import ResponseError
from quixstreams import Application


def process_news(value, llm):
    """Process a single news item and extract trading signals.

    Args:
        value: The news item dictionary containing 'title' and other fields
        llm: The LLM-based signal extractor instance

    Returns:
        dict: Processed message with signal data
    """
    try:
        signal_data = llm.get_signal(value['title'])
        return {
            'news': value['title'],
            **signal_data,
            'model_name': llm.model_name,
            'error': None,
        }
    except ResponseError as e:
        if 'model requires more system memory' in str(e):
            logger.error(f'Insufficient memory for model. Error: {e}')
            return {
                'news': value['title'],
                'sentiment': 'neutral',  # Default fallback
                'confidence': 0.0,
                'model_name': llm.model_name,
                'error': str(e),
            }
        raise  # Re-raise other ResponseErrors
    except Exception as e:
        logger.error(f'Error processing news: {e}')
        return {
            'news': value['title'],
            'sentiment': 'neutral',  # Default fallback
            'confidence': 0.0,
            'model_name': llm.model_name,
            'error': str(e),
        }


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    llm: BaseNewsSignalExtractor,
):
    """Process news articles and extract trading signals using LLM.

    Args:
        kafka_broker_address: The address of the Kafka broker.
        kafka_input_topic: The topic to consume news from.
        kafka_output_topic: The topic to produce signals to.
        kafka_consumer_group: The consumer group ID.
        llm: The LLM-based signal extractor instance.
    """
    logger.info('Hello from news-signal!')

    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='earliest',
    )

    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json')

    output_topic = app.topic(name=kafka_output_topic, value_serializer='json')

    sdf = app.dataframe(input_topic)

    # Process the incoming news and output the signal
    sdf = sdf.apply(
        lambda value: {
            'news': value['title'],
            **llm.get_signal(value['title']),
            'model_name': llm.model_name,
            'timestamp_ms': value['timestamp_ms'],
        }
    )

    sdf = sdf.update(lambda value: logger.debug(f'final message: {value}'))

    sdf = sdf.to_topic(output_topic)

    app.run()


if __name__ == '__main__':
    from config import config
    from llms.factory import get_llm

    logger.info(f'Using model provider: {config.model_provider}')
    llm = get_llm(config.model_provider)

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        llm=llm,
    )
