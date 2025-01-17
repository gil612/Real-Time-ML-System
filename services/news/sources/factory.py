from typing import Literal, Optional, Union
from loguru import logger

from .historical_data_source import HistoricalNewsDataSource
from .news_data_source import NewsDataSource as LiveNewsDataSource
from config import config

NewsDataSource = Union[LiveNewsDataSource, HistoricalNewsDataSource]


def get_source(
    data_source: Literal["live", "historical"],
    polling_interval_sec: Optional[int] = 10,
) -> NewsDataSource:
    if data_source == "live":
        # Set up the source to download news from the CryptoPanic API
        from config import cryptopanic_config

        from .news_downloader import NewsDownloader

        # News Downloader object
        news_downloader = NewsDownloader(cryptopanic_api_key=cryptopanic_config.api_key)

        # Quix Streams data source that wraps the news downloader
        news_source = LiveNewsDataSource(
            news_downloader=news_downloader,
            polling_interval_sec=polling_interval_sec,
        )

        return news_source

    elif data_source == "historical":
        # Debug log BEFORE creating source
        logger.info("Creating historical source with:")
        logger.info(f"URL RAR file: {config.historical_data_source_url_rar_file!r}")
        logger.info(f"Days back: {config.historical_days_back!r}")
        logger.info(f"All config: {config.model_dump()}")

        source = HistoricalNewsDataSource(
            url_rar_file=config.historical_data_source_url_rar_file,
            days_back=config.historical_days_back,
        )
        logger.info("Source created successfully")
        return source

    else:
        raise ValueError(f"Invalid data source: {data_source}")
