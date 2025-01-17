from datetime import datetime, timezone

import hopsworks
import pandas as pd

# from hopsworks.exceptions import FeatureStoreException
from loguru import logger
from quixstreams.sinks.base import BatchingSink, SinkBackpressureError, SinkBatch


class HopsworksFeatureStoreSink(BatchingSink):
    """
    Some sink writing data to a database
    """

    def __init__(
        self,
        api_key: str,
        project_name: str,
        feature_group_name: str,
        feature_group_version: int,
        feature_group_primary_keys: list[str],
        feature_group_event_time: str,
        feature_group_materialization_interval_minutes: int,
    ):
        """
        Establish a connection to the Hopsworks Feature Store
        """
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.materialization_interval_minutes = (
            feature_group_materialization_interval_minutes
        )

        # Establish a connection to the Hopsworks Feature Store
        project = hopsworks.login(project=project_name, api_key_value=api_key)
        self._fs = project.get_feature_store()

        # Get the feature group
        self._feature_group = self._fs.get_or_create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            primary_key=feature_group_primary_keys,
            event_time=feature_group_event_time,
            online_enabled=True,
        )

        # set the materialization interval
        try:
            self._feature_group.materialization_job.schedule(
                cron_expression=f"0 0/{self.materialization_interval_minutes} * ? * * *",
                start_time=datetime.now(tz=timezone.utc),
            )
        # TODO: handle the FeatureStoreException
        except Exception as e:
            logger.error(f"Failed to schedule materialization job: {e}")

        # call constructor of the base class to make sure the batches are initialized
        super().__init__()

    def write(self, batch: SinkBatch):
        # Transform the batch into a pandas DataFrame
        data = []
        for item in batch:
            # Each item.value might be a list of records
            if isinstance(item.value, list):
                data.extend(item.value)
            else:
                data.append(item.value)

        data = pd.DataFrame(data)

        # Add debug logging to see the columns
        logger.debug(f"DataFrame columns: {data.columns}")
        logger.debug(
            f"First row: {data.iloc[0] if not data.empty else 'Empty DataFrame'}"
        )

        # Remove duplicate insert
        try:
            self._feature_group.insert(data)
        except Exception as err:
            raise SinkBackpressureError(
                retry_after=30.0,
                topic=batch.topic,
                partition=batch.partition,
            ) from err
