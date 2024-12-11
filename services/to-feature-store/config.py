from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='settings.env')
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_consumer_group: str

    feature_group_name: str
    feature_group_version: str


# class HopsworkCredentials(BaseSettings):
#     model_config = SettingsConfigDict(env_file='credentials.env')
#     hopswork_api_key: str


config = Settings()
# hopswork_credentials = HopsworkCredentials()
