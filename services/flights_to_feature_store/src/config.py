from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    kafka_broker_address: Optional[str] = None
    kafka_topic_name: str
    kafka_consumer_group: str
    feature_group_name: str
    feature_group_version: int
    feature_view_name: Optional[str]
    feature_view_version: Optional[int] 
    # required to authenticate with Hopsworks API
    hopsworks_project_name: str
    hopsworks_api_key: str

    # by default we want our `kafka_to_feature_store` service to run in live mode
    live_or_historical: str = 'live'

    # buffer size to store messages before writing to the feature store
    buffer_size: int

    # force save to feature store every n seconds
    save_every_n_sec: int = 600

    # whether to create a new consumer group or not
    create_new_consumer_group: bool = False
    
    

    @field_validator('live_or_historical')
    @classmethod
    def validate_live_or_historical(cls, value):
        assert value in {
            'live',
            'historical',
        }, f'Invalid value for live_or_historical: {value}'
        return value


config = Config()