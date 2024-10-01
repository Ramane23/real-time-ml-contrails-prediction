from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os
from dotenv import load_dotenv

env_file = "../setup_historical_config.env"
# Load the environment variables from the .env file
load_dotenv(dotenv_path=env_file)

class Config(BaseSettings):
    
    # feature group our feature view reads data from
    feature_group_name: str
    feature_group_version: int

    # feature view name and version
    feature_view_name: str
    feature_view_version: int

    # required to authenticate with Hopsworks API
    hopsworks_project_name: str
    hopsworks_api_key: str
    
    #to read from the kafka topic
    kafka_topic_name: str
    kafka_broker_address: str
    kafka_consumer_group: str
    live_or_historical : str
    last_n_minutes: Optional[int] = None
     
    # Validate the value of the live_or_batch setting
    @field_validator("live_or_historical")
    def validate_live_or_historical(
        cls, value
    ):  # cls is the class itslef andd value is the value of the field
        if value not in {"live", "historical"}:
            raise ValueError("Invalid value for live_or_historical")
        return value

config = Config()