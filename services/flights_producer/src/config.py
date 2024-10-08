from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import field_validator

# Define a class to hold the configuration settings
class Config(BaseSettings):
    aviation_edge_api_key: str
    kafka_broker_address: Optional[str] = None #this is optional because quixcloud will provide the kafka broker address
    kafka_topic_name: str
    live_or_historical: str
    days : Optional[float] = 1

    # Validate the value of the live_or_historical settings
    @field_validator("live_or_historical")
    def validate_live_or_historical(
        cls, value
    ):  # cls is the class itslef andd value is the value of the field
        if value not in {"live", "historical"}:
            raise ValueError("Invalid value for live_or_historical")
        return value


# Create an instance of the Config class
config = Config()
