from pydantic_settings import BaseSettings
from pydantic import field_validator
# import dotenv

# Load the environment variables from the .env file
# dotenv.load_dotenv(dotenv.find_dotenv())


# Define a class to hold the configuration settings
class Config(BaseSettings):
    aviation_edge_api_key: str
    kafka_broker_address: str
    kafka_topic_name: str
    live_or_historical: str
    days : float

    # Validate the value of the live_or_historical setting
    @field_validator("live_or_historical")
    def validate_live_or_historical(
        cls, value
    ):  # cls is the class itslef andd value is the value of the field
        if value not in {"live", "historical"}:
            raise ValueError("Invalid value for live_or_historical")
        return value


# Create an instance of the Config class
config = Config()
