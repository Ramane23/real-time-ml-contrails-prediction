from pydantic_settings import BaseSettings
from pydantic import field_validator

# Define a class to hold the configuration settings
class Config(BaseSettings):
    kafka_broker_address: str
    kafka_input_topic_name: str
    kafka_consumer_group: str
    kafka_output_topic_name: str
    meteomatics_username: str
    meteomatics_password: str

# Create an instance of the Config class
config = Config()
