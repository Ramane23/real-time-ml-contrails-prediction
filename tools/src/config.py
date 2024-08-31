from pydantic_settings import BaseSettings
from pydantic import field_validator

# Define a class to hold the configuration settings
class Config(BaseSettings):
    hopsworks_project_name: str
    hopsworks_api_key: str
    feature_group_name: str
    feature_group_version: int
    
    
# Create an instance of the Config class
config = Config()
