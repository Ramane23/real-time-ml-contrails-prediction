from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List
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
    kafka_broker_address: Optional[str] = None
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
    
    comet_ml_api_key: str
    comet_ml_project_name: str
    comet_ml_workspace: str
    twenty_busiest_european_routes: List[str] = [
        'Toulouse - Paris', 'Madrid - Barcelona', 'Nice - Paris', 'Catania - Rome', 
        'Berlin - Munich', 'Oslo - Trondheim', 'Frankfurt - Berlin', 'Oslo - Bergen', 
        'Munich - Hamburg', 'London - Dublin', 'Barcelona - Palma de Mallorca', 
        'Paris - Marseille', 'Stockholm - Gothenburg', 'Athens - Thessaloniki', 
        'Vienna - Zurich', 'Milan - Rome', 'Helsinki - Oulu', 'Madrid - Malaga', 
        'London - Amsterdam', 'Paris - Lyon'
    ]


config = Config()