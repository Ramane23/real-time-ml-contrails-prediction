import hopsworks
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup
from loguru import logger
import pandas as pd

from src.config import config

# This class is responsible for writing the flights data to a feature group in Hopsworks
class HopsworksFlightsWriter:
    """_This class is responsible for writing the flights data to a feature group in Hopsworks_"""
    
    def __init__(
        self,
        hopsworks_project_name: str,
        hopsworks_api_key: str,
        feature_group_name: str,
        feature_group_version: int,
    ):
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.hopsworks_project_name = hopsworks_project_name
        self.hopsworks_api_key = hopsworks_api_key
    
    #Function that gets a pointer to the feature store
    def get_feature_store(self) -> FeatureStore:
        """Connects to Hopsworks and returns a pointer to the feature store

        Returns:
            hsfs.feature_store.FeatureStore: pointer to the feature store
        """
        # Log in to Hopsworks and get the project
        project = hopsworks.login(
            project= self.hopsworks_project_name,
            api_key_value= self.hopsworks_api_key
        )
        # Return the feature store for the project
        return project.get_feature_store()
    
    #Function that gets a pointer to the feature group
    def _get_feature_group(self) -> FeatureGroup:
        """
        Returns (and possibly creates) the feature group we will be writing to.
        """
        # Authenticate with Hopsworks API
        project = hopsworks.login(
            project=self.hopsworks_project_name,
            api_key_value=self.hopsworks_api_key,
        )

        # Get the feature store
        feature_store = self.get_feature_store()

        feature_group = feature_store.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
            description='Flights tracking data enriched with weather data',
            primary_key=['flight_id', 
                         'current_flight_time', 
                         'altitude', 
                         'flight_level', 
                         'latitude', 
                         'longitude'
                         ],
            event_time='current_flight_time',
            online_enabled=True, # Enable online feature serving
        )

        return feature_group
    
#Instantiate the HopsworksFlightsWriter class
hopsworks_flights_writer = HopsworksFlightsWriter(
    hopsworks_project_name=config.hopsworks_project_name,
    hopsworks_api_key=config.hopsworks_api_key,
    feature_group_name=config.feature_group_name,
    feature_group_version=config.feature_group_version,
)

    