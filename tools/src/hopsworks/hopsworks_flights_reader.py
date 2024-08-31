import os
from typing import List, Optional, Tuple, Dict, Any
import time
from loguru import logger
import pandas as pd
import hopsworks

from hsfs.feature_view import FeatureView
from src.hopsworks.hopsworks_flights_writer import hopsworks_flights_writer

class HopsworksFlightsReader:
    """This class reads the flights data from a feature group in Hopsworks"""
    
    def __init__(
        self,
        feature_view_name: Optional[str] = None,
        feature_view_version: Optional[int] = None,
        ):
        
        self.feature_store = hopsworks_flights_writer.get_feature_store()
        self.feature_group_name = hopsworks_flights_writer.feature_group_name
        self.feature_group_version = hopsworks_flights_writer.feature_group_version
        self.feature_view_name = feature_view_name
        self.feature_view_version = feature_view_version
        
        
    #Function to get the primary keys of the feature group, which is necessary to read the data from the 
    #online feature store
    def _get_primary_keys_to_read_from_online_store(
        self,
        flight_id: str,
        current_flight_time: int,
        altitude: float,
        flight_level: str,
        latitude: float,
        longitude: float
        ) -> dict:
        """
        Get the primary keys of the feature group, which is necessary to read the data from the online feature store.
        
        Args:
            flight_id (str): The ID of the flight.
            current_flight_time (int): The current flight time in a suitable timestamp format.
            altitude (float): The altitude of the flight.
            flight_level (str): The flight level (e.g., FL300).
            latitude (float): The latitude of the flight.
            longitude (float): The longitude of the flight.
        
        Returns:
            dict: A dictionary with the primary keys mapped to their values.
        """
        primary_keys = {
            'flight_id': flight_id,
            'current_flight_time': current_flight_time,
            'altitude': altitude,
            'flight_level': flight_level,
            'latitude': latitude,
            'longitude': longitude
        }
        
        return primary_keys

    #Function to get the feature view
    def _get_feature_view(self) -> FeatureView:
        """
        Returns the feature view object that reads data from the feature store
        """
        if self.feature_group_name is None:
            # We try to get the feature view without creating it.
            # If it does not exist, we will raise an error because we would
            # need the feature group info to create it.
            try:
                return self.feature_store.get_feature_view(
                    name=self.feature_view_name,
                    version=self.feature_view_version,
                )
            except Exception as e:
                raise ValueError(
                    'The feature group name and version must be provided if the feature view does not exist.'
                )
        
        # We have the feature group info, so we first get it
        feature_group = self.feature_store.get_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
        )

        # and we now create it if it does not exist
        feature_view = self.feature_store.get_or_create_feature_view(
            name=self.feature_view_name,
            version=self.feature_view_version,
            query=feature_group.select_all(),
        )
        # and if it already existed, we check that its feature group name and version match
        # the ones we have in `self.feature_group_name` and `self.feature_group_version`
        # otherwise we raise an error
        possibly_different_feature_group = \
            feature_view.get_parent_feature_groups().accessible[0]
        
        if possibly_different_feature_group.name != feature_group.name or \
            possibly_different_feature_group.version != feature_group.version:
            raise ValueError(
                'The feature view and feature group names and versions do not match.'
            )
        
        return feature_view
    
    #Function to read the data from the online feature store
    def read_from_online_store(
        self,
        flight_id: str,
        current_flight_time: int,
        altitude: float,
        flight_level: str,
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        """
        Reads flight tracking data enriched with weather data from the online feature store
        for the given `flight_id` and the associated primary keys.

        Args:
            flight_id (str): The flight ID for which we want to get the data.
            current_flight_time (int): The timestamp of the flight in milliseconds.
            altitude (float): The altitude of the flight.
            flight_level (str): The flight level (e.g., FL350).
            latitude (float): The latitude of the flight.
            longitude (float): The longitude of the flight.

        Returns:
            pd.DataFrame: A DataFrame containing the requested data sorted by timestamp.
        """
        # Get the primary keys for the query
        primary_keys = self._get_primary_keys_to_read_from_online_store(
            flight_id=flight_id,
            current_flight_time=current_flight_time,
            altitude=altitude,
            flight_level=flight_level,
            latitude=latitude,
            longitude=longitude,
        )
        logger.debug(f'Primary keys: {primary_keys}')

        # Get the feature view
        feature_view = self._get_feature_view()

        # Fetch the feature vectors using the primary keys
        features = feature_view.get_feature_vectors(
            entry=primary_keys,
            return_type="pandas"
        )

        # Sort the features by timestamp and reset the index
        features = features.sort_values(by='current_flight_time').reset_index(drop=True)

        return features

    #Function to read the data from the offline feature store
    def read_from_offline_store(
        self,
        flight_id: str,
        last_n_days: int,
        ) -> pd.DataFrame:
        """
        Reads flight tracking data enriched with weather data from the offline feature store
        for the given flight_id and the specified time range (last_n_days).

        Args:
            flight_id (str): The flight ID for which we want to get the data.
            last_n_days (int): The number of days to go back in time.

        Returns:
            pd.DataFrame: A DataFrame containing the requested data sorted by current_flight_time.
        """
        # Calculate the current timestamp and the starting timestamp based on last_n_days
        to_timestamp_s = int(time.time())  # Convert current time to seconds
        from_timestamp_s = to_timestamp_s - last_n_days * 24 * 60 * 60  # Convert days to seconds

        # Get the feature view associated with your feature group
        feature_view = self._get_feature_view()
        features = feature_view.get_batch_data()

        # Filter the features for the given flight_id and time range
        features = features[features['flight_id'] == flight_id]
        features = features[features['current_flight_time'] >= from_timestamp_s]
        features = features[features['current_flight_time'] <= to_timestamp_s]

        # Sort the features by current_flight_time (ascending)
        features = features.sort_values(by='current_flight_time').reset_index(drop=True)

        return features