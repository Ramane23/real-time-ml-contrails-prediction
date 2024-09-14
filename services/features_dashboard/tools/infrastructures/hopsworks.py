from typing import List, Optional, Tuple, Dict, Any
import time
from loguru import logger
import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
from hsfs.client.exceptions import FeatureStoreException
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView


# This class is responsible for writing the flights data to a feature group in Hopsworks
class HopsworksFlightsWriter:
    """_This class is responsible for writing the flights data to a feature group in Hopsworks_"""
    
    def __init__(
        self,
        feature_group_name : str, 
        feature_group_version : int, 
        hopsworks_project_name : str, 
        hopsworks_api_key : str         
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
        # Get the feature store
        feature_store = self.get_feature_store()

        feature_group = feature_store.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
            description='Flights tracking data enriched with weather data',
            primary_key=[
                         'flight_id', 
                         'latitude',
                         'longitude',
                         'current_flight_time', 
                         'flight_level'
                      ],
            event_time='current_flight_time',
            online_enabled=True, # Enable online feature serving
        )

        return feature_group
    
    #Function that writes the data to the feature group
    def push_flight_data_to_feature_store(
        self,
        flight_data: List[dict],
        online_or_offline: str,
    ) -> None:
        """
        Pushes the given `flight_data` to the feature store, writing it to the feature group
        with name `feature_group_name` and version `feature_group_version`.

        Args:
            feature_group_name (str): The name of the feature group to write to.
            feature_group_version (int): The version of the feature group to write to.
            flight_data (List[dict]): The flight data to write to the feature store.
            online_or_offline (str): Whether we are saving the `flight_data` to the online or offline
            feature group.

        Returns:
            None
        """
        # Authenticate with Hopsworks API
        project = hopsworks.login(
            project=self.hopsworks_project_name,
            api_key_value=self.hopsworks_api_key,
        )
        #breakpoint()
        # Get the feature store
        feature_store = project.get_feature_store()

        # Get or create the feature group we will be saving flight data to
        flight_feature_group = feature_store.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
            description='Flight tracking data enriched with weather data',
            primary_key=[
                         'flight_id', 
                         'latitude',
                         'longitude',
                         'current_flight_time', 
                         'flight_level'
                      ],
            event_time='current_flight_time',
            online_enabled=True,
        )

        # Transform the flight data (dict) into a pandas dataframe
        flight_data_df = pd.DataFrame(flight_data)

        # Write the flight data to the feature group
        flight_feature_group.insert(
            flight_data_df,
            write_options={
                'start_offline_materialization': True # we are telling the feature store to start copying the data to the offline database 
                if online_or_offline == 'offline' #if the online_offline argument is set to offline
                else False
            },
        )
        
        return None
    
#A class that reads the flights data from a feature group in Hopsworks
class HopsworksFlightsReader:
    """This class reads the flights data from a feature group in Hopsworks"""
    
    def __init__(
        self,
        feature_store : FeatureStore,
        feature_group_name : str, 
        feature_group_version : int ,
        feature_view_name : Optional[str], 
        feature_view_version : Optional[int] 
        ):
        
        self.feature_store = feature_store
        self.feature_group_name  = feature_group_name
        self.feature_group_version  = feature_group_version
        self.feature_view_name  = feature_view_name
        self.feature_view_version  = feature_view_version
        
        
    #Function to get the primary keys of the feature group, which is necessary to read the data from the 
    #online feature store
    def _get_primary_keys(
        self,
        flight_id: str,
        current_flight_time: int,
        #altitude: float,
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
            'latitude': latitude,
            'longitude': longitude,  
            'current_flight_time': current_flight_time,
            'flight_level': flight_level,
        }
        
        return primary_keys

    def get_primary_keys(
        self,
        last_n_minutes: int
        ) -> List[Dict]:
        """
        Returns a list of dictionaries with the primary keys of the rows we want to fetch
         
        Args:  
            last_n_minutes (int): The number of minutes to go back in time.
        
        Returns:
            List[Dict]: A list of dictionaries with the primary keys.
        """
        # get current UTC in seconds and floor it to the previous minute
        current_utc = int(time.time() * 1000)
        current_utc = current_utc - (current_utc % 60000)

        # generate a list of timestamps in miliseconds for the last 'last_n_minutes' minutes
        current_flight_times = [current_utc - i * 60000 for i in range(last_n_minutes)]
       
        primary_keys = [
            {
            #'flight_id': flight_id,
            #'latitude': latitude,
            #'longitude': longitude,  
            'current_flight_time': current_flight_time,
            #'flight_level': flight_level,
            } for current_flight_time in current_flight_times
        ]

        # breakpoint()

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
        #altitude: float,
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
        primary_keys = self._get_primary_keys(
            current_flight_time=current_flight_time,
            latitude=latitude,
            longitude=longitude,
            flight_id=flight_id,
            flight_level=flight_level
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

    def get_features_from_the_store(
        self,
        online_or_offline: str,
        ) -> pd.DataFrame:
        """
        Fetches the features from the store and returns them as a pandas DataFrame.
        All the config parameters are read from the src.config module

        Args:
            None

        Returns:
            pd.DataFrame: The features as a pandas DataFrame sorted by timestamp (ascending)
        """
        logger.debug('Getting the feature view')
        feature_view = self._get_feature_view()    

        # For the moment, let's get all rows from this feature group
        if online_or_offline == 'offline':
            try:
                #retrieving the data from the offline feature store which is a structured store (like SQLite)
                features: pd.DataFrame = feature_view.get_batch_data()

            except FeatureStoreException:
                # breakpoint()
                # retry the call with the use_hive option. This is what Hopsworks recommends
                features: pd.DataFrame = feature_view.get_batch_data(read_options={"use_hive": True})
        else:
            # we fetch from the online feature store.
            # we need to build this list of dictionaries with the primary keys because the online feature store does not support batch reads
            #it is designed for fast online reads of individual rows and require the primary keys to be passed
            features = feature_view.get_feature_vectors(
                entry= self._get_primary_keys(last_n_minutes=20),
                return_type="pandas"
            )

        # sort the features by timestamp (ascending)
        features = features.sort_values(by='current_flight_time')

        # breakpoint()

        # Python trick: You can also do a sort inplace. I think with this you avoid copying data and it is
        # paster
        # features.sort_values(by='timestamp', inplace=True)

        return features
