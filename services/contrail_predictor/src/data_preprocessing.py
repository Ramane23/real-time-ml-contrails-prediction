from hopsworks_fs import GetFeaturesFromTheStore
from config import config 
import pandas as pd
from loguru import logger
import datetime
import warnings
import numpy as np
from typing import List
from tqdm import tqdm
from openap import prop, FuelFlow, Emission


#Ignore warnings
warnings.filterwarnings('ignore')

#Creating a class to handle data preprocessing
class DataPreprocessing:
    def __init__(self):
        self.static_columns = [
            'flight_id', 'route', 'aircraft_mtow_kg', 'aircraft_malw_kg', 
            'aircraft_engine_class', 'aircraft_num_engines', 'aircraft_type',
            'departure_airport_lat', 'arrival_airport_lat', 'departure_airport_long', 'arrival_airport_long'
        ]
        self. dynamic_columns = [
            'mach_number', 'true_airspeed_ms', 'altitude', 'flight_level', 'latitude', 'longitude', 'horizontal_speed',
            'vertical_speed', 'temperature_c', 'pressure_hpa', 'wind_speed_u_ms', 'wind_speed_v_ms',
            'wind_speed_ms', 'geopotential_height_m', 'relative_humidity_percent', 
            'specific_humidity_kg_kg', 'total_cloud_cover_octas', 'high_cloud_cover_octas', 'global_radiation_w_m2',
            'prob_contrails_percent', 'direction'
        ]
        self.not_altitude_related_columns = [
            'mach_number', 'true_airspeed_ms', 'horizontal_speed', 'vertical_speed', 'latitude', 'longitude', 
            'departure_airport_lat', 'arrival_airport_lat', 'departure_airport_long', 'arrival_airport_long'
        ]
        self.weather_columns = [
            'temperature_c', 'pressure_hpa', 'wind_speed_u_ms', 'wind_speed_v_ms',
            'wind_speed_ms', 'geopotential_height_m', 'relative_humidity_percent', 
            'specific_humidity_kg_kg', 'total_cloud_cover_octas', 'high_cloud_cover_octas', 'global_radiation_w_m2',
            'prob_contrails_percent'
        ]
        self.altitude_related_columns = [
            'altitude', 'temperature_c', 'pressure_hpa', 'geopotential_height_m',
            'relative_humidity_percent', 'specific_humidity_kg_kg', 'total_cloud_cover_octas',
            'high_cloud_cover_octas', 'global_radiation_w_m2', 'prob_contrails_percent'
        ]
        self.columns_to_keep = [
            'current_flight_time', 'route', 'flight_id', 'aircraft_icao_code', 
            'global_radiation_w_m2', 'direction', 'departure_airport_lat', 'arrival_airport_lat', 
            'departure_airport_long', 'arrival_airport_long','aircraft_mtow_kg', 'aircraft_malw_kg', 'aircraft_engine_class',
            'aircraft_num_engines', 'horizontal_speed', 'mach_number', 'true_airspeed_ms', 'altitude', 'flight_level', 
            'latitude', 'longitude', 'vertical_speed', 'temperature_c', 'pressure_hpa', 'wind_speed_u_ms', 'wind_speed_v_ms',
            'wind_speed_ms', 'geopotential_height_m', 'relative_humidity_percent', 'specific_humidity_kg_kg',
            'total_cloud_cover_octas', 'high_cloud_cover_octas', 'prob_contrails_percent', 
        ]
        pass

    def _get_flight_data(self) -> pd.DataFrame:
        
        """a method to retrieve the flights data from the hopsworks feature store

        Returns:
            flights_data (pd.DataFrame): a pandas dataframe containing the flights data
        """
        logger.debug('Retrieving the flights data from the hopsworks feature store...')
        #Instantiate the GetFeaturesFromTheStore class
        get_features_from_the_store = GetFeaturesFromTheStore()
        
        #retrieve the flights data from the hopsworks feature store
        flights_data = get_features_from_the_store.get_features(live_or_historical=config.live_or_historical)
        
        #mask the data to keep only the columns to keep
        flights_data = flights_data[self.columns_to_keep]
        
        # set the index to current_flight_time
        flights_data = flights_data.set_index('current_flight_time')
        
        #rename the aircrat_icao_code column to aircraft_type
        flights_data.rename(columns={'aircraft_icao_code': 'aircraft_type'}, inplace=True)
        
        #cast latitude, longitude to float
        flights_data['latitude'] = flights_data['latitude'].astype(float)
        flights_data['longitude'] = flights_data['longitude'].astype(float)
        
        #Converting the 'current_flight_time' column to a timestamp
        flights_data.index = flights_data.index.map(lambda x: datetime.datetime.fromtimestamp(x))
        
        # Remove seconds, focus only on hours and minutes
        flights_data.index = pd.to_datetime(flights_data.index).floor('min')
        
        logger.debug(f"there are {flights_data.shape[0]} rows and {flights_data.shape[1]} columns in the data")
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data.head(10))
        #breakpoint()
        return flights_data
    
    #method to aggregate the flights data by flight_time and flight_id
    def _aggregate_flights_data(
        self, 
        flights_data : pd.DataFrame
        ) -> pd.DataFrame:
        
        """a method to aggregate the flights data by flight_time and flight_id

        Args:
            flights_data (pd.DataFrame): a pandas dataframe containing the flights data

        Returns:
            flights_data_by_flight_id (pd.DataFrame): a pandas dataframe containing the aggregated flights data
        """
        logger.debug('Aggregating the flights data by flight_id and current_flight_time...')
        # Aggregate by taking the mean for dynamic features and first for static features
        flights_data_by_flight_id = flights_data.groupby(['current_flight_time', 'flight_id']).agg({
            'route' : 'first',
            'aircraft_type': 'first',
            'aircraft_mtow_kg': 'first',
            'aircraft_malw_kg': 'first',
            'aircraft_engine_class': 'first',
            'aircraft_num_engines': 'first',
            'mach_number': 'mean',
            'true_airspeed_ms': 'mean',
            'altitude': 'mean',
            'flight_level': 'first',
            'latitude': 'mean',
            'longitude': 'mean',
            'horizontal_speed': 'mean',
            'vertical_speed': 'mean',
            'temperature_c': 'mean',
            'pressure_hpa': 'mean',
            'wind_speed_u_ms': 'mean',
            'wind_speed_v_ms': 'mean',
            'wind_speed_ms': 'mean',
            'geopotential_height_m': 'mean',
            'relative_humidity_percent': 'mean',
            'specific_humidity_kg_kg': 'mean',
            'total_cloud_cover_octas': 'mean',
            'high_cloud_cover_octas': 'mean',
            'global_radiation_w_m2': 'mean',
            'prob_contrails_percent': 'mean',
            'direction': 'mean',
            'departure_airport_lat': 'first',
            'arrival_airport_lat': 'first',
            'departure_airport_long': 'first',
            'arrival_airport_long': 'first'
        }).reset_index()

        # Sort by flight_id and current_flight_time
        flights_data_by_flight_id = (
            flights_data_by_flight_id
            .set_index('current_flight_time')
            .groupby('flight_id')
            .apply(lambda x: x)
            .drop('flight_id', axis=1)
            .reset_index(level='flight_id')
            )
        logger.debug(f"there are {flights_data_by_flight_id.shape[0]} rows and {flights_data_by_flight_id.shape[1]} columns in the aggregated data")
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_by_flight_id .head(10))
            
        return flights_data_by_flight_id
    
    #method to floor timestamps and  fill missing timestamps (to ensure time continuity) and remove duplicates
    def floor_and_fill_timestamps(
        self,
        flights_data_by_flight_id: pd.DataFrame, 
        flight_id_column: str, 
        keep: str = 'last'
        ) -> pd.DataFrame:
        """
        Floors timestamps to the nearest minute, removes duplicates within the same flight, 
        and fills missing timestamps at 1-minute intervals for each flight.

        Parameters:
        flights_data_by_flight_id (pd.DataFrame): Input DataFrame with flight tracking data.
        flight_id_column (str): Column name representing the flight identifier.
        keep (str): Whether to keep 'first' or 'last' duplicate rows within each flight. Default is 'last'.
        
        Returns:
        pd.DataFrame: Processed DataFrame with continuous timestamps.
        """
        logger.debug('Flooring timestamps to the nearest minute and filling missing timestamps...')
        # Step 1: Floor the timestamps to the nearest minute
        flights_data_by_flight_id.index = pd.to_datetime(flights_data_by_flight_id.index).floor('T')
        
        # Step 2: Remove duplicates within the same flight (since data is already sorted)
        flights_data_floored_and_filled = (
            flights_data_by_flight_id
            .groupby(flight_id_column)
            .apply(lambda group: group[~group.index
            .duplicated(keep=keep)])
            .reset_index(level=0, drop=True)
            )
        
        # Step 3: Resample to fill any missing timestamps at 1-minute intervals within each flight group
        flights_data_floored_and_filled = (
            flights_data_floored_and_filled
            .groupby(flight_id_column)
            .resample('1T')
            .asfreq()
            .reset_index(level=0, drop=True)
            )
        
        logger.debug(f"there are {flights_data_floored_and_filled.shape[0]} rows and {flights_data_floored_and_filled.shape[1]} columns in the resampled data")
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_floored_and_filled.head(10))
        
        return flights_data_floored_and_filled
        
    #method to fill static columns
    def forward_fill_static_columns(
        self,
        flights_data_floored_and_filled: pd.DataFrame, 
        static_columns: list
        ) -> pd.DataFrame:
        """
        Forward fills static columns for rows with NaN values.
        
        Parameters:
        flights_data_floored_and_filled (pd.DataFrame): Grouped DataFrame by flight_id.
        static_columns (list): List of static columns to forward fill.
        
        Returns:
            pd.DataFrame: Grouped DataFrame with forward filled static columns.
        """
        logger.debug('Forward filling static columns...')
        # Forward fill static columns only for rows with NaN values
        flights_data_floored_and_filled[static_columns] = flights_data_floored_and_filled[static_columns].ffill()
        flights_data_ffill_static_columns = flights_data_floored_and_filled
        
        return flights_data_ffill_static_columns
    
    #method to interpolate missing non-altitude related columns
    def interpolate_non_altitude_related_columns(
        self,
        flights_data_ffill_static_columns: pd.DataFrame, 
        columns: list
        ) -> pd.DataFrame:
        """method to interpolate missing non-altitude related columns in the flights data

        Args:
            flights_data_ffill_static_columns (pd.DataFrame): a pandas dataframe containing the flights data with forward filled static columns
            columns (list): a list of non altitude related columns to interpolate

        Returns:
            pd.DataFrame: a pandas dataframe containing the flights data with interpolated non-altitude related columns
        """
        logger.debug('Interpolating missing non-altitude related columns...')
        # Handle vertical speed separately to avoid interpolation when it should stay zero
        vertical_speed_column = 'vertical_speed'

        # Interpolate all non-altitude related columns except vertical speed
        columns_except_vertical_speed = [col for col in columns if col != vertical_speed_column]
        flights_data_ffill_static_columns[columns_except_vertical_speed] = flights_data_ffill_static_columns[columns_except_vertical_speed].interpolate(method='linear', limit_direction='both')

        # For vertical speed, only interpolate when the previous value is not zero with tqdm.pandas()
        for i in tqdm(range(1, len(flights_data_ffill_static_columns))):
            if flights_data_ffill_static_columns[vertical_speed_column].iloc[i-1] != 0:
                flights_data_ffill_static_columns[vertical_speed_column].iloc[i] = flights_data_ffill_static_columns[vertical_speed_column].iloc[i-1]
            else:
                # Interpolate only if the previous value was not zero
                if pd.isna(flights_data_ffill_static_columns[vertical_speed_column].iloc[i]):
                    flights_data_ffill_static_columns[vertical_speed_column].iloc[i] = flights_data_ffill_static_columns[vertical_speed_column].iloc[i-1]
        
        flights_data_ffill_non_altitude_related_columns = flights_data_ffill_static_columns
        logger.debug(f"there are {flights_data_ffill_non_altitude_related_columns.shape[0]} rows and {flights_data_ffill_non_altitude_related_columns.shape[1]} columns in the interpolated data")
        
        #Display the final preprocessed data
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_ffill_non_altitude_related_columns.head(10))
            
        return flights_data_ffill_non_altitude_related_columns
    
    #method to interpolate missing altitude related columns
    def handle_altitude_and_related_features(
        self,
        flights_data_ffill_non_altitude_related_columns: pd.DataFrame, 
        altitude_related_columns: list
        ) -> pd.DataFrame:
        """
        This function handles missing values in the 'altitude' column and related features
        by interpolating the values based on the vertical speed.
        
        Args:
            group (pd.DataFrame): Grouped DataFrame by flight_id.
            altitude_related_columns (list): List of columns related to altitude.
        
        Returns:
            pd.DataFrame: Grouped DataFrame with interpolated altitude and related features
        """
        logger.debug('Interpolating altitude and related features...')
        # Loop through each row to handle altitude and related features
        for i in tqdm(range(1, len(flights_data_ffill_non_altitude_related_columns))):
            # Only handle rows where altitude is NaN
            if pd.isna(flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i]):
                # Check the vertical speed
                vertical_speed_prev = flights_data_ffill_non_altitude_related_columns['vertical_speed'].iloc[i-1]

                if vertical_speed_prev == 0:
                    # Vertical speed is 0, so altitude remains constant
                    flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i] = flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i-1]
                    # Forward fill all altitude-related features
                    flights_data_ffill_non_altitude_related_columns.loc[flights_data_ffill_non_altitude_related_columns.index[i], altitude_related_columns] = flights_data_ffill_non_altitude_related_columns.loc[flights_data_ffill_non_altitude_related_columns.index[i-1], altitude_related_columns]

                elif vertical_speed_prev > 0:
                    # Ascending, interpolate altitude between previous row and next valid value
                    next_valid_altitude = flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i:].first_valid_index()
                    if next_valid_altitude is not None:
                        # Convert label index to positional index
                        next_valid_pos = flights_data_ffill_non_altitude_related_columns.index.get_loc(next_valid_altitude)
                        
                        # Interpolate altitude with respect to time
                        flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i] = np.interp(
                            flights_data_ffill_non_altitude_related_columns.index[i].timestamp(),
                            [flights_data_ffill_non_altitude_related_columns.index[i-1].timestamp(), flights_data_ffill_non_altitude_related_columns.index[next_valid_pos].timestamp()],
                            [flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i-1], flights_data_ffill_non_altitude_related_columns['altitude'].iloc[next_valid_pos]]
                        )
                        
                        # Interpolate altitude-related features with respect to time
                        for feature in altitude_related_columns:
                            if feature != 'altitude':  # Skip altitude as it has been handled
                                flights_data_ffill_non_altitude_related_columns[feature].iloc[i] = np.interp(
                                    flights_data_ffill_non_altitude_related_columns.index[i].timestamp(),
                                    [flights_data_ffill_non_altitude_related_columns.index[i-1].timestamp(), flights_data_ffill_non_altitude_related_columns.index[next_valid_pos].timestamp()],
                                    [flights_data_ffill_non_altitude_related_columns[feature].iloc[i-1], flights_data_ffill_non_altitude_related_columns[feature].iloc[next_valid_pos]]
                                )

                elif vertical_speed_prev < 0:
                    # Descending, interpolate altitude between previous row and next valid value
                    next_valid_altitude = flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i:].first_valid_index()
                    if next_valid_altitude is not None:
                        # Convert label index to positional index
                        next_valid_pos = flights_data_ffill_non_altitude_related_columns.index.get_loc(next_valid_altitude)
                        
                        # Interpolate altitude with respect to time
                        flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i] = np.interp(
                            flights_data_ffill_non_altitude_related_columns.index[i].timestamp(),
                            [flights_data_ffill_non_altitude_related_columns.index[i-1].timestamp(), flights_data_ffill_non_altitude_related_columns.index[next_valid_pos].timestamp()],
                            [flights_data_ffill_non_altitude_related_columns['altitude'].iloc[i-1], flights_data_ffill_non_altitude_related_columns['altitude'].iloc[next_valid_pos]]
                        )
                        
                        # Interpolate altitude-related features with respect to time
                        for feature in altitude_related_columns:
                            if feature != 'altitude':  # Skip altitude as it has been handled
                                flights_data_ffill_non_altitude_related_columns[feature].iloc[i] = np.interp(
                                    flights_data_ffill_non_altitude_related_columns.index[i].timestamp(),
                                    [flights_data_ffill_non_altitude_related_columns.index[i-1].timestamp(), flights_data_ffill_non_altitude_related_columns.index[next_valid_pos].timestamp()],
                                    [flights_data_ffill_non_altitude_related_columns[feature].iloc[i-1], flights_data_ffill_non_altitude_related_columns[feature].iloc[next_valid_pos]]
                                )
        flights_data_handled_altitude_related_features = flights_data_ffill_non_altitude_related_columns
        
        return flights_data_handled_altitude_related_features
    
    
    #method to drop duplicates by flight_id and flight time
    def drop_duplicates_by_flight_id_and_time(
        self,
        flights_data_handled_altitude_related_feature: pd.DataFrame
        ) -> pd.DataFrame:
        """
        This function drops duplicate rows based on the combination of 'flight_id'
        and the index (which is the timestamp), keeping the last occurrence of each duplicate.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with duplicates dropped.
        """
        logger.debug('Dropping duplicates by flight_id and current_flight_time...')
        if 'flight_id' not in flights_data_handled_altitude_related_feature.columns:
            raise ValueError("'flight_id' column not found in the DataFrame")
        
        # Reset the index to make it a column for duplicate check
        flights_data_handled_altitude_related_feature = flights_data_handled_altitude_related_feature.reset_index()

        # Drop duplicates based on 'flight_id' and the index (now a column), keeping the last occurrence
        flights_drop_duplicates_by_flight_id_and_time = flights_data_handled_altitude_related_feature.drop_duplicates(subset=['flight_id', 'current_flight_time'], keep='last').copy()

        # Set the index back to its original form
        flights_drop_duplicates_by_flight_id_and_time.set_index('current_flight_time', inplace=True)
        logger.debug(f"there are {flights_drop_duplicates_by_flight_id_and_time.shape[0]} rows and {flights_drop_duplicates_by_flight_id_and_time.shape[1]} columns in the data after dropping duplicates")

        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_drop_duplicates_by_flight_id_and_time.head(10))
            
        return flights_drop_duplicates_by_flight_id_and_time

    #Function to convert altitude in meters to flight level
    def _altitude_to_flight_level(
        self,
        altitude_meters : float
        ) -> int:
        """
        Convert altitude in meters to flight level (FL).
            
        Args:
            altitude_meters (float): Altitude in meters.
            
        Returns:
            int: Flight level (FL).
        """
        # Convert meters to feet
        altitude_feet = altitude_meters / 0.3048
            
        # Calculate the flight level
        flight_level = round(altitude_feet / 100)
            
        return flight_level

    #method to compute flight level for the newly added rows in the dataframe (for time continuity)
    def compute_flight_level(
        self,
        df: pd.DataFrame, 
        altitude_column: str = 'altitude'
        ) -> pd.DataFrame:
        """
        Computes the flight level based on altitude and adds it as a new column 'flight_level_computed' in the format 'FLXXX'.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the flight data.
        altitude_column (str): The name of the column containing altitude values in meters.
        
        Returns:
        pd.DataFrame: Updated DataFrame with computed flight levels in the 'FLXXX' format.
        """
        logger.debug('Computing flight level...')
        # Check if the altitude column exists
        if altitude_column not in df.columns:
            raise ValueError(f"Column '{altitude_column}' not found in the DataFrame")
        
        # Compute the flight level and format it as FLXXX
        df['flight_level'] = df[altitude_column].apply(
            lambda alt: f"FL{self._altitude_to_flight_level(alt):03}" if pd.notna(alt) else None
        )
        flights_data_with_adjusted_flight_level = df
        logger.debug(f"there are {flights_data_with_adjusted_flight_level.shape[0]} rows and {flights_data_with_adjusted_flight_level.shape[1]} columns in the data after computing flight level")
        
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_with_adjusted_flight_level.head(10))
            
        return flights_data_with_adjusted_flight_level


    # Function to add 'flight_phase' column and insert it after 'vertical_speed'
    def add_flight_phase(
        self,
        df: pd.DataFrame
        ) -> pd.DataFrame:
        """this function adds a new 'flight_phase' column based on the 'vertical_speed' column

        Args:
            df (pd.DataFrame): a pandas dataframe containing the flights data

        Returns:
            pd.DataFrame: a pandas dataframe containing the flights data with the 'flight_phase' column added
        """
        logger.debug('Adding flight phase column...')
        # Create the 'flight_phase' column based on 'vertical_speed'
        # Fix for the SettingWithCopyWarning using .loc[]
        df.loc[:, 'flight_phase'] = df['vertical_speed'].apply(
            lambda vs: 'cruise' if pd.isna(vs) else ('ascend' if vs > 0 else ('descend' if vs < 0 else 'cruise'))
        )
        
        # Find the index of 'vertical_speed' column
        vertical_speed_index = df.columns.get_loc('vertical_speed')

        # Reorder columns: move 'flight_phase' right after 'vertical_speed'
        cols = df.columns.tolist()
        # Insert 'flight_phase' right after 'vertical_speed'
        cols.insert(vertical_speed_index + 1, cols.pop(cols.index('flight_phase')))
        
        # Return dataframe with reordered columns
        df = df[cols]
        flights_with_flight_phase = df
        logger.debug(f"there are {flights_with_flight_phase.shape[0]} rows and {flights_with_flight_phase.shape[1]} columns in the data after adding flight phase")
        
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_with_flight_phase.head(10))
        #logger.info("Columns after reordering:", df.columns)
        return df
    
    #method to create the target variable
    def create_target_variable(
        self,
        df: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        This function creates the target variable based on the given column.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Column name to use as the target variable.
        
        Returns:
        pd.DataFrame: DataFrame with the target variable added.
        """
        logger.debug('Creating the target variable...')
        # Create the target variable based on the given column
        df['contrail_formation'] = df['prob_contrails_percent'].apply(lambda x: 1 if x >= 50 else 0)
        
        return df   
    
if __name__ == "__main__":
    
    #Instantiate the DataPreprocessing class
    data_preprocessing = DataPreprocessing()
    
    #Retrieve the flights data
    flights_data = data_preprocessing._get_flight_data()
    
    #Aggregate the flights data by flight_id and current_flight_time
    flights_data_by_flight_id = data_preprocessing._aggregate_flights_data(flights_data)
    
    #Floor timestamps and fill missing timestamps
    flights_data_floored_and_filled = data_preprocessing.floor_and_fill_timestamps(flights_data_by_flight_id, flight_id_column='flight_id', keep='last')
    
    #Forward fill static columns
    flights_data_ffill_static_columns = data_preprocessing.forward_fill_static_columns(flights_data_floored_and_filled, static_columns=data_preprocessing.static_columns)
    
    #Interpolate missing non-altitude related columns
    flights_data_ffill_non_altitude_related_columns = data_preprocessing.interpolate_non_altitude_related_columns(flights_data_ffill_static_columns, columns=data_preprocessing.not_altitude_related_columns)
    
    #Interpolate missing altitude related columns
    #flights_data_handled_altitude_related_columns = data_preprocessing.handle_altitude_and_related_features(flights_data_ffill_non_altitude_related_columns, columns=data_preprocessing.altitude_related_columns)
    
    #handdle altitude and related features for all groups
    groups = []  # Store processed groups
    for flight_id in tqdm(flights_data_ffill_non_altitude_related_columns['flight_id'].unique()):
        group = data_preprocessing.handle_altitude_and_related_features(
        flights_data_ffill_non_altitude_related_columns.groupby('flight_id').get_group(flight_id), 
            data_preprocessing.altitude_related_columns
        )
    groups.append(group)  # Add the processed group to the list
             
    logger.debug(f'there are {len(groups)} groups in the list')
        
    # Concatenate all groups at once after processing
    flights_data_handled_altitude_related_features = pd.concat(groups)
    logger.debug(f"there are {flights_data_handled_altitude_related_features.shape[0]} rows and {flights_data_handled_altitude_related_features.shape[1]} columns in the updated data")
    
    #Drop duplicates by flight_id and current_flight_time
    flights_drop_duplicates_by_flight_id_and_time = data_preprocessing.drop_duplicates_by_flight_id_and_time(flights_data_handled_altitude_related_features)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(flights_drop_duplicates_by_flight_id_and_time.head(10))
        
    #Compute flight level
    flights_data_with_adjusted_flight_level = data_preprocessing.compute_flight_level(flights_drop_duplicates_by_flight_id_and_time, altitude_column='altitude')
    
    #Add flight phase
    flights_with_flight_phase = data_preprocessing.add_flight_phase(flights_data_with_adjusted_flight_level)
    
    #Create the target variable
    flights_with_target_variable = data_preprocessing.create_target_variable(flights_with_flight_phase)
    
    #save the preprocessed data to a csv file
    flights_with_flight_phase.to_csv('../files/flights_data_preprocessed.csv')