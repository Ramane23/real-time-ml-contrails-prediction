import pandas as pd
from loguru import logger
import datetime
import warnings
import numpy as np
from typing import List
import tqdm
from openap import prop, FuelFlow, Emission


class FeaturesEngineering:
    """
    Class to engineer features for the contrail predictor using OpenAP aircraft performance model
    """
    
    def __init__(self):
        
        self.openap_supported_aircraft_types = [
            'a19n', 'a20n', 'a21n', 'a318', 'a319', 'a320', 'a321', 'a332', 'a333',
            'a343', 'a359', 'a388', 'b37m', 'b38m', 'b39m', 'b3xm', 'b734', 'b737',
            'b738', 'b739', 'b744', 'b748', 'b752', 'b763', 'b772', 'b773', 'b77w',
            'b788', 'b789', 'c550', 'e145', 'e170', 'e190', 'e195', 'e75l', 'glf6'
        ]
        #this is because openap fuel flow and emission models are not available for all aircraft types (it's a fallback)
        self.aircraft_type_mapping = {
            'a21n': 'a320',   # Map A321neo to A320
            'a20n': 'a320',   # Map A320neo to A320
            'a318': 'a320',   # Map A318 to A320
            'b38m': 'b737',   # Map B737 MAX to B737
            'e190': 'a320',   # Map Embraer E190 to Embraer E170
            'e195': 'a320',   # Map E195 to E170
            'e145': 'a320',   # Map E145 to E170
            'b734': 'b737', # Map B737-400 to B737   
            'b744': 'b737',
            'b789': 'b737',
            'b788': 'b737',
            # Add more mappings if necessary
        }
        pass
    
    # Function to drop unsupported flights and debug unsupported types
    def drop_unsupported_flights(
        self,
        df : pd.DataFrame
        ) -> pd.DataFrame:
        
        """this function drops unsupported flights by the OpenAP aircraft performance model
        Args :
            df (pd.DataFrame): a pandas dataframe containing the flights data
        Returns:
            supported_df (pd.DataFrame): a pandas dataframe containing the supported flights
            
        """
        logger.debug('Dropping flights unsupported by openap...')
        # Ensure aircraft types are all lowercase and stripped of extra spaces
        df['aircraft_type'] = df['aircraft_type'].str.lower().str.strip()

        # Identify any aircraft types not in the supported list
        unsupported_types = df[~df['aircraft_type'].isin(self.openap_supported_aircraft_types)]['aircraft_type'].unique()
        
        if len(unsupported_types) > 0:
            logger.info(f"Unsupported aircraft types found: {unsupported_types}")

        # Filter out rows with aircraft types not in the supported list
        flights_data_without_unsupported_aircrafts = df[df['aircraft_type'].isin(self.openap_supported_aircraft_types)]
        logger.debug(f"there are {flights_data_without_unsupported_aircrafts.shape[0]} rows and {flights_data_without_unsupported_aircrafts.shape[1]} columns in the data after dropping unsupported flights")
        
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_without_unsupported_aircrafts.head(10))
        
        return flights_data_without_unsupported_aircrafts
    
    #method to add engine type column from OpenAP
    def add_engine_type_to_df(
        self,
        df : pd.DataFrame 
        ) -> pd.DataFrame:
        """
        This function takes a dataframe containing flight information and adds an 'engine_type' column
        based on the aircraft types present. 
        Args:
            df (pd.DataFrame): Input dataframe with an 'aircraft_type' column.

        Returns:
            pd.DataFrame: Dataframe with the added 'engine_type' column.
        """
        logger.debug('adding engine type to the dataframe...')
        # List of unique aircraft types from the dataframe
        aircraft_types = df['aircraft_type'].unique()

        # Create a dictionary to map aircraft types to engine types
        aircraft_to_engine = {}

        # Loop through each aircraft type and retrieve engine information
        for aircraft_type in aircraft_types:
            try:
                aircraft_data = prop.aircraft(aircraft_type.upper())  # Ensure aircraft type is uppercase
                engine_info = aircraft_data['engine']['default']  # Get the default engine type
                aircraft_to_engine[aircraft_type.lower()] = engine_info  # Store in lowercase for consistency
            except KeyError:
                print(f"Engine information not found for aircraft type: {aircraft_type}")

        # Apply the mapping to the dataframe
        df['engine_type'] = df['aircraft_type'].map(aircraft_to_engine)

        # Reorder the columns to place 'engine_type' right after 'aircraft_type'
        cols = df.columns.tolist()  # Get all column names as a list
        aircraft_type_index = cols.index('aircraft_type')  # Find the index of 'aircraft_type'
        
        # Reorder columns to place 'engine_type' after 'aircraft_type'
        cols.insert(aircraft_type_index + 1, cols.pop(cols.index('engine_type')))
        flights_data_with_engine_type = df[cols]  # Reassign the dataframe with reordered columns
        logger.debug(f"there are {flights_data_with_engine_type.shape[0]} rows and {flights_data_with_engine_type.shape[1]} columns in the data after adding engine type")
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_with_engine_type.head(10))
        
        return flights_data_with_engine_type

    def add_engine_characteristics_to_df(
        self,
        filtered_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        This function takes a dataframe containing flight information with an 'engine_type' column, 
        retrieves engine characteristics relevant to contrail formation, and adds these as new columns.

        Args:
            df (pd.DataFrame): Input dataframe with an 'engine_type' column.

        Returns:
            pd.DataFrame: Dataframe with added engine characteristic columns.
        """
        logger.debug('adding engine characteristics to the dataframe...')
        # Function to retrieve engine characteristics relevant to contrail formation
        def get_engine_characteristics(engine_type):
            try:
                # Get engine data from OpenAP
                engine = prop.engine(engine_type)
                # Extract relevant characteristics
                return {
                    'bypass_ratio': engine.get('bpr', None),
                    'cruise_thrust': engine.get('cruise_thrust', None),
                    'cruise_sfc': engine.get('cruise_sfc', None),  # Specific Fuel Consumption at cruise
                    'ei_nox_to': engine.get('ei_nox_to', None),
                    'ei_nox_co': engine.get('ei_nox_co', None),
                    'ei_nox_app': engine.get('ei_nox_app', None),
                    'ei_nox_idl': engine.get('ei_nox_idl', None),
                    'ei_co_to': engine.get('ei_co_to', None),
                    'ei_co_co': engine.get('ei_co_co', None),
                    'ei_co_app': engine.get('ei_co_app', None),
                    'ei_co_idl': engine.get('ei_co_idl', None)
                }
            except KeyError:
                print(f"Engine data not found for engine type: {engine_type}")
                return {}

        # Step 1: Get the engine characteristics and expand them into separate columns
        engine_characteristics_df = filtered_df['engine_type'].apply(get_engine_characteristics).apply(pd.Series)

        # Step 2: Insert the new engine characteristics columns right after 'engine_type'
        engine_type_index = filtered_df.columns.get_loc('engine_type') + 1  # Find the index after 'engine_type'

        # Step 3: Concatenate the original dataframe and the engine characteristics
        flights_data_with_engine_characteristics = pd.concat([filtered_df.iloc[:, :engine_type_index], engine_characteristics_df, filtered_df.iloc[:, engine_type_index:]], axis=1)
        logger.debug(f"there are {flights_data_with_engine_characteristics.shape[0]} rows and {flights_data_with_engine_characteristics.shape[1]} columns in the data after adding engine characteristics")
        
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_with_engine_characteristics.head(10))
            
        return flights_data_with_engine_characteristics
    
    #method to add temporal features to the dataframe
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create temporal features, group flight data by datetime index and route,
        add rolling window features, and aggregate relevant statistics for each group.

        Args:
        df (pd.DataFrame): The input dataframe with flight data and datetime index.

        Returns:
        pd.DataFrame: Transformed and grouped dataframe with aggregated features and rolling windows.
        """
        logger.debug('adding temporal features to the dataframe...')
        # Step 1: Ensure datetime index is set correctly and create temporal features
        df.index = pd.to_datetime(df.index)  # Ensure the index is datetime

        # Create temporal features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if x < 6 or x >= 18 else 0)

        # Step 2: Calculate Rolling Features for Selected Columns
        rolling_features = ['altitude', 'latitude', 'longitude', 'fuel_flow', 'temperature_c', 'contrails_formation']
        
        for feature in rolling_features:
            # Rolling windows over 1 hour
            df[f'{feature}_rolling_mean_1H'] = df.groupby('route')[feature].transform(lambda x: x.rolling('60min').mean())
            df[f'{feature}_rolling_sum_1H'] = df.groupby('route')[feature].transform(lambda x: x.rolling('60min').sum())
            df[f'{feature}_rolling_std_1H'] = df.groupby('route')[feature].transform(lambda x: x.rolling('60min').std())

            # Rolling windows over 1 day
            df[f'{feature}_rolling_mean_1D'] = df.groupby('route')[feature].transform(lambda x: x.rolling('24H').mean())
            df[f'{feature}_rolling_sum_1D'] = df.groupby('route')[feature].transform(lambda x: x.rolling('24H').sum())
            df[f'{feature}_rolling_std_1D'] = df.groupby('route')[feature].transform(lambda x: x.rolling('24H').std())

        # Step 3: Group by 'current_flight_time' (datetime index) and 'route'
        grouped_df = df.groupby([df.index, 'route']).agg(
            {
                'departure_airport_lat': 'first',  # Using 'first' for consistent route-specific data
                'departure_airport_long': 'first',
                'arrival_airport_lat': 'first',
                'arrival_airport_long': 'first',
                'altitude': 'mean',  # Aggregating altitude data with mean
                'latitude': 'mean',
                'longitude': 'mean',
                'vertical_speed': 'mean',
                'mach_number': 'mean',
                'fuel_flow': 'mean',
                'direction': 'mean',
                'flight_phase': 'first',  # Using 'first' as flight_phase should be consistent
                'temperature_c': 'mean',
                'relative_humidity_percent': 'mean',
                'specific_humidity_kg_kg': 'mean',
                'pressure_hpa': 'mean',
                'wind_speed_u_ms': 'mean',
                'wind_speed_v_ms': 'mean',
                'total_cloud_cover_octas': 'mean',
                'global_radiation_w_m2': 'mean',
                'prob_contrails_percent': 'mean',
                'contrails_formation': 'max',  # Use 'max' to detect if any contrail formation occurred
                'hour_of_day': 'first',  # Retain the first value for time features
                'day_of_week': 'first',
                'day_of_month': 'first',
                'week_of_year': 'first',
                'month': 'first',
                'quarter': 'first',
                'year': 'first',
                'is_weekend': 'first',
                'is_night': 'first',
                # Include aggregated rolling window features
                **{f'{feature}_rolling_mean_1H': 'mean' for feature in rolling_features},
                **{f'{feature}_rolling_sum_1H': 'sum' for feature in rolling_features},
                **{f'{feature}_rolling_std_1H': 'mean' for feature in rolling_features},
                **{f'{feature}_rolling_mean_1D': 'mean' for feature in rolling_features},
                **{f'{feature}_rolling_sum_1D': 'sum' for feature in rolling_features},
                **{f'{feature}_rolling_std_1D': 'mean' for feature in rolling_features}
            }
        ).reset_index(level='route')

        # Rename the index column to 'current_flight_time' after grouping
        grouped_df.rename(columns={'index': 'current_flight_time'}, inplace=True)
        flights_data_with_temporal_features = grouped_df
        logger.debug(f"there are {flights_data_with_temporal_features.shape[0]} rows and {flights_data_with_temporal_features.shape[1]} columns in the data after adding temporal features")
        # Display the grouped dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_with_temporal_features.head(10))

        return flights_data_with_temporal_features

if __name__ == '__main__':
    # Create an instance of the FeaturesEngineering class
    features_engineering = FeaturesEngineering()

    # Load the historical flights data
    flights_data = pd.read_csv('../files/flights_data_preprocessed.csv')
    
    # Drop unsupported aircraft types
    #Drop unsupported flights
    supported_flights = features_engineering.drop_unsupported_flights(flights_data)
    
    #add engine type to the dataframe
    flights_data_with_engine_type = features_engineering.add_engine_type_to_df(supported_flights)
    
    #add engine characteristics to the dataframe
    flights_data_with_engine_characteristics = features_engineering.add_engine_characteristics_to_df(flights_data_with_engine_type)

    #add temporal features to the dataframe
    flights_data_with_temporal_features : pd.DataFrame = features_engineering.add_temporal_features(flights_data_with_engine_characteristics)
    
    #save the dataframe to a csv file
    flights_data_with_temporal_features.to_csv('../files/flights_data_with_features.csv')