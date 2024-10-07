import pandas as pd
from loguru import logger
import warnings
from typing import Tuple
from openap import prop, FuelFlow, Emission
from sklearn.base import BaseEstimator, TransformerMixin

#ignore warnings
warnings.filterwarnings('ignore')

class FeaturesEngineering(BaseEstimator, TransformerMixin):
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
        
        #firt we ensure that the current flight time is the index and ensure it is a datetime object
        #first check if the index is set to the current flight time
        if df.index.name != 'current_flight_time':
            df.set_index('current_flight_time', inplace=True)
            
        #ensure the index is a datetime object
        df.index = pd.to_datetime(df.index)
        #breakpoint()
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
        #firt we ensure that the current flight time is the index and ensure it is a datetime object
        #df.set_index('current_flight_time', inplace=True)
        #breakpoint()
        
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
        #breakpoint()
        return flights_data_with_engine_type
    
    def get_engine_characteristics(self,engine_type):
        try:
            engine = prop.engine(engine_type)
            return {
                'engine_type': engine_type,
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
            return {'engine_type': engine_type}

    #method to add engine characteristics to the dataframe
    def add_engine_characteristics_to_df(
        self,
        df : pd.DataFrame
        ) -> pd.DataFrame:
        """
        This function takes a dataframe containing flight information and adds engine characteristics columns
        based on the engine types present.
        
        Args:
            df (pd.DataFrame): Input dataframe with an 'engine_type' column.
            
        Returns:
            pd.DataFrame: Dataframe with the added engine characteristics columns.
        """
        logger.debug('adding engine characteristics to the dataframe...')
        #breakpoint()
        #ensure the index is the current flight time
        #df.set_index('current_flight_time', inplace=True)
        
        #let's keep only the first 10000 rows for nows
        #df = df.head(400000)
        # Get unique engine types from the dataframe
        engine_types = df['engine_type'].unique()
        
        # Create a list to store engine characteristics
        engine_characteristics = []
        
        # Loop through each engine type and retrieve engine characteristics
        for engine_type in engine_types:
            engine_data = self.get_engine_characteristics(engine_type)
            engine_characteristics.append(engine_data)
        
        # Create a dataframe from the list of engine characteristics
        engine_characteristics_df = pd.DataFrame(engine_characteristics)
        
        # Step 1: Reset the index to preserve 'current_flight_time' as a column
        df_reset = df.reset_index()

        # Step 2: Merge the dataframes on 'engine_type'
        flights_data_with_engine_characteristics = pd.merge(
            df_reset, engine_characteristics_df, on='engine_type', how='left'
        )

        # Step 3: Restore 'current_flight_time' as the index
        flights_data_with_engine_characteristics.set_index('current_flight_time', inplace=True)
        #sort the index
        flights_data_with_engine_characteristics.sort_index(inplace=True)
        #breakpoint()
        #breakpoint()
        # Reorder the columns to place engine characteristics after 'engine_type'
        cols = flights_data_with_engine_characteristics.columns.tolist()  # Get all column names as a list
        engine_type_index = cols.index('engine_type')  # Find the index of 'engine_type'
        
        # Reorder columns to place engine characteristics after 'engine_type'
        cols = cols[:engine_type_index + 1] + cols[-11:] + cols[engine_type_index + 1:-11]
        flights_data_with_engine_characteristics = flights_data_with_engine_characteristics[cols]  # Reassign the dataframe with reordered columns
        logger.debug(f"there are {flights_data_with_engine_characteristics.shape[0]} rows and {flights_data_with_engine_characteristics.shape[1]} columns in the data after adding engine characteristics")
        #breakpoint()
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(flights_data_with_engine_characteristics.head(10))
            
        return flights_data_with_engine_characteristics
    
    #method to calculate fuel flow and emissions for the dataframe
    def calculate_fuel_and_emissions(
        self,
        df: pd.DataFrame, 
        aircraft_type_mapping: dict
        ) -> pd.DataFrame:
        """
        Calculate fuel flow and emissions for a given dataframe using aircraft-specific models.
        
        Args:
            df (pd.DataFrame): Dataframe containing flight data.
            aircraft_type_mapping (dict): Dictionary mapping aircraft types to standardized names.
        
        Returns:
            pd.DataFrame: Dataframe with calculated fuel flow and emissions columns.
        """
        logger.debug('calculating fuel flow and emissions for the dataframe...')
        # Create a new column for mapped aircraft types without modifying the original 'aircraft_type'
        df['mapped_aircraft_type'] = df['aircraft_type'].apply(lambda x: aircraft_type_mapping.get(x, x))

        for ac_type in df['mapped_aircraft_type'].unique():
            try:
                # Initialize fuel flow and emission models for the mapped aircraft type
                fuelflow = FuelFlow(ac=ac_type)
                emission = Emission(ac=ac_type)

                # Apply calculations for rows where the 'mapped_aircraft_type' matches
                mask = df['mapped_aircraft_type'] == ac_type

                # Calculate fuel flow for each point in the dataframe
                df.loc[mask, 'fuel_flow'] = df[mask].apply(
                    lambda row: fuelflow.enroute(
                        mass=row['aircraft_mtow_kg'],  # Using MTOW for max takeoff conditions
                        tas=row['true_airspeed_ms'] * 1.94384,  # Convert m/s to knots
                        alt=row['altitude'] * 3.28084,          # Convert meters to feet
                        vs=row['vertical_speed'] * 196.85       # Convert m/s to feet/min
                    ), axis=1)

                # Calculate emissions based on the fuel flow
                df.loc[mask, 'co2_flow'] = df.loc[mask, 'fuel_flow'].apply(emission.co2)
                df.loc[mask, 'h2o_flow'] = df.loc[mask, 'fuel_flow'].apply(emission.h2o)
                df.loc[mask, 'nox_flow'] = df[mask].apply(
                    lambda row: emission.nox(
                        row['fuel_flow'], 
                        tas=row['true_airspeed_ms'] * 1.94384,  # Convert m/s to knots
                        alt=row['altitude'] * 3.28084           # Convert meters to feet
                    ), axis=1)
                df.loc[mask, 'co_flow'] = df[mask].apply(
                    lambda row: emission.co(
                        row['fuel_flow'], 
                        tas=row['true_airspeed_ms'] * 1.94384,  # Convert m/s to knots
                        alt=row['altitude'] * 3.28084           # Convert meters to feet
                    ), axis=1)
                df.loc[mask, 'hc_flow'] = df[mask].apply(
                    lambda row: emission.hc(
                        row['fuel_flow'], 
                        tas=row['true_airspeed_ms'] * 1.94384,  # Convert m/s to knots
                        alt=row['altitude'] * 3.28084           # Convert meters to feet
                    ), axis=1)
                
                # Calculate soot flow and add it to the dataframe
                df.loc[mask, 'soot_flow'] = df.loc[mask, 'fuel_flow'].apply(emission.soot)

            except Exception as e:
                # Log the error and skip this aircraft type if any issue arises
                print(f"Skipping {ac_type} due to missing drag polar or kinematic model: {e}")
                continue

        # Drop the temporary `mapped_aircraft_type` column after use
        df.drop(columns=['mapped_aircraft_type'], inplace=True)
        logger.debug(f"there are {df.shape[0]} rows and {df.shape[1]} columns in the data after calculating fuel flow and emissions")
        
        # Display the dataframe with fuel flow and emissions
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(df.head(10))
            
        return df

    #method to reorder fuel and emission columns in the dataframe
    def reorder_fuel_emission_columns(
        self,
        df: pd.DataFrame, 
        flight_phase_col='flight_phase'
        ) -> pd.DataFrame:
        """
        Reorder fuel and emission columns in the dataframe to be right after a specified column.
        
        Args:
            df (pd.DataFrame): Dataframe containing fuel and emission data.
            flight_phase_col (str): Column name to place fuel and emissions columns after.
        
        Returns:
            pd.DataFrame: Reordered dataframe with fuel and emissions columns placed appropriately.
        """
        logger.debug('reordering fuel and emission columns in the dataframe...')
        # Set aircraft_type to uppercase again
        df['aircraft_type'] = df['aircraft_type'].str.upper()

        # List of fuel and emission columns to place after the specified column
        fuel_emission_cols = ['fuel_flow', 'co2_flow', 'h2o_flow', 'nox_flow', 'co_flow', 'hc_flow', 'soot_flow']

        # Remove any existing duplicate columns in the dataframe (if any)
        df = df.loc[:, ~df.columns.duplicated()]

        # Reorder columns to place fuel and emission data after the specified column
        columns = df.columns.tolist()
        flight_phase_index = columns.index(flight_phase_col)

        # Create the new order of columns
        new_order = columns[:flight_phase_index + 1] + [col for col in fuel_emission_cols if col in df.columns] + columns[flight_phase_index + 1:]

        # Reorder the dataframe with the new columns order
        df = df[new_order]

        # Remove any pre-existing duplicate columns in the dataframe
        df = df.loc[:, ~df.columns.duplicated()]

        logger.info(f'There are {df.shape[0]} rows and {df.shape[1]} columns in the updated data.')
        
        # Display the reordered dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(df.head(10))

        return df
    
    #TODO: dropping highly correlated features
    
    #method to add temporal features to the dataframe
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create temporal features, group flight data by datetime index and route,
        add rolling window features, and aggregate relevant statistics for each group.

        Args:
        df (pd.DataFrame): The input dataframe with flight data and datetime index.

        Returns:
        pd.DataFrame: Transformed and grouped dataframe with aggregated features and rolling windows.
        """
        #reduce the size of the dataset to avoi memory error*
        #df = df.head(100000)
        
        logger.debug('adding temporal features to the dataframe...')

        # Step 1: Ensure datetime index is set correctly and create temporal features
        df.index = pd.to_datetime(df.index)  # Ensure the index is datetime
        logger.debug(f"there are {df.shape[0]} rows and {df.shape[1]} columns in the data before adding temporal features")
       
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

        # Define rolling features
        rolling_features = [
            'altitude', 'latitude', 'longitude', 'direction', 'fuel_flow', 'co2_flow',
            'h2o_flow', 'nox_flow', 'co_flow', 'hc_flow', 'soot_flow', 'temperature_c',
            'relative_humidity_percent', 'specific_humidity_kg_kg', 'contrail_formation',
            'pressure_hpa', 'wind_speed_u_ms', 'wind_speed_v_ms', 'wind_speed_ms',
            'total_cloud_cover_octas', 'global_radiation_w_m2', 'prob_contrails_percent'
        ]

        # Step 2: Compute all rolling features in a separate dataframe
        rolling_features_df = pd.DataFrame(index=df.index)  # Create a new dataframe to store rolling features

        # Group by route and compute rolling features for each column
        grouped = df.groupby('route')
        for feature in rolling_features:
            # Rolling windows over 1 hour
            rolling_features_df[f'{feature}_rolling_mean_1H'] = grouped[feature].transform(lambda x: x.rolling('60min').mean())
            rolling_features_df[f'{feature}_rolling_sum_1H'] = grouped[feature].transform(lambda x: x.rolling('60min').sum())
            rolling_features_df[f'{feature}_rolling_std_1H'] = grouped[feature].transform(lambda x: x.rolling('60min').std())

            # Rolling windows over 1 day
            rolling_features_df[f'{feature}_rolling_mean_1D'] = grouped[feature].transform(lambda x: x.rolling('24H').mean())
            rolling_features_df[f'{feature}_rolling_sum_1D'] = grouped[feature].transform(lambda x: x.rolling('24H').sum())
            rolling_features_df[f'{feature}_rolling_std_1D'] = grouped[feature].transform(lambda x: x.rolling('24H').std())

        # Step 3: Forward-fill missing values in the rolling features to prevent row loss
        rolling_features_df.ffill(inplace=True)  # Forward-fill missing values in all rolling features

        # Step 4: Concatenate the new rolling features back to the original dataframe
        df = pd.concat([df, rolling_features_df], axis=1)

        # Step 5: Forward-fill missing values in the entire dataframe
        df.ffill(inplace=True)

        # Step 6: Drop rows where all rolling features are still NaN
        rolling_columns = [col for col in df.columns if 'rolling' in col]
        df.dropna(subset=rolling_columns, how='all', inplace=True)

        # Step 7: Group by 'current_flight_time' (datetime index) and 'route' for final aggregation
        grouped_df = df.groupby([df.index, 'route']).agg(
            {
                'departure_airport_lat': 'first',
                'departure_airport_long': 'first',
                'arrival_airport_lat': 'first',
                'arrival_airport_long': 'first',
                'altitude': 'mean',
                'latitude': 'mean',
                'longitude': 'mean',
                'true_airspeed_ms': 'mean',
                'horizontal_speed': 'mean',
                'vertical_speed': 'mean',
                'mach_number': 'mean',
                'co2_flow': 'mean',
                'h2o_flow': 'mean',
                'nox_flow': 'mean',
                'co_flow': 'mean',
                'hc_flow': 'mean',
                'soot_flow': 'mean',
                'fuel_flow': 'mean',
                'direction': 'mean',
                'flight_phase': 'first',
                'temperature_c': 'mean',
                'relative_humidity_percent': 'mean',
                'specific_humidity_kg_kg': 'mean',
                'pressure_hpa': 'mean',
                'wind_speed_u_ms': 'mean',
                'wind_speed_v_ms': 'mean',
                'wind_speed_ms': 'mean',
                'total_cloud_cover_octas': 'mean',
                'global_radiation_w_m2': 'mean',
                'contrail_formation': 'mean',
                'prob_contrails_percent': 'mean',
                'hour_of_day': 'first',
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

        # Drop the first row if it has missing values
        grouped_df = grouped_df.iloc[1:]
        
        #force 30% of the data to be contrail formation
        grouped_df.loc[grouped_df.sample(frac=0.05, replace=True).index, 'contrail_formation'] = 1 #this is to force 30% of the data to be contrail formation because the data is highly imbalanced
        
        # Final dataframe with temporal features
        logger.debug(f"there are {grouped_df.shape[0]} rows and {grouped_df.shape[1]} columns in the data after adding temporal features")
        
        return grouped_df


    #method to downsample the majority class (no contrail formation) in the dataset to match the minority class (contrail formation)
    #def time_aware_downsample(
        #self,
        #df: pd.DataFrame, 
        #target_column='contrail_formation', 
        #time_block='H'
        #) -> pd.DataFrame:
        #"""
        #Function to perform time-aware downsampling on an imbalanced dataset.
        
        #Args:
            #grouped_df (pd.DataFrame): Original dataframe with a datetime index.
            #target_column (str): Column name for the target variable (default is 'contrail_formation').
            #time_block (str): The time block to group by (e.g., 'H' for hourly, 'D' for daily).
            
        #Returns:
            #pd.DataFrame: Downsampled dataframe with balanced class distribution.
        #"""
        #logger.debug('Performing time-aware downsampling...')
        
        # Step 1: Convert index to datetime if not already done
        #df.index = pd.to_datetime(df.index)

        # Step 2: Separate majority and minority classes
        #df_majority = df[df[target_column] == 0].copy()
        #df_minority = df[df[target_column] == 1].copy()

        # Step 3: Print initial class sizes
        #logger.info("Initial class sizes:")
        #logger.info(f"Majority class (0): {df_majority.shape[0]}")
        #logger.info(f"Minority class (1): {df_minority.shape[0]}")

        # If the minority class is empty, skip downsampling and return the original dataframe
        #if df_minority.shape[0] == 0:
            #logger.warning("Minority class is empty. Skipping downsampling.")
            #return df

        # Step 4: Create a time block column
        #df_majority['hour_block'] = df_majority.index.floor(time_block)
        #df_minority['hour_block'] = df_minority.index.floor(time_block)

        # Step 5: Filter majority class to include only the blocks present in the minority class
        #minority_blocks = df_minority['hour_block'].unique()
        #df_majority_filtered = df_majority[df_majority['hour_block'].isin(minority_blocks)]

        # Step 6: Downsample the filtered majority class to match the minority class size
        #df_majority_downsampled = resample(
            #df_majority_filtered, 
            #replace=False, 
            #n_samples=df_minority.shape[0],  # Match minority class size
            #random_state=42
        #)

        # Step 7: Print downsampled class sizes
        #logger.info(f"Filtered majority class size (before downsampling): {df_majority_filtered.shape[0]}")
        #logger.info(f"Downsampled majority class size (after downsampling): {df_majority_downsampled.shape[0]}")

        # Step 8: Combine the downsampled majority class with the minority class
        #df_downsampled = pd.concat([df_majority_downsampled, df_minority])

        # Step 9: Drop the temporary `hour_block` column
        #df_downsampled.drop(columns=['hour_block'], inplace=True)

        # Step 10: Print the new class proportions
        #class_proportions = df_downsampled[target_column].value_counts(normalize=True)
        #logger.info(f"New class proportions after downsampling: {class_proportions}")
        #breakpoint()
        #return df_downsampled

    # Function to apply the feature engineering steps
    def apply_feature_engineering(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'contrail_formation'
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply all feature engineering steps to the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe with flight data.
            target_column (str): The target column to predict, default is 'contrail_formation'.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Transformed dataframe with engineered features and target series.
        """
        logger.debug('Applying feature engineering steps...')
        #breakpoint()
        # Step 1: Drop unsupported flights
        df = self.drop_unsupported_flights(df)
        #log the cas where the dataframe is empty
        if df.empty:
            logger.warning(f"No data available for the inputed route, after dropping flights with unsupported aircraft types.")
            return df
        
        # Step 2: Add engine type to the dataframe
        df = self.add_engine_type_to_df(df)
        #log the case where the dataframe is empty
        if df.empty:
            logger.warning(f"No data available for the inputed route, after adding engine type.")
            return df
        
        # Step 3: Add engine characteristics to the dataframe
        df = self.add_engine_characteristics_to_df(df)
        #log the case where the dataframe is empty
        if df.empty:
            logger.warning(f"No data available for the inputed route, after adding engine characteristics.")
            return df
        
        # Step 4: Calculate fuel flow and emissions for the dataframe
        df = self.calculate_fuel_and_emissions(df, self.aircraft_type_mapping)
        #log the case where the dataframe is empty
        if df.empty:
            logger.warning(f"No data available for the inputed route, after calculating fuel flow and emissions.")
            return df
        
        # Step 5: Reorder fuel and emission columns in the dataframe
        df = self.reorder_fuel_emission_columns(df)
        #log the case where the dataframe is empty
        if df.empty:
            logger.warning(f"No data available for the inputed route, after reordering fuel and emission columns.")
            return df
        
        # Step 6: Add temporal features to the dataframe
        df = self.add_temporal_features(df)
        #log the case where the dataframe is empty
        if df.empty:
            logger.warning(f"No data available for the inputed route, after adding temporal features.")
            return df
        
        # Step 7: Drop the prob contrails percent column to avoid data leakage
        logger.debug('Dropping the prob_contrails_percent column to avoid data leakage...')
        df.drop(columns=['prob_contrails_percent'], inplace=True)
        
        #Separate the target column from the features
        y_transformed = df[target_column].astype(int)
        X_transformed = df.drop(columns=[target_column])
        #breakpoint()
        
        return X_transformed, y_transformed
    
if __name__ == '__main__':
    # Create an instance of the FeaturesEngineering class
    features_engineering = FeaturesEngineering()

    # Load the historical flights data
    flights_data = pd.read_csv('./files/flights_data_preprocessed.csv')
    
    flights_data.set_index('current_flight_time', inplace=True)
    
    flights_data : pd.DataFrame = features_engineering.apply_feature_engineering(flights_data)
    
    #Drop unsupported aircraft types
    #supported_flights = features_engineering.drop_unsupported_flights(flights_data)
    
    #add engine type to the dataframe
    #flights_data_with_engine_type = features_engineering.add_engine_type_to_df(supported_flights)
    
    #add engine characteristics to the dataframe
    #flights_data_with_engine_characteristics = features_engineering.add_engine_characteristics_to_df(flights_data_with_engine_type)

    #calculate fuel flow and emissions for the dataframe
    #flights_data_with_fuel_emissions = features_engineering.calculate_fuel_and_emissions(flights_data_with_engine_characteristics, features_engineering.aircraft_type_mapping)
    
    #reorder fuel and emission columns in the dataframe
    #flights_data_with_reordered_fuel_emissions = features_engineering.reorder_fuel_emission_columns(flights_data_with_fuel_emissions)
    
    #add temporal features to the dataframe
    #flights_data_with_temporal_features : pd.DataFrame = features_engineering.add_temporal_features(flights_data_with_engine_characteristics)
    
    #downsample the majority class (no contrail formation) in the dataset to match the minority class (contrail formation)
    #flights_data_downsampled = features_engineering.time_aware_downsample(flights_data_with_temporal_features)
    
    #save the dataframe to a csv file
    flights_data.to_csv('./files/flights_data_with_features.csv')