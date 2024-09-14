from loguru import logger
from typing import Dict
from datetime import datetime, timezone
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
import xarray as xr
import io
import numpy as np
from netCDF4 import Dataset
from src.gfs_api.flight import Flight, FlightWeather  # Assuming you have a module for Flight and FlightWeather classes

class AddFlightsWeather:
    def __init__(self) -> None:
        """
        Initialize the AddFlightsWeather class.
        This class fetches weather data and integrates it into the flight object.
        """
        logger.info("AddFlightsWeather class initialized.")
    
    def altitude_to_pressure(
        self,
        altitude_meters : float,
        ) -> float:
        # Constants for the barometric formula
        P0 = 1013.25  # Standard pressure at sea level (hPa)
        T0 = 288.15  # Standard temperature at sea level (K)
        L = 0.0065  # Temperature lapse rate (K/m)
        R = 8.3144598  # Universal gas constant (J/(mol·K))
        g = 9.80665  # Acceleration due to gravity (m/s²)
        M = 0.0289644  # Molar mass of Earth's air (kg/mol)
        
        # Convert altitude in meters to pressure in hPa
        pressure_hPa = P0 * (1 - (L * altitude_meters) / T0) ** (g * M / (R * L))
        
        return round(pressure_hPa)

    def fetch_weather_data(self, latitude: float, longitude: float, altitude: float, flight_time: datetime) -> Dict:
        logger.info(f"Fetching weather data for lat: {latitude}, lon: {longitude}, altitude: {altitude} meters at {flight_time} UTC.")
        
        pressure_level = self.altitude_to_pressure(altitude)
        logger.debug(f"Using pressure level: {pressure_level} hPa for altitude: {altitude} meters")

        # Access the GFS forecast data from Unidata's TDS catalog
        cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml')
        
        available_datasets = list(cat.datasets.values())  # Retrieve all datasets in the catalog, assuming they are sorted (latest first)
        
        # List of required variables
        required_variables = [
            'Temperature_isobaric',
            'u-component_of_wind_isobaric',
            'v-component_of_wind_isobaric',
            'Geopotential_height_isobaric',
            'Relative_humidity_isobaric',
            'Specific_humidity_isobaric',
            'Total_cloud_cover_entire_atmosphere',
            'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average'
        ]
        
        # Initialize the weather data dictionary with None for each variable
        weather_data = {var: None for var in [
            "temperature_C", 
            "wind_speed_u_ms", 
            "wind_speed_v_ms", 
            "geopotential_height_m", 
            "relative_humidity_percent", 
            "specific_humidity_kg_kg", 
            "total_cloud_cover_octas", 
            "downward_shortwave_radiation_W_m2"
        ]}

        # Track which variables have been found
        found_variables = set()
        
        for dataset in available_datasets:
            # If all variables have been found, break the loop
            if all(value is not None for value in weather_data.values()):
                break

            ncss = NCSS(dataset.access_urls['NetcdfSubset'])
            query = ncss.query()
            query.lonlat_point(longitude, latitude)
            query.time(flight_time)
            query.vertical_level(pressure_level)
            query.variables(*required_variables)  # Add the required variables to the query
            query.accept('netcdf')

            logger.debug(f"Attempting to fetch data from dataset: {dataset.name}")
            
            try:
                data = ncss.get_data(query)  # Try to fetch data
                available_variables = list(data.variables.keys())
                
                logger.debug(f"Available variables in {dataset.name}: {available_variables}")
                
                # Fetch variables only if they haven't been found yet
                if 'Temperature_isobaric' in available_variables and 'temperature_C' not in found_variables:
                    temperature = data.variables['Temperature_isobaric'][:].data.item() - 273.15  # Convert from Kelvin to Celsius
                    weather_data["temperature_C"] = temperature
                    found_variables.add('temperature_C')

                if 'u-component_of_wind_isobaric' in available_variables and 'wind_speed_u_ms' not in found_variables:
                    u_wind = data.variables['u-component_of_wind_isobaric'][:].data.item()
                    weather_data["wind_speed_u_ms"] = u_wind
                    found_variables.add('wind_speed_u_ms')

                if 'v-component_of_wind_isobaric' in available_variables and 'wind_speed_v_ms' not in found_variables:
                    v_wind = data.variables['v-component_of_wind_isobaric'][:].data.item()
                    weather_data["wind_speed_v_ms"] = v_wind
                    found_variables.add('wind_speed_v_ms')

                if 'Geopotential_height_isobaric' in available_variables and 'geopotential_height_m' not in found_variables:
                    geopotential_height = data.variables['Geopotential_height_isobaric'][:].data.item()
                    weather_data["geopotential_height_m"] = geopotential_height
                    found_variables.add('geopotential_height_m')

                if 'Relative_humidity_isobaric' in available_variables and 'relative_humidity_percent' not in found_variables:
                    relative_humidity = data.variables['Relative_humidity_isobaric'][:].data.item()
                    weather_data["relative_humidity_percent"] = relative_humidity
                    found_variables.add('relative_humidity_percent')

                if 'Specific_humidity_isobaric' in available_variables and 'specific_humidity_kg_kg' not in found_variables:
                    specific_humidity = data.variables['Specific_humidity_isobaric'][:].data.item()
                    weather_data["specific_humidity_kg_kg"] = specific_humidity
                    found_variables.add('specific_humidity_kg_kg')

                if 'Total_cloud_cover_entire_atmosphere' in available_variables and 'total_cloud_cover_octas' not in found_variables:
                    total_cloud_cover = data.variables['Total_cloud_cover_entire_atmosphere'][:].data.item()
                    weather_data["total_cloud_cover_octas"] = total_cloud_cover
                    found_variables.add('total_cloud_cover_octas')

                if 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average' in available_variables and 'downward_shortwave_radiation_W_m2' not in found_variables:
                    downward_shortwave_radiation = data.variables['Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average'][:].data.item()
                    weather_data["downward_shortwave_radiation_W_m2"] = downward_shortwave_radiation
                    found_variables.add('downward_shortwave_radiation_W_m2')

                logger.debug(f"Weather data after querying {dataset.name}: {weather_data}")

            except Exception as e:
                logger.warning(f"Failed to retrieve data from {dataset.name}: {str(e)}")
        
        # Return the weather data, with missing variables filled as None
        logger.info(f"Final weather data (with None values for missing variables): {weather_data}")
        return weather_data


    def add_weather_inputs(self, flight: Flight) -> FlightWeather:
        """
        Add weather inputs to the flight object.
        
        :param flight: A Flight object containing flight data.
        :return: A FlightWeather object with integrated weather information.
        """
        logger.info(f"Adding weather inputs for flight {flight.flight_icao_number}, en-route from {flight.departure_city} to {flight.arrival_city}.")
        
        # Convert the current flight time (assumed to be in Unix timestamp) to a datetime object in UTC
        flight_time = datetime.fromtimestamp(flight.current_flight_time, tz=timezone.utc)
        logger.debug(f"Flight time in UTC: {flight_time}")

        # Fetch weather data based on the flight's current position and altitude
        weather_data = self.fetch_weather_data(
            latitude=flight.latitude,
            longitude=flight.longitude,
            altitude=flight.altitude,  # Assuming altitude is in meters
            flight_time=flight_time
        )

        # Convert flight object to dictionary
        flight_dict = flight.dict()

        # Safely add the weather data to the flight object with fallback to None for missing values
        flight_dict['temperature_C'] = weather_data.get("temperature_C", None)  # Temperature in Celsius
        flight_dict['wind_speed_u_ms'] = weather_data.get("wind_speed_u_ms", None)  # U component of wind speed
        flight_dict['wind_speed_v_ms'] = weather_data.get("wind_speed_v_ms", None)  # V component of wind speed
        flight_dict['wind_direction'] = weather_data.get("wind_direction", None)  # Wind direction in degrees
        flight_dict['pressure_hPa'] = weather_data.get("pressure_hPa", None)  # Pressure in hPa
        flight_dict['geopotential_height_m'] = weather_data.get("geopotential_height_m", None)  # Geopotential height in meters
        flight_dict['relative_humidity_percent'] = weather_data.get("relative_humidity_percent", None)  # Relative humidity in percent
        flight_dict['specific_humidity_kg_kg'] = weather_data.get("specific_humidity_kg_kg", None)  # Specific humidity in kg/kg
        flight_dict['total_cloud_cover_octas'] = weather_data.get("total_cloud_cover_octas", None)  # Total cloud cover in octas
        flight_dict['high_cloud_cover_octas'] = weather_data.get("high_cloud_cover_octas", None)  # High cloud cover in octas
        flight_dict['downward_shortwave_radiation_W_m2'] = weather_data.get("downward_shortwave_radiation_W_m2", None)  # Downward shortwave radiation in W/m^2

        # Log what data was successfully retrieved and what was missing
        logger.debug(f"Final weather data: {weather_data}")

        # Create or update a FlightWeather object using the updated flight dictionary
        flight_weather = FlightWeather(**flight_dict)

        logger.info(f"Weather inputs added to flight {flight.flight_id}.")
        return flight_weather



if __name__ == "__main__":
    # Create an instance of the AddFlightsWeather class
    logger.info("Starting flight weather integration.")
    live_flights_weather = AddFlightsWeather()

    # Create a Flight object
    flight = Flight(
        aircraft_iata_code="A333",
        aircraft_icao_code="A333",
        airline_iata_code="MU",
        airline_icao_code="CES",
        airline_name="China Eastern Airlines",
        altitude=3147.06,  # Example altitude in meters
        flight_level="FL103",
        arrival_airport_iata="DXB",
        arrival_airport_icao="OMDB",
        arrival_city="Dubai",
        current_flight_time=1725461894,  # Unix timestamp for current flight time
        departure_airport_iata="PVG",
        departure_airport_icao="ZSPD",
        departure_city="Shanghai",
        direction=310.31,
        flight_icao_number="CES245",
        flight_number="245",
        flight_id="MU245",
        flight_status="unknown",
        horizontal_speed=512.46,
        isGround=False,
        latitude=24.9735,
        longitude=56.0713,
        vertical_speed=-2.34,
        departure_country="China",
        arrival_country="United Arab Emirates",
        route="Shanghai - Dubai"
    )
    
    # Call the add_weather_inputs method to fetch and add weather data to the flight
    flight_weather = live_flights_weather.add_weather_inputs(flight)

    # Print the resulting FlightWeather object
    from pprint import pprint
    pprint(flight_weather)
