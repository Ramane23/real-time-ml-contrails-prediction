from loguru import logger
from ecmwfapi import ECMWFDataServer
from datetime import datetime, time
#import math
import xarray as xr
import os


from src.ecmwf_api.flight import Flight, FlightWeather


class LiveFlightsWeather:
    """A class to deal with the weather data for historical flights from the ECMWF CDS API.
    More specifically we will use the ECMWF Web API, 
    the HRES (High-Resolution) model outputs for variables such as temperature, 
    humidity, and wind at various altitudes
    """

    # Initialize the ECMWF server client
    server = ECMWFDataServer()
    
    # Initializing the class
    def __init__(self):
        pass

    # Function to retrieve HRES near rel time or forecast data
    def add_weather_inputs(
        self, 
        flight : Flight
        ) -> FlightWeather:
        """this method will take in a live flight object and return a FlightWeather object with the weather data for that flight

        Args:
            flight (Flight): a flight object with the flight data from the Aviation Edge API

        Returns:
            FlightWeather: a flight object with the weather data from the ECMWF API
        """
        #get the forecast date and time as required by the ECMWF API, from the flight timestamp
        #breakpoint()
        forecast_date = self.convert_flight_to_ecmwf_format(flight.current_flight_time)['forecast_date']
        forecast_time = self.convert_flight_to_ecmwf_format(flight.current_flight_time)['forecast_time']
        #get the pressure level as required by the ECMWF API, from the flight altitude
        pressure_level = self.find_nearest_standard_level(self.altitude_to_pressure(flight.altitude))
        #get the latitude and longitude of the flight
        lat = flight.latitude
        lon = flight.longitude
        #compute the relative humidity from the specific humidity, temperature and pressure
        #relative_humidity = self.calculate_rhi(specific_humidity, temperature, pressure)
        #defining a temporary file in the current directory to store the weather data
        temp_file = 'hres_weather.nc'

        self.server.retrieve({
            'class': 'od',                     # Operational data (high-resolution forecasts)
            'dataset': 'HRES',                 # High-Resolution Forecast (HRES) dataset
            'stream': 'oper',                  # Operational stream
            'expver': '1',                     # Experiment version (usually '1' for operational)
            'type': 'fc',                      # Forecast data (use 'an' for analysis if needed)
            'date': forecast_date,             # The date of the forecast run
            'time': forecast_time,             # Time of the forecast run (00:00, 06:00, 12:00, 18:00 UTC)
            'step': '0',                       # Forecast step (0 for the initial time, use '1', '3', etc., for forecast steps)
            'levtype': 'pl',                   # Pressure levels for vertical data
            'levelist': pressure_level,        # Pressure levels in hPa available in the HRES model
            'param': '130.128/131.128/132.128/129.128/133.128/164.128/186.128/187.128/188.128',  
                                                # Parameters for T, U, V, Geopotential, Specific Humidity, TCC, HCC, LCC, MCC
            'area': [lat, lon, lat, lon],      # Geographical area (specifying a point here)
            'grid': '0.1/0.1',                 # High spatial resolution, best for point data 
            'format': 'netcdf',                # Output format
            'target': temp_file                # Output temporary file
        })

        # Convert the NetCDF file to JSON format
        weather_data = self.convert_netcdf_to_dict(temp_file)
        breakpoint()
        #update the flight object with the weather data
        
        
        return flight
    
    #Function to convert the netcdf file into a dictionary
    def convert_netcdf_to_dict(
        self,
        netcdf_file) -> dict:
        """
        Load NetCDF data from a file, convert it to a dictionary, and remove the file.
        
        Args:
            netcdf_file (str): Path to the NetCDF file to be processed.
        
        Returns:
            dict: The data from the NetCDF file as a Python dictionary.
        """
        # Load the NetCDF data into an xarray.Dataset
        ds = xr.open_dataset(netcdf_file)

        # Convert the dataset to a dictionary
        data_dict = ds.to_dict()

        # Remove the temporary file
        os.remove(netcdf_file)

        return data_dict
        
    # Function to convert the flight timestamp to the correct date and time format 
    # for ECMWF HRES data requests   
    def convert_flight_to_ecmwf_format(
        self,
        timestamp_seconds : int
        ):
        """
        Converts a flight timestamp in seconds to the correct date and time format
        for ECMWF HRES data requests.
        
        Args:
            timestamp_seconds (int): The flight timestamp in Unix time (seconds since epoch).

        Returns:
            dict: A dictionary with formatted date and the closest forecast time.
        """
        # Convert the timestamp from seconds to a datetime object
        flight_time = datetime.utcfromtimestamp(timestamp_seconds)
        
        # Extract the date in YYYY-MM-DD format
        forecast_date = flight_time.strftime('%Y-%m-%d')
        
        # Determine the closest ECMWF forecast run time (00:00, 06:00, 12:00, 18:00 UTC)
        forecast_hour = flight_time.hour // 6 * 6  # Round down to the nearest forecast run time
        # Format the forecast time as HH:00:00
        forecast_time = f'{forecast_hour:02d}:00:00'
        
        # Print the formatted date and time for debugging (optional)
        print(f"Flight timestamp: {timestamp_seconds} -> ECMWF date: {forecast_date}, time: {forecast_time}")

        # Return the formatted date and time
        return {
            'forecast_date': forecast_date,
            'forecast_time': forecast_time
        }
        
    #Function to convert the flight altitude to the required pressure levels for ECMWF HRES data requests
    def altitude_to_pressure(
        self,
        altitude_m : float
        ) -> float:
        """
        Converts altitude in meters to the corresponding pressure level in hPa.
        
        Args:
            altitude_m (float): Altitude in meters.
            
        Returns:
            float: Approximate pressure level in hPa.
        """
        # Constants
        P0 = 1013.25  # Sea level standard atmospheric pressure in hPa
        T0 = 288.15   # Standard temperature at sea level in Kelvin
        L = 0.0065    # Temperature lapse rate in K/m
        g = 9.80665   # Acceleration due to gravity in m/s^2
        M = 0.0289644 # Molar mass of Earth's air in kg/mol
        R = 8.31432   # Universal gas constant in N·m/(mol·K)
        
        # Calculate the pressure level using the barometric formula
        pressure_hPa = P0 * (1 - (L * altitude_m) / T0) ** (g * M / (R * L))
        
        return pressure_hPa
    
    #Function to find the nearest pressure level in the ECMWF HRES model to the flight altitude
    def find_nearest_standard_level(
        self,
        calculated_pressure : float, 
        standard_levels=[1000, 925, 850, 700, 500, 250, 200, 100]
        ) -> int:
        """
        Finds the nearest standard pressure level for the given calculated pressure.
        
        Args:
            calculated_pressure (float): The calculated pressure level in hPa.
            standard_levels (list): A list of standard pressure levels in hPa.
        
        Returns:
            int: The nearest standard pressure level.
        """
        nearest_level = min(standard_levels, key=lambda x: abs(x - calculated_pressure))
        return nearest_level
    
    #Function to compute relative humidity from specific humidity temperature and pressure
    def calculate_rhi(
        self,
        specific_humidity : float, 
        temperature : float,
        pressure : float
        ) -> float:
        """
        Calculate Relative Humidity with respect to Ice (RHi).
        
        Args:
            specific_humidity (float): Specific humidity in kg/kg.
            temperature (float): Temperature in Kelvin.
            pressure (float): Pressure in hPa.
        
        Returns:
            float: Relative Humidity with respect to Ice (RHi) in percentage.
        """
        # Constants for saturation vapor pressure over ice (Goff-Gratch formula)
        E0_ice = 6.112  # hPa
        A = 22.46
        B = 272.62  # K
        
        # Calculate saturation vapor pressure over ice (es_ice)
        es_ice = E0_ice * 10 ** ((A * (temperature - 273.15)) / (temperature - B))
        
        # Calculate water vapor pressure (e)
        e = specific_humidity * pressure / (0.622 + 0.378 * specific_humidity)
        
        # Calculate Relative Humidity with respect to Ice (RHi)
        rhi = (e / es_ice) * 100
        
        return rhi
    
#if this file is run as a script, run the following code
if __name__ == "__main__":
    #create an instance of the LiveFlightsWeather class
    live_flights_weather = LiveFlightsWeather()
    #create a dummy flight object
    flight = Flight(
        aircraft_iata_code='B734',
        aircraft_icao_code='B734',
        airline_iata_code='5T',
        airline_icao_code='AKT',
        altitude=7620,
        arrival_airport_iata='YRT',
        arrival_airport_icao='CYRT',
        arrival_city='Rankin Inlet',
        current_flight_time=1724511537,
        departure_airport_iata='YWG',
        departure_airport_icao='CYWG',
        departure_city='Winnipeg',
        direction=11,
        flight_icao_number='AKT150',
        flight_number='150',
        flight_id='5T150',
        flight_status='en-route',
        horizontal_speed=827.844,
        isGround=False,
        latitude=56.9029,
        longitude=-94.7729,
        vertical_speed=0
    )

    #call the add_weather_inputs method to get the weather data for the flight
    weather_data = live_flights_weather.add_weather_inputs(flight)
    #print the weather data
    print(weather_data)