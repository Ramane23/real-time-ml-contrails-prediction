from cdsapi import Client
import xarray as xr
from datetime import datetime
import os
from loguru import logger
from typing import List
import warnings

from src.ecmwf_api.flight import Flight, FlightWeather

class historicalFlightsWeather:
    """A class to deal with the weather data for historical flights from the ECMWF CDS API."""
    
    # Initialize the ECMWF CDS server client
    cds = Client()

    def add_weather_inputs(
        self, 
        flight : Flight
        ) -> FlightWeather:
        """This method will take in a live flight object and return a FlightWeather object with 
        the weather data for that flight.
        
        Args: 
            flight (Flight): A flight object with the flight data from the Aviation Edge API.
            
        Returns:
            FlightWeather: A flight object with the weather data from the ECMWF API.
        """
        
        # Get the forecast date and time
        forecast_date = self.convert_flight_to_ecmwf_format(flight.current_flight_time)['forecast_date']
        forecast_time = self.convert_flight_to_ecmwf_format(flight.current_flight_time)['forecast_time']
        #breakpoint()
        # Get the pressure level from the flight altitude
        pressure_level = self.find_nearest_standard_level(self.altitude_to_pressure(flight.altitude))
        
        # Get the latitude and longitude of the flight
        lat, lon = flight.latitude, flight.longitude
        
        # Define a temporary file to store the weather data
        temp_file = 'era5_weather.nc'

        # Retrieve the weather data from ERA5
        self.cds.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'temperature',
                    'u_component_of_wind',
                    'v_component_of_wind',
                    'geopotential',
                    'specific_humidity',
                    'relative_humidity',
                ],
                'pressure_level': str(pressure_level),
                'year': forecast_date[:4],
                'month': forecast_date[5:7],
                'day': forecast_date[8:10],
                'time': forecast_time,
                'area': [lat, lon, lat, lon],
                'format': 'NetCDF4',
            },
            temp_file
        )
        
        # Retrieve cloud cover data (optional, if not available in pressure levels)
        self.cds.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'total_cloud_cover',
                    'high_cloud_cover',
                ],
                'year': forecast_date[:4],
                'month': forecast_date[5:7],
                'day': forecast_date[8:10],
                'time': forecast_time,
                'area': [lat, lon, lat, lon],
                'format': 'NetCDF4',
            },
            temp_file
        )

        # Convert the NetCDF file to a dictionary
        weather_data = self.convert_netcdf_to_dict(temp_file)
        breakpoint()
        #cast the flight object to a dictionary
        flight = flight.dict()
        
        # Update the flight object with the weather data
       
        
        return flight

    def convert_netcdf_to_dict(
        self, 
        netcdf_file : str
        ) -> dict:
        """Load NetCDF data from a file, convert it to a dictionary, and remove the file.
        
        Args: 
            netcdf_file (str): The path to the NetCDF file.
        Returns:
            dict: The data in the NetCDF file as a dictionary.
        """
        # Load the NetCDF file
        ds = xr.open_dataset(netcdf_file)
        # Convert the data to a dictionary
        data_dict = ds.to_dict()
        # Remove the NetCDF file
        os.remove(netcdf_file)
        return data_dict

    def convert_flight_to_ecmwf_format(
        self, 
        timestamp_seconds : int
        ) -> dict:
        """Convert a flight timestamp in seconds to the correct date and time format for ECMWF HRES data requests."""
        # Convert the timestamp to a datetime object
        flight_time = datetime.utcfromtimestamp(timestamp_seconds)
        # Get the forecast date 
        forecast_date = flight_time.strftime('%Y-%m-%d')
        # Get the forecast time (rounded to the nearest 6 hours)
        #forecast_hour = flight_time.hour // 6 * 6
        #forecat hour rounded to the nearest hour
        forecast_hour = flight_time.hour 
        # Format the forecast time
        forecast_time = f'{forecast_hour:02d}:00:00'
        return {
            'forecast_date': forecast_date, 
            'forecast_time': forecast_time
            }

    def altitude_to_pressure(
        self,
        altitude_m : float
        ) -> float:
        """Convert altitude in meters to the corresponding pressure level in hPa.
        
        Args: 
            altitude_m (float): The altitude in meters.
            
        Returns:
            float: The pressure level in hPa.
        
        """
        P0 = 1013.25
        T0 = 288.15
        L = 0.0065
        g = 9.80665
        M = 0.0289644
        R = 8.31432
        pressure_hPa = P0 * (1 - (L * altitude_m) / T0) ** (g * M / (R * L))
        return pressure_hPa

    def find_nearest_standard_level(
        self, 
        calculated_pressure : float, 
        standard_levels : List[int] = [1000, 925, 850, 700, 500, 250, 200, 100]
        ) -> int:
        """Find the nearest standard pressure level for the given calculated pressure.
        
        Args:
            calculated_pressure (float): The calculated pressure in hPa.
            standard_levels (List[int]): A list of standard pressure levels in hPa.
        Returns:
            int: The nearest standard pressure level in hPa.
        """
        
        nearest_level = min(standard_levels, key=lambda x: abs(x - calculated_pressure))
        return nearest_level
        

# Example usage
if __name__ == "__main__":
    live_flights_weather = historicalFlightsWeather()
    flight = Flight(
        aircraft_iata_code='B738',
        aircraft_icao_code='B738',
        airline_iata_code='FR',
        airline_icao_code='RYR',
        altitude=11277.6,
        arrival_airport_iata='ACE',
        arrival_airport_icao='GCRR',
        arrival_city='Arrecife',
        current_flight_time=1724025600,
        departure_airport_iata='CRL',
        departure_airport_icao='EBCI',
        departure_city='Charleroi',
        direction=213.0,
        flight_icao_number='RYR9EC',
        flight_number='8174',
        flight_id='FR8174',
        flight_status='',
        horizontal_speed=831.548,
        isGround=False,
        latitude=38.8078,
        longitude=-4.8481,
        vertical_speed=0.0
    )

    weather_data = live_flights_weather.add_weather_inputs(flight)
    print(weather_data)