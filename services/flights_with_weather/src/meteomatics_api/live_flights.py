from loguru import logger
import requests
from datetime import datetime, time
from time import sleep
import math
from math import radians, degrees, sin, cos, atan2
from geopy.distance import geodesic, distance as geodesic_distance
from geopy import Point
import pandas as pd
from typing import List
import pdb

from src.config import config
from src.meteomatics_api.flight import Flight, FlightWeather


class AddFlightsWeather:
    """A class to deal with the weather data for historical flights from the meteomatics API.
    """
    
    # Initializing the class
    def __init__(self):
        # API credentials
        self.username : str = config.meteomatics_username
        self.password : str = config.meteomatics_password
        #Initialize the meteomatics_calls tracker
        self.meteomatics_calls = 0
        # Initialize list to store the weather data
        self.weather_data : List[dict] = []
        #Initialize time_bound_weather_data list
        self.time_bound_weather_data : List[dict] = []
        # Initialize the sampled flight points with timestamps
        self.time_bound_sampled_flight_points : List[dict] = []
        #breakpoint()
        self.sampled_flight_points : List[tuple] =  []
        self.length_of_weather_data = 0
        self.length_of_sampled_flight_points = 0
    
    
    # helper Function to re-initialize the sampled_flight_points list
    def _reinitialize_sampled_flight_points_and_weather_data(self) -> None:
        """
        Re-initializes the sampled_flight_points list and its corresponding weather_data list.
        """
        
        if len(self.time_bound_sampled_flight_points) > 0:
            # Get the unique routes
            unique_routes = set(flight["route"] for flight in self.time_bound_sampled_flight_points)
            Lists_unique_routes = {route: [] for route in unique_routes}
            Lists_unique_weather_data = {route: [] for route in unique_routes}

            # Populate the dictionary with flight points and corresponding weather data for each unique route
            for flight, weather in zip(self.time_bound_sampled_flight_points, self.time_bound_weather_data):
                route = flight["route"]
                Lists_unique_routes[route].append({
                    "time": flight["time"],
                    "latitude": flight["latitude"],
                    "longitude": flight["longitude"]
                })
                Lists_unique_weather_data[route].append(weather)

            # Check if any route's data needs reinitialization based on the 2-hour rule
            for route, points in Lists_unique_routes.items():
                if len(points) > 0:
                    first_time = points[0]["time"]
                    last_time = points[-1]["time"]
                    if (last_time - first_time) >= 7200:  # 2 hours in seconds
                        Lists_unique_routes[route] = []  # Reinitialize list for that route
                        Lists_unique_weather_data[route] = []  # Reinitialize corresponding weather data
                        logger.debug(f"sampled_flight_points and weather_data lists have been emptied for route {route}")
                        logger.debug(f"length of sampled_flight_points for route {route} is {len(Lists_unique_routes[route])}")
                        logger.debug(f"length of weather_data for route {route} is {len(Lists_unique_routes[route])}")
                        #breakpoint()
            # Update the class lists based on the potentially reinitialized routes
            self.time_bound_sampled_flight_points = [
                {
                    "time": point["time"],
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                    "route": route
                }
                for route, points in Lists_unique_routes.items() for point in points
            ]

            self.time_bound_weather_data = [
                weather for route, weathers in Lists_unique_weather_data.items() for weather in weathers
            ]

            # Reset sampled_flight_points and weather_data
            self.sampled_flight_points = [
                (flight["latitude"], flight["longitude"])
                for flight in self.time_bound_sampled_flight_points
            ]
            self.weather_data = [
                {
                    "temperature_C": weather["temperature_C"],
                    "pressure_hPa": weather["pressure_hPa"],
                    "wind_speed_u_ms": weather["wind_speed_u_ms"],
                    "wind_speed_v_ms": weather["wind_speed_v_ms"],
                    "geopotential_height_m": weather["geopotential_height_m"],
                    "relative_humidity_percent": weather["relative_humidity_percent"],
                    "total_cloud_cover_octas": weather["total_cloud_cover_octas"],
                    "high_cloud_cover_octas": weather["high_cloud_cover_octas"],
                    "specific_humidity_kg_kg": weather["specific_humidity_kg_kg"],
                    "prob_contrails_percent": weather["prob_contrails_percent"],
                    "global_radiation_W_m2": weather["global_radiation_W_m2"]
                }
                for weather in self.time_bound_weather_data
            ]
            logger.debug(f"sampled_flight_points and weather_data lists have been reinitialized")
            logger.debug(f"length of sampled_flight_points: {self.length_of_sampled_flight_points}")
            logger.debug(f"length of weather_data: {self.length_of_weather_data}")
        return None
    # Function to retrieve meteomatics near rel time or forecast data
    def add_weather_inputs(
        self, 
        flight : Flight
        ) -> FlightWeather:
        """this method will take in a live flight object and return a FlightWeather object with 
        the weather data for that flight

        Args:
            flight (Flight): a flight object with the flight data from the Aviation Edge API

        Returns:
            FlightWeather: a flight object with the weather data from the ECMWF API
        """
        #get the flight current time 
        flight_time = self.convert_flight_time_to_meteomatics_format(flight.current_flight_time)
        #get the altitude, latitude and longitude of the flight
        altitude = self.get_nearest_allowed_altitude(flight.altitude)
        flight_level = flight.flight_level
        latitude = flight.latitude
        longitude = flight.longitude
        #The meteomatics API boundaries for those parameters
        flight_level_boundaries ={"min": 10,"max": 900}
        altitude_boundaries = {"min": 1, "max": 20000}
        #breakpoint()
        #We need to ensure all those parameters falls within the meteomatics API limits
        if int(flight_level[2:]) < flight_level_boundaries["min"]:
            flight_level = f"FL{flight_level_boundaries['min']}"
        elif int(flight_level[2:]) > flight_level_boundaries["max"]:
            flight_level = f"FL{flight_level_boundaries['max']}"
        if altitude < altitude_boundaries["min"]:
            altitude = altitude_boundaries["min"]
        elif altitude > altitude_boundaries["max"]:
            altitude = altitude_boundaries["max"]
        # Define the parameters, including altitude where applicable
        parameters = (
            f"t_{flight_level}:C",
            f"pressure_{altitude}m:hPa",        #Atmospheric pressure at specific altitude in hPa
            f"wind_speed_u_{flight_level}:ms",  # U component of wind at specific flight level in meters per second                                 
            f"wind_speed_v_{flight_level}:ms",  # V-component of wind at specific flight level in meters per second
            f"gh_{flight_level}:m",                # Geopotential at specific flight level in hPa
            f"relative_humidity_{flight_level}:p",  # Relative Humidity at specific altitude in percentage
            "total_cloud_cover:octas",                         # Total Cloud Cover (not altitude-dependent)
            "high_cloud_cover:octas",                        # High Cloud Cover (not altitude-dependent)
            f"contrail_{flight_level}:p",          #probability of contrail formation at specific flight level in percentage
            f"global_rad:W"                       #Global radiation in Watts per square meter
        )
        
        # Join the parameters into a single string
        parameters_str = ",".join(parameters)

        # Define the URL for the API request
        url = f'https://api.meteomatics.com/{flight_time}/{parameters_str}/{latitude},{longitude}/json'

        # Make the API request
        response = requests.get(url, auth=(self.username, self.password))
    
        #breakpoint()
        # Check if the request was successful
        if response.status_code == 200:
            #whenever a successfull call is made to the meteomatics API, increment the meteomatics_calls tracker
            self.meteomatics_calls += 1
            logger.info(f"Weather data retrieved successfully for flight {flight.flight_id}")
            logger.info(f"Number of calls made to the meteomatics API: {self.meteomatics_calls}")
            # Convert the response to JSON
            weather_data = response.json()
            #breakpoint()
            #cast the flight object to a dictionary
            flight = flight.dict()
            # Extract the weather data from the response and assign it to the FlightWeather object
            flight['temperature_C'] = weather_data['data'][0]['coordinates'][0]['dates'][0]['value']
            flight['pressure_hPa'] = weather_data['data'][1]['coordinates'][0]['dates'][0]['value']
            flight['wind_speed_u_ms'] = weather_data['data'][2]['coordinates'][0]['dates'][0]['value']
            flight['wind_speed_v_ms'] = weather_data['data'][3]['coordinates'][0]['dates'][0]['value']
            flight['geopotential_height_m'] = weather_data['data'][4]['coordinates'][0]['dates'][0]['value']
            flight['relative_humidity_percent'] = weather_data['data'][5]['coordinates'][0]['dates'][0]['value']
            flight['total_cloud_cover_octas'] = weather_data['data'][6]['coordinates'][0]['dates'][0]['value']
            flight['high_cloud_cover_octas'] = weather_data['data'][7]['coordinates'][0]['dates'][0]['value']
            flight['specific_humidity_kg_kg'] = self.calculate_specific_humidity(flight['temperature_C'], flight['relative_humidity_percent'], flight['pressure_hPa'])
            flight['prob_contrails_percent'] = weather_data['data'][8]['coordinates'][0]['dates'][0]['value']
            flight['global_radiation_W_m2'] = weather_data['data'][9]['coordinates'][0]['dates'][0]['value']
            # Create a FlightWeather object from the dictionary
            flight_weather = FlightWeather(**flight)
        elif response.status_code == 429:
            logger.error(f" status code 429, Meteomatics API rate limit reached for flight {flight.flight_id}")
        else:
            logger.error(f"Failed to retrieve weather data for flight {flight.flight_id}, request status code is: {response.status_code} for the request {url}")
            flight_weather = None
        return flight_weather
    
    # helper Function to compute initial compass bearing between two points
    def _calculate_initial_compass_bearing(
        self,
        pointA: Point, 
        pointB: Point
        ) -> float:
        """
        Calculates the initial bearing between two points.
        """
        # Convert the latitude of the start point from degrees to radians
        lat1 = radians(pointA.latitude)
        # Convert the latitude of the end point from degrees to radians
        lat2 = radians(pointB.latitude)
        # Calculate the difference in longitude between the start and end points, and convert it to radians
        diff_long = radians(pointB.longitude - pointA.longitude)
        # Calculate the x component for the bearing calculation
        x = sin(diff_long) * cos(lat2)
        # Calculate the y component for the bearing calculation
        y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(diff_long))
        # Calculate the initial bearing using the atan2 function, which gives the angle in radians
        initial_bearing = atan2(x, y)
        # Convert the bearing from radians to degrees
        initial_bearing = degrees(initial_bearing)
        # Normalize the bearing to a compass bearing between 0° and 360°
        compass_bearing = (initial_bearing + 360) % 360
        # Return the calculated compass bearing
        return compass_bearing
    
    #Function to generate intermediate points every 10km along the great-circle path from start to end
    def interpolate_points(
        self,
        start: Point, 
        end: Point, 
        distance_km: float
        ) -> list:
        """
        Generates intermediate points every `distance_km` along the great-circle path from start to end.

        Args:
            start (Point): The starting point.
            end (Point): The ending point.
            distance_km (float): The distance between each intermediate point in kilometers (e.g., 10 km).

        Returns:
            list: A list of tuples containing the latitude and longitude of the intermediate points.
        """
        # Calculate the full distance between the start and end points
        full_distance = geodesic(start, end).km
        # Calculate the number of intermediate points to generate
        num_points = int(full_distance // distance_km)
        # Initialize a list to store the intermediate points
        points = []
        # Calculate the initial bearing between start and end points
        bearing = self._calculate_initial_compass_bearing(start, end)
        # Generate the intermediate points
        for i in range(1, num_points + 1):
            point = geodesic(kilometers=i * distance_km).destination(point=start, bearing=bearing)
            points.append((point.latitude, point.longitude))

        return points
            
    #Function to fetch wheather data for a given flight /route every 10km
    def get_weather_along_route(
        self,
        flight: Flight
    ) -> FlightWeather:
        """
        This method will take in a live flight object on a given route and return a FlightWeather object with weather
        parameters for every 10km along the route. This is due to the fact that the meteomatics API is very expensive
        and because weather conditions do not change very fast at flight levels.

        Args:
            flight: a flight object containing departure and arrival coordinates

        Returns:
            FlightWeather: The same flight object with enriched weather parameters.
        """
        # Get the departure and arrival airport coordinates
        departure_coords = Point(flight.departure_airport_coords['Latitude'], flight.departure_airport_coords['Longitude'])
        arrival_coords = Point(flight.arrival_airport_coords['Latitude'], flight.arrival_airport_coords['Longitude'])
        #store the latitude and longitude of the flight before updating thm with the closest point ones
        flight_latitude_init = flight.latitude
        flight_longitude_init = flight.longitude
        # Interpolate points every 10 km along the route
        interpolated_positions = self.interpolate_points(departure_coords, arrival_coords, distance_km=10)
        #Define a proximity threshold in km (between 1 km and 5 km)
        proximity_threshold = 5.0
        #Initialize an empty list to store all the points that are within the proximity threshold
        points_within_proximity = []
        # compare the flight coordinates with the interpolated points
        for position in interpolated_positions:
            # Calculate the distance between the flight position and the interpolated point
            distance = geodesic(position, (flight.latitude, flight.longitude)).km
            # Check if the distance is within the proximity threshold
            if distance <= proximity_threshold:
                #Assign the point to the points_within_proximity list
                points_within_proximity.append(position)
        #check if the flight is within the proximity threshold of any interpolated point
        if len(points_within_proximity) > 0:
            #Between the points within the proximity threshold, get the closest point to the flight position
            closest_point = min(points_within_proximity, key=lambda x: geodesic(x, (flight.latitude, flight.longitude)).km)
            #Project the closest point onto the flight direction
            projected_point = self.project_point_on_direction(flight.latitude, flight.longitude, flight.direction, closest_point[0], closest_point[1])
            #Assign the latitude and longitude of the projected closest point to the flight object
            flight.latitude = projected_point[0]
            flight.longitude = projected_point[1]
            # Get the weather data for the flight with the coordinates of the closest point
            flight_weather = self.add_weather_inputs(flight)
            # Pause for 1.2s to respect the rate limit
            sleep(1.2)
            #breakpoint()
            if flight_weather is None:
                #pdb.set_trace()  # This will pause execution and open an interactive debugger
                logger.error(f"Failed to retrieve weather data for flight {flight.flight_id}")
            # Append the weather data to the weather_data list
            self.weather_data = self._append_weather_data(flight_weather)
            self.length_of_weather_data += 1
            logger.debug(f"weather data appended to the weather_data list with current length {self.length_of_weather_data}")
            #Append the time_bound_weather_data list
            self.time_bound_weather_data = self._update_time_bound_weather_data(flight_weather)
            # Pause for 1.2s to respect the rate limit
            sleep(1.2)
            #reasign the initial latitude and longitude to the flight object
            flight_weather.latitude = flight_latitude_init
            flight_weather.longitude = flight_longitude_init
            # Append the sampled position to the sampled_flight_points list
            self.sampled_flight_points = self._update_sampled_flight_points(flight_weather)
            self.length_of_sampled_flight_points += 1
            logger.debug(f"sampled flight point appended to the sampled_flight_points list with current length {self.length_of_sampled_flight_points}")
            #Append the time_bound_sampled_flight_points list
            self.time_bound_sampled_flight_points = self._update_time_bound_sampled_flight_points(flight_weather)
            #Reinitialize the sampled_flight_points list
            self._reinitialize_sampled_flight_points_and_weather_data()
        # check if the flight is not within the proximity threshold of any interpolated point
        elif len(points_within_proximity) == 0:
            # Interpolate or extrapolate the weather data for the flight position
            flight_weather = self.interpolate_weather_data(flight, interpolated_positions)
            logger.debug(f"Number of calls made to the meteomatics API: {self.meteomatics_calls}")
            logger.debug(f"current length of sampled_flight_points: {self.length_of_sampled_flight_points}")
            logger.debug(f"current length of weather_data: {self.length_of_weather_data}")   
            #breakpoint()
        return flight_weather
    
    #Function to interpolate or extrapolate weather data for flight positions without direct data   
    def interpolate_weather_data(
        self,
        flight : Flight, 
        interpolated_positions : List[tuple],
        ) -> FlightWeather:
        """
        Interpolates or extrapolates weather data for flight positions without direct data.

        Args:
            flight: The flight object to update.
            interpolated_positions: Interpolated positions along the flight path.

        Returns:
            Updated flight object with weather data interpolated/extrapolated for all positions.
        """
        #store the latitude and longitude of the flight before updating thm with the closest point ones
        flight_latitude_init = flight.latitude
        flight_longitude_init = flight.longitude
        #if the sampled_flight_points is still empty, compute the weather data at the closest interpolated point
        if not self.sampled_flight_points:
            #Get the closest interpolated point to the flight position
            closest_interpolated_point = min(interpolated_positions, key=lambda x: geodesic(x, (flight.latitude, flight.longitude)).km)
            #Project the closest interpolated point onto the flight direction
            projected_point = self.project_point_on_direction(flight.latitude, flight.longitude, flight.direction, closest_interpolated_point[0], closest_interpolated_point[1])
            logger.debug(f"No flight points found in sampled_flight_points. Interpolating weather data for flight {flight.flight_id} with the closest interpolated point {closest_interpolated_point}")
            #breakpoint()  
            #Assign the latitude and longitude of the projected closest sampled point to the flight object
            flight.latitude = projected_point[0]
            flight.longitude = projected_point[1]
            #Get the weather data for the flight with the coordinates of the closest sampled point
            flight_weather = self.add_weather_inputs(flight)
            # Pause for 1.2s to respect the rate limit
            sleep(1.2)
            #breakpoint()
            # Append the weather data to the weather_data list
            self.weather_data = self._append_weather_data(flight_weather)
            self.length_of_weather_data += 1
            logger.debug(f"weather data appended to the weather_data list with current length {self.length_of_weather_data}")
            #Append the time_bound_sampled_flight_points list
            self.time_bound_sampled_flight_points = self._update_time_bound_sampled_flight_points(flight_weather)
            #store the retrieved weather data in a dictionary
            if flight_weather is None:
                #pdb.set_trace()  # This will pause execution and open an interactive debugger
                logger.error(f"Failed to retrieve weather data for flight {flight.flight_id}")
            #reasign the initial latitude and longitude to the flight object
            flight_weather.latitude = flight_latitude_init
            flight_weather.longitude = flight_longitude_init
            # Append the sampled position to the sampled_flight_points list
            self.sampled_flight_points = self._update_sampled_flight_points(flight_weather)
            #breakpoint()
            self.length_of_sampled_flight_points += 1
            logger.debug(f"sampled flight point appended to the sampled_flight_points list with current length {self.length_of_sampled_flight_points}")
            #Append the time_bound_sampled_flight_points list
            #breakpoint()
            self.time_bound_sampled_flight_points = self._update_time_bound_sampled_flight_points(flight_weather)
            #breakpoint()
            #Reinitialize the sampled_flight_points list
            self._reinitialize_sampled_flight_points_and_weather_data() 
        #if the sampled_flight_points is not empty, compute the weather data at the closest sampled flight point
        else: 
            #Cast the flight object to a dictionary
            flight_dict = flight.dict()
            #find the closest sampled flight point to the flight position
            closest_sampled_point = min(self.sampled_flight_points, key=lambda x: geodesic(x, (flight.latitude, flight.longitude)).km)
            logger.debug(f"Interpolating weather data for flight {flight.flight_id} with the closest sampled flight point {closest_sampled_point}")
            #Assign the weather data of the closest sampled point to the flight object
            flight_dict.update(self.weather_data[self.sampled_flight_points.index(closest_sampled_point)])
            #recast the flight object to a FlightWeather object
            flight_weather = FlightWeather(**flight_dict)
            #breakpoint()
            if flight_weather is None:
                #pdb.set_trace()  # This will pause execution and open an interactive debugger
                logger.error(f"Failed to retrieve weather data for flight {flight.flight_id}")
            #breakpoint()
            #Reinitialize the sampled_flight_points list
            self._reinitialize_sampled_flight_points_and_weather_data()
        return flight_weather
    
    #A helper function to append the weather data every time a call is made to the meteomatics API
    def _append_weather_data(
        self, 
        flight_weather : FlightWeather,
        ) -> List[dict]:
        """A helper function to append the weather data every time a call is made to the meteomatics API

        Args:
            flight_weather (FlightWeather): flight with weather data

        Returns:
            None
        """
        # Append the weather data to the weather_data list
        self.weather_data.append({
            "temperature_C": flight_weather.temperature_C,
            "pressure_hPa": flight_weather.pressure_hPa,
            "wind_speed_u_ms": flight_weather.wind_speed_u_ms,
            "wind_speed_v_ms": flight_weather.wind_speed_v_ms,
            "geopotential_height_m": flight_weather.geopotential_height_m,
            "relative_humidity_percent": flight_weather.relative_humidity_percent,
            "total_cloud_cover_octas": flight_weather.total_cloud_cover_octas,
            "high_cloud_cover_octas": flight_weather.high_cloud_cover_octas,
            "specific_humidity_kg_kg": flight_weather.specific_humidity_kg_kg,
            "prob_contrails_percent": flight_weather.prob_contrails_percent,
            "global_radiation_W_m2": flight_weather.global_radiation_W_m2
            })
        return self.weather_data
    
    #Function to update the time_bound_weather_data list
    def _update_time_bound_weather_data(
        self,
        flight_weather : FlightWeather
        ) -> List[dict]:
        """A function to update the time_bound_weather_data list with the latest weather data
        
        Args:
            flight_weather (FlightWeather): flight with weather data
            
        Returns:
            None
        """
        # Append the weather data to the time_bound_weather_data list
        self.time_bound_weather_data.append({
            "temperature_C": flight_weather.temperature_C,
            "pressure_hPa": flight_weather.pressure_hPa,
            "wind_speed_u_ms": flight_weather.wind_speed_u_ms,
            "wind_speed_v_ms": flight_weather.wind_speed_v_ms,
            "geopotential_height_m": flight_weather.geopotential_height_m,
            "relative_humidity_percent": flight_weather.relative_humidity_percent,
            "total_cloud_cover_octas": flight_weather.total_cloud_cover_octas,
            "high_cloud_cover_octas": flight_weather.high_cloud_cover_octas,
            "specific_humidity_kg_kg": flight_weather.specific_humidity_kg_kg,
            "prob_contrails_percent": flight_weather.prob_contrails_percent,
            "global_radiation_W_m2": flight_weather.global_radiation_W_m2,
            "time": flight_weather.current_flight_time, 
            "latitude": flight_weather.latitude, 
            "longitude" : flight_weather.longitude,
            "departure_city": flight_weather.departure_city,
            "arrival_city": flight_weather.arrival_city,
            "route": flight_weather.route
        })
        
        return self.time_bound_weather_data
    
    #Function to update time_bound_sampled_flight_points list
    # helper Function to update the sampled flight points List
    def _update_time_bound_sampled_flight_points(
        self, 
        flight_weather : FlightWeather
        ) -> List[dict]:
        """A function to update the sampled flight points list with the latest sampled flight point

        Args:
            flight_weather (FlightWeather): flight with weather data

        Returns:
            None
        """
        # Append the sampled position to the time_bound_sampled_flight_points list
        self.time_bound_sampled_flight_points.append(
            {
                "time": flight_weather.current_flight_time, 
                "latitude": flight_weather.latitude, 
                "longitude" : flight_weather.longitude,
                "departure_city": flight_weather.departure_city,
                "arrival_city": flight_weather.arrival_city,
                "route": flight_weather.route
            }
        )
        return self.time_bound_sampled_flight_points
    
    # helper Function to update the sampled flight points List
    def _update_sampled_flight_points(
        self, 
        flight_weather : FlightWeather
        ) -> List[tuple]:
        """A function to update the sampled flight points list with the latest sampled flight point

        Args:
            flight_weather (FlightWeather): flight with weather data

        Returns:
            None
        """
        # Append the sampled position to the sampled_flight_points list
        self.sampled_flight_points.append((flight_weather.latitude,flight_weather.longitude))
        return self.sampled_flight_points
    
    #A Function to project an interpolated point onto the flight direction of the flight
    def project_point_on_direction(
        self,
        flight_lat: float, 
        flight_lon: float, 
        flight_direction: float, 
        interp_lat: float, 
        interp_lon: float
    ) -> tuple:
        """
        Projects an interpolated point onto the flight's direction.
        
        Args:
            flight_lat (float): The latitude of the current flight position.
            flight_lon (float): The longitude of the current flight position.
            flight_direction (float): The direction of the flight in degrees.
            interp_lat (float): The latitude of the interpolated point.
            interp_lon (float): The longitude of the interpolated point.
            
        Returns:
            tuple: The projected latitude and longitude.
        """
        # Create Point objects for flight position and interpolated point
        flight_point = Point(flight_lat, flight_lon)
        interp_point = Point(interp_lat, interp_lon)

        # Calculate distance between the flight position and the interpolated point
        distance = geodesic(flight_point, interp_point).km
        
        # Project the interpolated point along the flight's direction
        projected_point = geodesic(kilometers=distance).destination(point=flight_point, bearing=flight_direction)
        
        return (projected_point.latitude, projected_point.longitude)
    
    #Function to convert the flight time to meteomatics time format 2024-08-24T12:00:00Z
    def convert_flight_time_to_meteomatics_format(
        self,
        flight_time_seconds : int
        ) -> str:
        """
        Converts flight time from seconds since the Unix epoch to Meteomatics time format (YYYY-MM-DDTHH:MM:SSZ).

        Args:
        - flight_time_seconds (int): Flight time in seconds since the Unix epoch.

        Returns:
        - str: Flight time in Meteomatics format.
        """
        # Convert the time from seconds to a datetime object
        flight_datetime = datetime.utcfromtimestamp(flight_time_seconds)
        
        # Format the datetime object to the required Meteomatics format
        meteomatics_time_format = flight_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return meteomatics_time_format
    
    #Function to get the nearest allowed altitude by meteomatics
    def get_nearest_allowed_altitude(
        self,
        requested_altitude : float
        ) -> float:
        # Define the available altitudes in meters (from Meteomatics documentation)
        allowed_altitudes = [
            -150, -50, -15, -5, 0, 2, 10, 20, 50, 100, 150, 200, 300, 400, 500,
            700, 850, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 
            15000, 20000
        ]
        
        # Find the nearest allowed altitude
        nearest_altitude = min(allowed_altitudes, key=lambda x: abs(x - requested_altitude))
        
        return nearest_altitude
    
    
    #Function to compute specific humidity from relative humidity temperature and pressure
    def calculate_specific_humidity(
        self,
        temp_celsius : float, 
        relative_humidity : float, 
        pressure_hpa : float
        ) -> float:
        """
        Calculate specific humidity from temperature, relative humidity, and pressure.
        
        Parameters:
            param temp_celsius: Temperature in degrees Celsius
            param relative_humidity: Relative humidity as a percentage (0 to 100)
            param pressure_hpa: Atmospheric pressure in hPa
        
        return: 
            Specific humidity in kg/kg 
        """

        # Convert relative humidity to a fraction
        rh_fraction = relative_humidity / 100.0

        # Calculate the saturation vapor pressure (e_s) in hPa
        e_s = 6.112 * math.exp((17.67 * temp_celsius) / (temp_celsius + 243.5))

        # Calculate the actual vapor pressure (e) in hPa
        e = rh_fraction * e_s

        # Calculate the specific humidity (q) in kg/kg
        specific_humidity = (0.622 * e) / (pressure_hpa - 0.378 * e)

        return specific_humidity
    
    
    
if __name__ == "__main__":
    # Create an instance of the LiveFlightsWeather class
    live_flights_weather = AddFlightsWeather()

   # Create a Flight object
    flight = Flight(
        aircraft_iata_code="B763",
        aircraft_icao_code="B763",
        airline_iata_code="5X",
        airline_icao_code="UPS",
        altitude=9974.58,
        arrival_airport_iata="CGN",
        arrival_airport_icao="EDDK",
        arrival_city="Cologne",
        current_flight_time=1724792828,
        departure_airport_iata="EMA",
        departure_airport_icao="EGNX",
        departure_city="East Midlands",
        direction=126,
        flight_icao_number="UPS237",
        flight_number="237",
        flight_id="5X237",
        flight_status="en-route",
        horizontal_speed=892.664,
        isGround=False,
        latitude=51.8126,
        longitude=-0.5698,
        vertical_speed=0,
        departure_country="United Kingdom",
        arrival_country="Germany",
        departure_airport_coords={"Latitude": 52.8311004639, "Longitude": -1.32806003094},
        arrival_airport_coords={"Latitude": 50.8658981323, "Longitude": 7.1427397728}
    )
    # Call the add_weather_inputs method
    flight_weather = live_flights_weather.get_weather_along_route(flight)
    from pprint import pprint
    # Print the resulting FlightWeather object
    pprint(flight_weather)