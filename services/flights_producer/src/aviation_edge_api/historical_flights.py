import requests
from typing import List
from src.config import config
import pandas as pd
from datetime import datetime

from src.aviation_edge_api.flight import Flight


class historicalFlights:
    # getting all the live flights
    url = f"https://aviation-edge.com/v2/public/flights?key={config.aviation_edge_api_key}"

    # getting all the live flights within a circle area based on lat and lng values and radius as the distance
    # url = f'https://aviation-edge.com/v2/public/flights?key={config.aviation_edge_api_key}&lat={config.latitude}&lng={config.longititude}&distance={config.distance}&arrIata=LHR'

    def __init__(self) -> None:
        self.is_done: bool = False
        self.airports_database = self._airports_database_to_dataframe()
        self.days: float = config.days
        # the start date should be the exact time we started fetching data in historical mode
        # Set the start date to the exact time of instantiation
        self.start_date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # create a method to get the start date of the historical data fetching
    def get_start_date(self) -> str:
        """get the date and time when we started fetching historical data

        Returns:
            start_date (str): _description_
        """
        # we need to track when we launched the historical data fetching

    # get the historical flights data from the aviation edge API
    def get_flights(self) -> List[Flight]:
        """
        Get live flights data from the Aviation Edge API

        Args:
            None

        Returns:
            List[Flight]: Live flights data as Flight objects
        """
        # Make a request to the API
        data = requests.get(self.url)
        # Check if the request was successful
        if data.status_code == 200:
            # Convert the response to a list of dictionaries
            data = data.json()
            # Refactor the flights to keep only relevant data and convert to Flight objects
            response = []
            for flight_data in data:
                flight = {
                    "current_flight_time": flight_data["system"]["updated"],
                    "departure_airport_icao": flight_data["departure"]["icaoCode"],
                    "departure_airport_iata": flight_data["departure"]["iataCode"],
                    "arrival_airport_icao": flight_data["arrival"]["icaoCode"],
                    "arrival_airport_iata": flight_data["arrival"]["iataCode"],
                    "airline_icao_code": flight_data["airline"]["icaoCode"],
                    "airline_iata_code": flight_data["airline"]["iataCode"],
                    "aircraft_icao_code": flight_data["aircraft"]["icaoCode"],
                    "aircraft_iata_code": flight_data["aircraft"]["iataCode"],
                    "flight_number": flight_data["flight"]["number"],
                    "flight_id": flight_data["flight"]["iataNumber"],
                    "flight_icao_number": flight_data["flight"]["icaoNumber"],
                    "altitude": flight_data["geography"]["altitude"],
                    "latitude": flight_data["geography"]["latitude"],
                    "longitude": flight_data["geography"]["longitude"],
                    "direction": flight_data["geography"]["direction"],
                    "horizontal_speed": flight_data["speed"]["horizontal"],
                    "vertical_speed": flight_data["speed"]["vspeed"],
                    "isGround": flight_data["speed"]["isGround"],
                    "flight_status": flight_data["status"],
                }

                # Check if flight_id is 'XXF' and drop it if so
                if flight["flight_number"] == "XXF":
                    continue
                # drop the flight whenever the isGround is True
                elif flight["isGround"]:
                    continue
                # drop the flight whenever the flight_status is 'landed'
                elif flight["flight_status"] == "landed":
                    continue
                else:
                    # Create a new flight_id key by concatenating the airline IATA code and the flight_number
                    # flight['flight_id'] = f"{flight['airline_iata_code']}{flight['flight_number']}"
                    # Get the departures and arrival city names based on the ICAO and IATA codes
                    departure_city, arrival_city = self.get_city_by_code(
                        flight["departure_airport_icao"],
                        flight["arrival_airport_icao"],
                        flight["departure_airport_iata"],
                        flight["arrival_airport_iata"],
                    )
                    # Append the city names to the flight dictionary
                    flight["departure_city"] = departure_city
                    flight["arrival_city"] = arrival_city

                    # Convert the flight dictionary to a Flight object
                    flight_obj = Flight(**flight)
                    response.append(flight_obj)
        else:
            # Print an error message if the request was not successful
            print(f"Failed to retrieve data: {data.status_code}")
        # breakpoint()
        return response

    def _airports_database_to_dataframe(self) -> None:
        """Downloads the openflights airports database and loads it into a pandas DataFrame.

        Args:
            None

        Returns:
            None
        """
        # load the openflights airports database into a pandas DataFrame
        # return the DataFrame
        openflights_airports = pd.read_csv("./airports.csv")
        #breakpoint()
        # return the DataFrame
        return openflights_airports

    def get_city_by_code(
        self,
        departure_airport_icao: str,
        arrival_airport_icao: str,
        departure_airport_iata: str,
        arrival_airport_iata: str,
    ) -> List[str]:
        """
        Takes in the departure and arrival airport ICAO & IATA codes and returns the cities names
        by searching and retrieving the equivalent cities names in the openflights airports
        databases.

        Args:
            departure_airport_icao (str): the ICAO code of the departure airport
            arrival_airport_icao (str): the ICAO code of the arrival airport
            departure_airport_iata (str): the IATA code of the departure airport
            arrival_airport_iata (str): the IATA code of the arrival airport

        Returns:
            List[str]: City names for departure and arrival airports
        """
        # Function to get city name from ICAO or IATA code
        def get_city(code: str, column: str) -> str:
            result = self.airports_database[self.airports_database[column] == code]
            if not result.empty:
                city_value = result["City"].values[0]
                if pd.notna(city_value):  # Check if the value is not NaN
                    return str(city_value).strip()  # Convert to string and strip whitespace
            return "Unknown"

        # Use ICAO if available, otherwise use IATA
        departure_city = (
            get_city(departure_airport_icao, "ICAO")
            if departure_airport_icao
            else get_city(departure_airport_iata, "IATA")
        )
        arrival_city = (
            get_city(arrival_airport_icao, "ICAO")
            if arrival_airport_icao
            else get_city(arrival_airport_iata, "IATA")
        )

        return [departure_city, arrival_city]


    # create a helper method to convert in milliseconds
    def _convert_to_milliseconds(self, days: int) -> int:
        """Convert days to milliseconds"""
        return days * 24 * 60 * 60 * 1000

    def _is_done(self) -> bool:
        """Check if the fetching process has reached the specified number of days."""

        # Get the current date and time
        current_date = pd.Timestamp.now()

        # Get the start date in datetime
        start_date = pd.Timestamp(self.start_date)

        # Calculate the difference in time between the current and start date in milliseconds
        time_diff = (current_date - start_date).total_seconds() * 1000

        # If the difference in milliseconds is greater than or equal to the specified number of days in milliseconds, mark as done
        if time_diff >= self._convert_to_milliseconds(self.days):
            # Set is_done to True
            self.is_done = True
        else:
            # Set is_done to False
            self.is_done = False

        return self.is_done


# Create an instance of the LiveFlights class
#historical_flights = historicalFlights()
# Get the live flights data
# response = historical_flights.get_flights()
# Print the response
# from pprint import pprint
# pprint(response)
#airports = historical_flights._airports_database_to_dataframe()
#airports.head()
