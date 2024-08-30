import requests
from typing import List, Dict
import pandas as pd
from loguru import logger

from src.aviation_edge_api.flight import Flight
from src.config import config
from src.path import get_airports_path



class LiveFlights:
    # getting all the live flights
    url = f"https://aviation-edge.com/v2/public/flights?key={config.aviation_edge_api_key}"

    # getting all the live flights within a circle area based on lat and lng values and radius as the distance
    # url = f'https://aviation-edge.com/v2/public/flights?key={config.aviation_edge_api_key}&lat={config.latitude}&lng={config.longititude}&distance={config.distance}&arrIata=LHR'

    def __init__(self) -> None:
        self.is_done = False
        self.airports_database = self._airports_database_to_dataframe()

    def get_flights(self) -> List[Flight]:
        """
        Get live flights data from the Aviation Edge API

        Args:
            None

        Returns:
            List[Flight]: Live flights data as Flight objects
        """
        #Define a list of European countries
        european_countries = [
                "Albania",
                "Andorra",
                "Armenia",
                "Austria",
                "Azerbaijan",
                "Belarus",
                "Belgium",
                "Bosnia and Herzegovina",
                "Bulgaria",
                "Croatia",
                "Cyprus",
                "Czech Republic",
                "Denmark",
                "Estonia",
                "Finland",
                "France",
                "Georgia",
                "Germany",
                "Greece",
                "Hungary",
                "Iceland",
                "Ireland",
                "Italy",
                "Kazakhstan",
                "Kosovo",
                "Latvia",
                "Liechtenstein",
                "Lithuania",
                "Luxembourg",
                "Malta",
                "Moldova",
                "Monaco",
                "Montenegro",
                "Netherlands",
                "North Macedonia",
                "Norway",
                "Poland",
                "Portugal",
                "Romania",
                "Russia",
                "San Marino",
                "Serbia",
                "Slovakia",
                "Slovenia",
                "Spain",
                "Sweden",
                "Switzerland",
                "Turkey",
                "Ukraine",
                "United Kingdom",
                "Vatican City"
        ]
        # Make a request to the API
        data = requests.get(self.url)
        #breakpoint()
        # Check if the request was successful
        if data.status_code == 200:
            #breakpoint()
            # Convert the response to a list of dictionaries
            data = data.json()
            if data != {'error': 'No Record Found', 'success': False}:
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
                    #drop the flight with potential mistakes in the altitude reported
                    elif flight["altitude"] <= 0:
                        continue
                    # drop the flight whenever the flight_status is 'landed'
                    elif flight["flight_status"] == "landed":
                        continue
                    else:
                        #Add flight level to the flight dictionary
                        flight['flight_level'] = f"FL{self.altitude_to_flight_level(flight['altitude'])}"
                        #Add the departures and arrivals coordinates to the flight dictionary
                        flight['departure_airport_coords'] = self.get_airport_coordinates_by_code(
                            flight["departure_airport_icao"],
                            flight["arrival_airport_icao"],
                            flight["departure_airport_iata"],
                            flight["arrival_airport_iata"],
                        )[0]
                        flight['arrival_airport_coords'] = self.get_airport_coordinates_by_code(
                            flight["departure_airport_icao"],
                            flight["arrival_airport_icao"],
                            flight["departure_airport_iata"],
                            flight["arrival_airport_iata"],
                        )[1]
                        # Get the departures and arrival city names based on the ICAO and IATA codes
                        departure, arrival = self.get_city_and_country_by_code(
                            flight["departure_airport_icao"],
                            flight["arrival_airport_icao"],
                            flight["departure_airport_iata"],
                            flight["arrival_airport_iata"],
                        )
                        #breakpoint()
                        # Append the city and country  names to the flight dictionary
                        flight["departure_city"] = departure['City']
                        flight["departure_country"] = departure['Country']
                        flight["arrival_city"] = arrival['City']
                        flight["arrival_country"] = arrival['Country']
                        flight['route'] = f"{flight['departure_city']} - {flight['arrival_city']}"
                        # Check if the departure and arrival cities are in Europe
                        if flight["departure_country"] in european_countries and flight["arrival_country"] in european_countries:
                            # Convert the flight dictionary to a Flight object
                            flight_obj = Flight(**flight)
                            response.append(flight_obj)
                        else:
                            continue
            else :
                response = data
                logger.debug(f"No flights found, response: {response}")
                
        else:
            # Print an error message if the request was not successful
            print(f"Failed to retrieve data: {data.status_code}")

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
        openflights_airports = pd.read_csv(get_airports_path())
        # breakpoint()
        # return the DataFrame
        return openflights_airports
    
    def get_airport_coordinates_by_code(
        self,
        departure_airport_icao: str,
        arrival_airport_icao: str,
        departure_airport_iata: str,
        arrival_airport_iata: str,
    ) -> List[Dict[str, float]]:
        """
        Retrieves the coordinates (latitude and longitude) of the departure and arrival airports
        based on their ICAO and IATA codes from the airport database.

        Args:
            departure_airport_icao (str): The ICAO code of the departure airport.
            arrival_airport_icao (str): The ICAO code of the arrival airport.
            departure_airport_iata (str): The IATA code of the departure airport.
            arrival_airport_iata (str): The IATA code of the arrival airport.

        Returns:
            List[Dict[str, float]]: Coordinates (latitude and longitude) for departure and arrival airports.
        """
        def get_coordinates(code: str, column: str) -> Dict[str, float]:
            result = self.airports_database[self.airports_database[column] == code]
            if not result.empty:
                latitude = result["Latitude"].values[0]
                longitude = result["Longitude"].values[0]
                return {"Latitude": float(latitude), "Longitude": float(longitude)}
            return {"Latitude": None, "Longitude": None}

        # Use ICAO if available, otherwise use IATA
        departure_coords = (
            get_coordinates(departure_airport_icao, "ICAO")
            if departure_airport_icao
            else get_coordinates(departure_airport_iata, "IATA")
        )
        arrival_coords = (
            get_coordinates(arrival_airport_icao, "ICAO")
            if arrival_airport_icao
            else get_coordinates(arrival_airport_iata, "IATA")
        )

        return [departure_coords, arrival_coords]

    #Function to convert altitude in meters to flight level
    def altitude_to_flight_level(
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
    
    # Get the city and country names based on the ICAO and IATA codes
    def get_city_and_country_by_code(
        self,
        departure_airport_icao: str,
        arrival_airport_icao: str,
        departure_airport_iata: str,
        arrival_airport_iata: str,
    ) -> List[Dict[str, str]]:
        """
        Takes in the departure and arrival airport ICAO & IATA codes and returns the city and country names
        by searching and retrieving the equivalent city and country names in the openflights airports
        databases.

        Args:
            departure_airport_icao (str): the ICAO code of the departure airport
            arrival_airport_icao (str): the ICAO code of the arrival airport
            departure_airport_iata (str): the IATA code of the departure airport
            arrival_airport_iata (str): the IATA code of the arrival airport

        Returns:
            List[Dict[str, str]]: City and country names for departure and arrival airports
        """
        # Function to get city and country from ICAO or IATA code
        def get_city_and_country(code: str, column: str) -> Dict[str, str]:
            result = self.airports_database[self.airports_database[column] == code]
            if not result.empty:
                city_value = result["City"].values[0]
                country_value = result["Country"].values[0]
                return {
                    "City": str(city_value).strip() if pd.notna(city_value) else "Unknown",
                    "Country": str(country_value).strip() if pd.notna(country_value) else "Unknown"
                }
            return {"City": "Unknown", "Country": "Unknown"}

        # Use ICAO if available, otherwise use IATA
        departure_info = (
            get_city_and_country(departure_airport_icao, "ICAO")
            if departure_airport_icao
            else get_city_and_country(departure_airport_iata, "IATA")
        )
        arrival_info = (
            get_city_and_country(arrival_airport_icao, "ICAO")
            if arrival_airport_icao
            else get_city_and_country(arrival_airport_iata, "IATA")
        )

        return [departure_info, arrival_info]


    # Check if we are done fetching data
    def _is_done(self) -> bool:
        # We are never done fetching live data
        return self.is_done


# Create an instance of the LiveFlights class
# live_flights = LiveFlights()
# Get the live flights data
# response = live_flights.get_flights()
# Print the response
# from pprint import pprint
# pprint(response)
# airports = live_flights._airports_database_to_dataframe()
