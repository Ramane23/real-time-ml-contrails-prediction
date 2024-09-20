import requests
from typing import List, Dict
from src.config import config
import pandas as pd
from datetime import datetime
from loguru import logger

from src.aviation_edge_api.flight import Flight
from src.paths import get_airports_path, get_airlines_path , get_aircrafts_path


class historicalFlights:
    # getting all the live flights
    url = f"https://aviation-edge.com/v2/public/flights?key={config.aviation_edge_api_key}"

    # getting all the live flights within a circle area based on lat and lng values and radius as the distance
    # url = f'https://aviation-edge.com/v2/public/flights?key={config.aviation_edge_api_key}&lat={config.latitude}&lng={config.longititude}&distance={config.distance}&arrIata=LHR'

    def __init__(self) -> None:
        self.is_done: bool = False
        self.airports_database = self._airports_database_to_dataframe()
        self.airlines_database = self._airlines_database_to_dataframe()
        self.aircrafts_database = self._aircrafts_database_to_dataframe()
        self.days: float  = config.days
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
        #Define a list of European countries
        #european_countries = [
                # "Albania",
                # "Andorra",
                # "Armenia",
                # "Austria",
                # "Azerbaijan",
                # "Belarus",
                # "Belgium",
                # "Bosnia and Herzegovina",
                # "Bulgaria",
                # "Croatia",
                # "Cyprus",
                # "Czech Republic",
                # "Denmark",
                # "Estonia",
                # "Finland",
                # "France",
                # "Georgia",
                # "Germany",
                # "Greece",
                # "Hungary",
                # "Iceland",
                # "Ireland",
                # "Italy",
                # "Kazakhstan",
                # "Kosovo",
                # "Latvia",
                # "Liechtenstein",
                # "Lithuania",
                # "Luxembourg",
                # "Malta",
                # "Moldova",
                # "Monaco",
                # "Montenegro",
                # "Netherlands",
                # "North Macedonia",
                # "Norway",
                # "Poland",
                # "Portugal",
                # "Romania",
                # "Russia",
                # "San Marino",
                # "Serbia",
                # "Slovakia",
                # "Slovenia",
                # "Spain",
                # "Sweden",
                # "Switzerland",
                # "Turkey",
                # "Ukraine",
                # "United Kingdom",
                # "Vatican City"
        # ]
        #20 busiest European routes according WorldAtlas and Wikipedia
        busiest_european_routes_cities = [
            "Toulouse - Paris",
            "Madrid - Barcelona",
            "Nice - Paris",
            "Catania - Rome",
            "Berlin - Munich",
            "Oslo - Trondheim",
            "Frankfurt - Berlin",
            "Oslo - Bergen",
            "Munich - Hamburg",
            "London - Dublin",
            "Barcelona - Palma de Mallorca",
            "Paris - Marseille",
            "Stockholm - Gothenburg",
            "Athens - Thessaloniki",
            "Vienna - Zurich",
            "Milan - Rome",
            "Helsinki - Oulu",
            "Madrid - Malaga",
            "London - Amsterdam",
            "Paris - Lyon"
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
                        #Add the aircraft true airspeed to the flight dictionary
                        flight['true_airspeed_ms'] = self.true_airspeed(flight['horizontal_speed'], flight['vertical_speed'])
                        #breakpoint()
                        #Add the aircraft mass MTOW to the flight dictionary
                        flight['aircraft_mtow_kg'] = self.get_aircraft_weight_mtow(flight['aircraft_icao_code'])
                        #Add the aircraft mass MALW to the flight dictionary
                        flight['aircraft_malw_kg'] = self.get_aircraft_weight_malw(flight['aircraft_icao_code'])
                        #Add the aircraft engine class to the flight dictionary
                        flight['aircraft_engine_class'] = self.get_aircraft_engine_class(flight['aircraft_icao_code'])
                        #Add the number of engines to the flight dictionary
                        flight['aircraft_num_engines'] = self.get_aircraft_num_engines(flight['aircraft_icao_code'])
                        #Add the airline name to the flight dictionary
                        flight['airline_name'] = self.get_airline_name_by_code(
                            flight["airline_iata_code"],
                            flight["airline_icao_code"],
                        )
                        #Add flight level to the flight dictionary
                        flight['flight_level'] = f"FL{self.altitude_to_flight_level(flight['altitude'])}"
                        #Add the departures and arrivals coordinates to the flight dictionary
                        departure_coords, arrival_coords = self.get_airport_coordinates_by_code(
                            flight["departure_airport_icao"],
                            flight["arrival_airport_icao"],
                            flight["departure_airport_iata"],
                            flight["arrival_airport_iata"],
                        )
                        #check if the coordinates are not None
                        flight["departure_airport_lat"] = departure_coords['Latitude']
                        flight["departure_airport_long"] = departure_coords['Longitude']
                        flight["arrival_airport_lat"] = arrival_coords['Latitude']
                        flight["arrival_airport_long"] = arrival_coords['Longitude']
                        #breakpoint()
                        
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
                        #TODO : 
                        #1- Add aircraft mass, true airspeed and Mach number
                        #2- Add, flight distance, load factors
                        # Check if the route is one of the busiest European routes
                        if flight['route'] in busiest_european_routes_cities:
                            # Convert the flight dictionary to a Flight object
                            #breakpoint()
                            flight_obj = Flight(**flight)
                            #breakpoint()
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

    #create a function to get the aircrafts engine class from the FAA aircrafts characteristics database
    def get_aircraft_engine_class(self, aircraft_icao_code: str) -> str:
        """
        Gets the aircraft's engine class (Physical_Class_Engine) from the aircrafts database based on ICAO code.

        Args:
            aircraft_icao_code (str): The ICAO code of the aircraft.

        Returns:
            str: The engine class (e.g., Jet, Turboprop, Piston). Returns None if ICAO code is not found.
        """
        # Ensure the database is loaded
        if self.aircrafts_database is None:
            self._aircrafts_database_to_dataframe()
        
        # Filter the DataFrame for the specified ICAO code
        row = self.aircrafts_database[self.aircrafts_database['ICAO_Code'] == aircraft_icao_code]
        
        if not row.empty:
            # Extract the engine class (Physical_Class_Engine)
            engine_class = row['Physical_Class_Engine'].values[0]
            return engine_class
        else:
            return None  # Return None if no matching ICAO code is found
        
    #create a function to get the aircrafts number of engines from the FAA aircrafts characteristics database
    def get_aircraft_num_engines(
        self, 
        aircraft_icao_code: str
        ) -> int:
        """
        Gets the number of engines (Num_Engines) from the aircrafts database based on ICAO code.

        Args:
            aircraft_icao_code (str): The ICAO code of the aircraft.

        Returns:
            int: The number of engines. Returns None if ICAO code is not found.
        """
        # Ensure the database is loaded
        if self.aircrafts_database is None:
            self._aircrafts_database_to_dataframe()
        
        # Filter the DataFrame for the specified ICAO code
        row = self.aircrafts_database[self.aircrafts_database['ICAO_Code'] == aircraft_icao_code]
        
        if not row.empty:
            # Extract the number of engines (Num_Engines)
            num_engines = row['Num_Engines'].values[0]
            return num_engines
        else:
            return None  # Return None if no matching ICAO code is found
    #create a function to compute the true airspeed TAS from the ground speed and the vertical speed
    def true_airspeed(
        self, 
        ground_speed: float, 
        vertical_speed: float
        ) -> float:
        """
        Compute the true airspeed (TAS) in m/s from the ground speed and vertical speed in m/s.
        
        Args:
            ground_speed (float): The ground speed in meters per second (m/s).
            vertical_speed (float): The vertical speed in meters per second (m/s).
        
        Returns:
            float: The true airspeed (TAS) in meters per second (m/s).
        """
        # Use Pythagorean theorem to calculate the TAS
        tas = (ground_speed**2 + vertical_speed**2) ** 0.5
        
        return tas
            
    #create a function to get the aircrafts characteristics database to pandas DataFrame
    def _aircrafts_database_to_dataframe(self) -> None:
        """Downloads the FAA aircrafts characteristics database and loads it into a pandas DataFrame.

        Args:
            None

        Returns:
            None
        """
        # load the FAA aircrafts characteristics database into a pandas DataFrame
        # return the DataFrame
        aircrafts = pd.read_excel(get_aircrafts_path())
        #breakpoint()
        # return the DataFrame
        return aircrafts
    
    #Function to get the aircrafts weight MALW from the FAA aircrafts characteristics database
    def get_aircraft_weight_malw(
        self, 
        aircraft_icao_code: str
        ) -> float:
        """
        Gets the aircraft's MALW (Maximum Allowed Landing Weight) from the aircrafts database based on ICAO code.

        Args:
            aircraft_icao_code (str): The ICAO code of the aircraft.

        Returns:
            float: The MALW of the aircraft in kilograms. Returns None if ICAO code is not found.
        """
        # Ensure the database is loaded
        if self.aircrafts_database is None:
            self._aircrafts_database_to_dataframe()
        
        # Filter the DataFrame for the specified ICAO code
        row = self.aircrafts_database[self.aircrafts_database['ICAO_Code'] == aircraft_icao_code]
        
        if not row.empty:
            # Extract the MALW in pounds and convert to kilograms
            malw_lb = row['MALW_lb'].values[0]
            malw_kg = malw_lb * 0.453592  # Conversion factor from lb to kg
            return malw_kg
        else:
            return None  # Return None if no matching ICAO code is found
        
    #Function to get the aircrafts weight MTOW from the FAA aircrafts characteristics database
    def get_aircraft_weight_mtow(
        self, 
        aircraft_icao_code: str
        ) -> float:
        """
        Gets the aircraft's MTOW (Maximum Takeoff Weight) in kilograms from the aircrafts database based on ICAO code.

        Args:
            aircraft_icao_code (str): The ICAO code of the aircraft.

        Returns:
            float: The MTOW of the aircraft in kilograms. Returns None if ICAO code is not found.
        """
        # Ensure the database is loaded
        if self.aircrafts_database is None:
            self._aircrafts_database_to_dataframe()
        
        # Filter the DataFrame for the specified ICAO code
        row = self.aircrafts_database[self.aircrafts_database['ICAO_Code'] == aircraft_icao_code]
        
        if not row.empty:
            # Extract the MTOW in pounds and convert to kilograms
            mtow_lb = row['MTOW_lb'].values[0]
            mtow_kg = mtow_lb * 0.453592  # Conversion factor from lb to kg
            return mtow_kg
        else:
            return None  # Return None if no matching ICAO code is found
        
    #Function to get the airports csv database to pandas DataFrame
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
                latitude = round(result["Latitude"].values[0],7)
                longitude = round(result["Longitude"].values[0],7)
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
        #breakpoint()
        # return the DataFrame
        return openflights_airports
    
    #Function to get airlines csv database to pandas DataFrame
    def _airlines_database_to_dataframe(self) -> None:
        """Downloads the openflights airports database and loads it into a pandas DataFrame.

        Args:
            None

        Returns:
            None
        """
        # load the openflights airports database into a pandas DataFrame
        # return the DataFrame
        airlines = pd.read_csv(get_airlines_path())
        #breakpoint()
        # return the DataFrame
        return airlines
    
    #Function to get the airline name based on the ICAO and IATA codes
    def get_airline_name_by_code(
        self,
        airline_iata: str = None,
        airline_icao: str = None
        ) -> str:
        """
        Retrieves the airline name using the IATA or ICAO code from the airlines database.

        Args:
            airline_iata (str, optional): The IATA code of the airline.
            airline_icao (str, optional): The ICAO code of the airline.

        Returns:
            str: The name of the airline if found, otherwise 'Unknown'.
        """
        if airline_iata:
            result = self.airlines_database[self.airlines_database["IATA"] == airline_iata]
        elif airline_icao:
            result = self.airlines_database[self.airlines_database["ICAO"] == airline_icao]
        else:
            return "Unknown"

        if not result.empty:
            return result["Name"].values[0].strip()
        
        return "Unknown"
    
    #Function to get the city and country names based on the ICAO and IATA codes
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

    # create a helper method to convert in seconds
    def _convert_to_seconds(self, days: int) -> int:
        """Convert days to seconds"""
        return days * 24 * 60 * 60

    def _is_done(self) -> bool:
        """Check if the fetching process has reached the specified number of days."""

        # Get the current date and time
        current_date = pd.Timestamp.now()

        # Get the start date in datetime
        start_date = pd.Timestamp(self.start_date)

        # Calculate the difference in time between the current and start date in seconds
        time_diff = (current_date - start_date).total_seconds() 

        # If the difference in milliseconds is greater than or equal to the specified number of days in milliseconds, mark as done
        if time_diff >= self._convert_to_seconds(self.days):
            # Set is_done to True
            self.is_done = True
        else:
            # Set is_done to False
            self.is_done = False

        return self.is_done

if __name__ == "__main__":
    # Create an instance of the LiveFlights class
    historical_flights = historicalFlights()
    # Get the live flights data
    response = historical_flights.get_flights()
    # Print the response
    from pprint import pprint
    pprint(response)
    #airports = historical_flights._airports_database_to_dataframe()
    #airports.head()

