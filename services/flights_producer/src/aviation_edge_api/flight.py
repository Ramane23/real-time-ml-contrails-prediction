from pydantic import BaseModel


class Flight(BaseModel):
    """
    A class to represent a flight using pydantic BaseModel.
    """

    aircraft_iata_code: str
    aircraft_icao_code: str
    airline_iata_code: str
    airline_icao_code: str
    altitude: float
    flight_level: str
    arrival_airport_iata: str
    arrival_airport_icao: str
    arrival_city: str
    current_flight_time: int
    departure_airport_iata: str
    departure_airport_icao: str
    departure_city: str
    direction: float
    flight_icao_number: str
    flight_number: str
    flight_id: str
    flight_status: str
    horizontal_speed: float
    isGround: bool
    latitude: float
    longitude: float
    vertical_speed: float
    departure_country: str
    arrival_country: str
    route: str
    departure_airport_coords : dict
    arrival_airport_coords : dict
    
