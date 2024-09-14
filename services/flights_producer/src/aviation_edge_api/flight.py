from pydantic import BaseModel
from  typing import Optional


class Flight(BaseModel):
    """
    A class to represent a flight using pydantic BaseModel.
    """

    aircraft_iata_code: str
    aircraft_icao_code: str
    aircraft_mtow_kg : Optional[float] = None
    aircraft_malw_kg : Optional[float] = None
    aircraft_engine_class : Optional[str] = "unknown"
    aircraft_num_engines : Optional[float] = None
    airline_iata_code: str
    airline_icao_code: str
    airline_name: str
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
    true_airspeed_ms: float
    departure_country: str
    arrival_country: str
    route: str
    departure_airport_lat : Optional[float] = "unknown"
    departure_airport_long : Optional[float] = "unknown"
    arrival_airport_lat : Optional[float] = "unknown"
    arrival_airport_long : Optional[float] = "unknown"
    
   