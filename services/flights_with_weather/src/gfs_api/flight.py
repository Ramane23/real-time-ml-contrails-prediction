from pydantic import BaseModel
from typing import Optional

class Flight(BaseModel):
    """
    A class to represent a flight using pydantic BaseModel.
    """

    aircraft_iata_code: str
    aircraft_icao_code: str
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
    departure_country: str
    arrival_country: str
    route: str
    #departure_airport_alt : float
    #arrival_airport_alt : float
    #departure_airport_long : float
    #arrival_airport_long : float
    


class FlightWeather(BaseModel):
    """
    A class to represent a flight with weather data using pydantic BaseModel.
    """

    # Flight details
    aircraft_iata_code: str
    aircraft_icao_code: str
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
    departure_country: str
    arrival_country: str
    route: str

    # Weather data (with Optional type)
    temperature_C: Optional[float] = None  # Temperature in Celsius
    wind_speed_u_ms: Optional[float] = None  # U component of wind speed
    wind_speed_v_ms: Optional[float] = None  # V component of wind speed
    wind_direction: Optional[float] = None  # Wind direction in degrees
    pressure_hPa: Optional[float] = None  # Atmospheric pressure in hPa
    geopotential_height_m: Optional[float] = None  # Geopotential height in meters
    relative_humidity_percent: Optional[float] = None  # Relative humidity in percent
    specific_humidity_kg_kg: Optional[float] = None  # Specific humidity in kg/kg
    total_cloud_cover_octas: Optional[float] = None  # Total cloud cover in octas
    high_cloud_cover_octas: Optional[float] = None  # High cloud cover in octas
    downward_shortwave_radiation_W_m2: Optional[float] = None  # Downward shortwave radiation flux in W/m^2

    def to_dict(self):
        """
        Convert the FlightWeather instance to a dictionary.
        """
        return {
            # Flight details
            "aircraft_iata_code": self.aircraft_iata_code,
            "aircraft_icao_code": self.aircraft_icao_code,
            "airline_iata_code": self.airline_iata_code,
            "airline_icao_code": self.airline_icao_code,
            "airline_name": self.airline_name,
            "altitude": self.altitude,
            "flight_level": self.flight_level,
            "arrival_airport_iata": self.arrival_airport_iata,
            "arrival_airport_icao": self.arrival_airport_icao,
            "arrival_city": self.arrival_city,
            "current_flight_time": self.current_flight_time,
            "departure_airport_iata": self.departure_airport_iata,
            "departure_airport_icao": self.departure_airport_icao,
            "departure_city": self.departure_city,
            "direction": self.direction,
            "flight_icao_number": self.flight_icao_number,
            "flight_number": self.flight_number,
            "flight_id": self.flight_id,
            "flight_status": self.flight_status,
            "horizontal_speed": self.horizontal_speed,
            "isGround": self.isGround,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "vertical_speed": self.vertical_speed,
            "departure_country": self.departure_country,
            "arrival_country": self.arrival_country,
            "route": self.route,

            # Weather data
            "temperature_C": self.temperature_C,
            "wind_speed_u_ms": self.wind_speed_u_ms,
            "wind_speed_v_ms": self.wind_speed_v_ms,
            "wind_direction": self.wind_direction,
            "pressure_hPa": self.pressure_hPa,
            "geopotential_height_m": self.geopotential_height_m,
            "relative_humidity_percent": self.relative_humidity_percent,
            "specific_humidity_kg_kg": self.specific_humidity_kg_kg,
            "total_cloud_cover_octas": self.total_cloud_cover_octas,
            "high_cloud_cover_octas": self.high_cloud_cover_octas,
            "downward_shortwave_radiation_W_m2": self.downward_shortwave_radiation_W_m2,
        }