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


class FlightWeather(BaseModel):
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
    temperature_C: float
    pressure_hPa: float
    wind_speed_u_ms: float
    wind_speed_v_ms: float
    geopotential_height_m: float
    relative_humidity_percent: float
    total_cloud_cover_octas: float
    high_cloud_cover_octas: float
    specific_humidity_kg_kg: float
    prob_contrails_percent: float

    def to_dict(self):
        """
        Convert the FlightWeather instance to a dictionary.
        """
        return {
            "aircraft_iata_code": self.aircraft_iata_code,
            "aircraft_icao_code": self.aircraft_icao_code,
            "airline_iata_code": self.airline_iata_code,
            "airline_icao_code": self.airline_icao_code,
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
            "temperature_C": self.temperature_C,
            "pressure_hPa": self.pressure_hPa,
            "wind_speed_u_ms": self.wind_speed_u_ms,
            "wind_speed_v_ms": self.wind_speed_v_ms,
            "geopotential_height_m": self.geopotential_height_m,
            "relative_humidity_percent": self.relative_humidity_percent,
            "total_cloud_cover_octas": self.total_cloud_cover_octas,
            "high_cloud_cover_octas": self.high_cloud_cover_octas,
            "specific_humidity_kg_kg": self.specific_humidity_kg_kg,
            "prob_contrails_percent": self.prob_contrails_percent
        }