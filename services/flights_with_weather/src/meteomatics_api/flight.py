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
    aircraft_num_engines : Optional[int] = None
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
    


class FlightWeather(BaseModel):
    """
    A class to represent a flight using pydantic BaseModel.
    """

    aircraft_iata_code: str
    aircraft_icao_code: str
    aircraft_mtow_kg : Optional[float] = None
    aircraft_malw_kg : Optional[float] = None
    aircraft_engine_class : Optional[str] = "unknown"
    aircraft_num_engines : Optional[int] = None
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
    latitude: str
    longitude: str
    vertical_speed: float
    true_airspeed_ms: float
    mach_number: float
    departure_country: str
    arrival_country: str
    route: str
    departure_airport_lat : Optional[float] = "unknown"
    departure_airport_long : Optional[float] = "unknown"
    arrival_airport_lat : Optional[float] = "unknown"
    arrival_airport_long : Optional[float] = "unknown"
    temperature_C: float
    pressure_hPa: float
    wind_speed_u_ms: float
    wind_speed_v_ms: float
    wind_speed_ms : float
    geopotential_height_m: float
    relative_humidity_percent: float
    total_cloud_cover_octas: float
    high_cloud_cover_octas: float
    specific_humidity_kg_kg: float
    prob_contrails_percent: float
    global_radiation_W_m2:  float
    
    
    def to_dict(self):
        """
        Convert the FlightWeather instance to a dictionary.
        """
        return {
            "aircraft_iata_code": self.aircraft_iata_code,
            "aircraft_icao_code": self.aircraft_icao_code,
            "aircraft_mtow_kg" : self.aircraft_mtow_kg,
            "aircraft_malw_kg" : self.aircraft_malw_kg,
            "aircraft_engine_class" : self.aircraft_engine_class,
            "aircraft_num_engines" : self.aircraft_num_engines,
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
            "true_airspeed_ms": self.true_airspeed_ms,
            "mach_number": self.mach_number,
            "departure_country": self.departure_country,
            "arrival_country": self.arrival_country,
            "route": self.route,
            "departure_airport_lat": self.departure_airport_lat,
            "arrival_airport_lat": self.arrival_airport_lat,
            "departure_airport_long": self.departure_airport_long,
            "arrival_airport_long": self.arrival_airport_long,
            "temperature_C": self.temperature_C,
            "pressure_hPa": self.pressure_hPa,
            "wind_speed_u_ms": self.wind_speed_u_ms,
            "wind_speed_v_ms": self.wind_speed_v_ms,
            "wind_speed_ms": self.wind_speed_ms,
            "geopotential_height_m": self.geopotential_height_m,
            "relative_humidity_percent": self.relative_humidity_percent,
            "total_cloud_cover_octas": self.total_cloud_cover_octas,
            "high_cloud_cover_octas": self.high_cloud_cover_octas,
            "specific_humidity_kg_kg": self.specific_humidity_kg_kg,
            "prob_contrails_percent": self.prob_contrails_percent,
            "global_radiation_W_m2": self.global_radiation_W_m2,
        }