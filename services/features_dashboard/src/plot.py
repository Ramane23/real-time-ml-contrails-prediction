import folium
import numpy as np
import pandas as pd
from typing import List, Tuple
from geopy.distance import geodesic

class FlightPlotter:
    
    def __init__(self, dataframe: pd.DataFrame, map_center: List[float] = [54.5260, 15.2551], zoom_start: int = 5):
        """
        Initialize the flight plotter with a dataframe and set up the map.
        The map will focus on Europe.
        
        Args:
        - dataframe: pd.DataFrame, containing flight information.
        - map_center: List of latitude, longitude for the map's center (default to central Europe).
        - zoom_start: Initial zoom level of the map (default is 5).
        """
        self.df = dataframe.copy()

        # Convert relevant fields to float
        self.df['departure_airport_lat'] = pd.to_numeric(self.df['departure_airport_lat'], errors='coerce')
        self.df['departure_airport_long'] = pd.to_numeric(self.df['departure_airport_long'], errors='coerce')
        self.df['arrival_airport_lat'] = pd.to_numeric(self.df['arrival_airport_lat'], errors='coerce')
        self.df['arrival_airport_long'] = pd.to_numeric(self.df['arrival_airport_long'], errors='coerce')
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')

        # Initialize the map, focusing on Europe
        self.map = folium.Map(
            location=map_center,
            zoom_start=zoom_start,
            tiles="OpenStreetMap",  # Using a basic tile set for easier debugging
            attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
        )

        # Optional: Set map bounds to restrict view to Europe (approximate coordinates for Europe)
        self.map.fit_bounds([[34.0, -25.0], [71.0, 45.0]])  # Boundaries around Europe

    def add_flight_point(self, lat: float, lon: float, flight: pd.Series):
        """
        Adds the actual flight point on the map based on its latitude and longitude,
        and displays relevant flight information in the popup.

        Args:
        - lat: Latitude of the flight.
        - lon: Longitude of the flight.
        - flight: A Pandas Series representing the flight row, which contains flight details.
        """
        # Safely access flight details, handling any missing fields
        popup_text = (
            f"<b>Flight ID:</b> {flight.get('flight_id', 'N/A')}<br>"
            f"<b>Airline Name:</b> {flight.get('airline_name', 'N/A')}<br>"
            f"<b>Route:</b> {flight.get('route', 'N/A')}<br>"
            f"<b>Altitude:</b> {flight.get('altitude', 'N/A')}<br>"
            f"<b>Flight Level:</b> {flight.get('flight_level', 'N/A')}<br>"
            f"<b>Latitude:</b> {flight.get('latitude', 'N/A')}<br>"
            f"<b>Longitude:</b> {flight.get('longitude', 'N/A')}<br>"
            f"<b>Temperature (C):</b> {flight.get('temperature_C', 'N/A')}<br>"
            f"<b>Wind Speed (m/s):</b> {flight.get('wind_speed_ms', 'N/A')}<br>"
            f"<b>Specific Humidity:</b> {flight.get('specific_humidity_kg_kg', 'N/A')}<br>"
            f"<b>Geopotential Height:</b> {flight.get('geopotential_height_m', 'N/A')}"
        )
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='black',
            fill=True,
            fill_color='black',
            popup=popup_text
        ).add_to(self.map)

    
    def add_bezier_curve(self, dep_lat: float, dep_lon: float, arr_lat: float, arr_lon: float, control_point: List[float], color="blue"):
        """
        Adds a curved Bezier route between two points using a control point for the curvature.
        """
        # Generate the Bezier curve points
        t_values = np.linspace(0, 1, 100)
        route = [
            [
                (1 - t)**2 * dep_lat + 2 * (1 - t) * t * control_point[0] + t**2 * arr_lat,
                (1 - t)**2 * dep_lon + 2 * (1 - t) * t * control_point[1] + t**2 * arr_lon
            ]
            for t in t_values
        ]
        folium.PolyLine(route, color=color, weight=2.5, opacity=0.7).add_to(self.map)

        return route

    def project_point_on_curve(self, curve: List[Tuple[float, float]], point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Project a flight point onto the closest point on the Bezier curve.
        
        Args:
        - curve: List of (latitude, longitude) points on the Bezier curve.
        - point: Tuple of (latitude, longitude) of the flight point.

        Returns:
        - The closest point (latitude, longitude) on the curve to the flight point.
        """
        lat, lon = point

        # Compute the closest point on the Bezier curve
        closest_point = min(curve, key=lambda curve_point: geodesic(curve_point, (lat, lon)).meters)
        
        return closest_point

    def plot_flights(self):
        """
        Plot the flights on the map with the Bezier routes and projected flight positions.
        """
        for _, flight in self.df.iterrows():
            dep_lat = flight['departure_airport_lat']
            dep_lon = flight['departure_airport_long']
            arr_lat = flight['arrival_airport_lat']
            arr_lon = flight['arrival_airport_long']
            lat = flight['latitude']
            lon = flight['longitude']
            flight_id = flight['flight_id']

            # Generate a control point for the Bezier curve (midpoint for simplicity)
            control_point = [(dep_lat + arr_lat) / 2, (dep_lon + arr_lon) / 2 + 2]  # Adjust the control point for curvature

            # Add the Bezier route and get the curve points
            curve_points = self.add_bezier_curve(dep_lat, dep_lon, arr_lat, arr_lon, control_point)

            # Project the actual flight point onto the Bezier curve
            projected_point = self.project_point_on_curve(curve_points, (lat, lon))

            # Plot the projected flight location
            self.add_flight_point(projected_point[0], projected_point[1], flight)

        return self.map
