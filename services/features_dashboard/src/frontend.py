import streamlit as st
import pandas as pd
from loguru import logger
from streamlit_folium import st_folium  # To render Folium maps in Streamlit
from backend import GetFeaturesFromTheStore  # Function to fetch flight_data from the store
from config import config  # Configuration settings
from plot import FlightPlotter  # FlightPlotter class
from streamlit_autorefresh import st_autorefresh  # Import st_autorefresh

# Instantiate the GetFeaturesFromTheStore class
get_features_from_the_store = GetFeaturesFromTheStore()

# Function to fetch flight data from the store
@st.cache_data(ttl=60)  # Cache the data to avoid unnecessary loads within the TTL (time-to-live)
def get_flight_data():
    try:
        return get_features_from_the_store.get_features(live_or_historical=config.live_or_historical)
    except Exception as e:
        logger.error(f"Error occurred while fetching flight data: {e}")
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame()

# Instantiate the FlightPlotter class 
def plot_flight_map():
    flight_data = get_flight_data()

    # Check if flight_data is empty, then skip plotting
    if flight_data.empty:
        st.warning(f"No flight_data available in the store.")
    else:
        # Instantiate the FlightPlotter class with new data
        flight_plotter = FlightPlotter(
            flight_data,
            map_center=[48.8566, 2.3522],
            zoom_start=5
        )
        # Generate the map
        map_obj = flight_plotter.plot_flights()
        
        # Display the map in the Streamlit dashboard
        st_folium(map_obj, width=725)

# Define the title for the Streamlit application
st.write("""
# Flight Tracking Dashboard
Real-time flight tracking on the 20 busiest intra-European routes.
""")

# Adding a selectbox to the sidebar to switch between the online store and the offline store
online_or_offline = st.sidebar.selectbox(
    'Select the store',  # Label for the selectbox
    ('online', 'offline')  # Options available in the selectbox
)

# Adding a slider to control the refresh rate of the map
refresh_rate = st.sidebar.slider(
    'Refresh rate (seconds)',
    min_value=60, max_value=1800, value=600, step=1
)

# Adding a container for the map
with st.container():
    plot_flight_map()

# Use st_autorefresh to refresh the map at regular intervals based on the slider value
st_autorefresh(interval=refresh_rate * 1000, key="flight_map_refresh")
