import pygrib
from src.gfs_api.live_weather import AddFlightsWeather

flights_with_weather = AddFlightsWeather()

grbs = pygrib.open(flights_with_weather.download_gfs_data())
count = 0
for grb in grbs:
    count += 1
    print(grb)  # This should print each GRIB message

print(f"Total number of messages in the GRIB file: {count}")