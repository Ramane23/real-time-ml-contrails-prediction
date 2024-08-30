from loguru import logger
from quixstreams import Application
from typing import Any, List, Optional, Tuple
from datetime import datetime

from src.config import config
from src.meteomatics_api.flight import Flight, FlightWeather
from src.meteomatics_api.live_flights import AddFlightsWeather


def custom_ts_extractor(
    value: Any,    # Represents the message value, here a flight object
    headers: Optional[List[Tuple[str, bytes]]],
    timestamp: float,  # This is the Kafka-generated timestamp that is passed to the function
    timestamp_type,  #: TimestampType, This is the type of the timestamp (e.g., creation time, log time) as defined by Kafka
) -> int:
    """
    Specifying a custom timestamp extractor to use the timestamp from the message payload
    instead of the Kafka timestamp.

    We want to use the `current_flight_time` field from the message value and convert it 
    into a Unix timestamp that Kafka can use.
    
    See the Quix Streams documentation here
    https://quix.io/docs/quix-streams/windowing.html#extracting-timestamps-from-messages
    """
    flight_time_seconds = value['current_flight_time']
    # Convert flight time from seconds to a datetime object
    flight_time_datetime = datetime.utcfromtimestamp(flight_time_seconds)
    # Convert the datetime object back to Unix timestamp in milliseconds
    return int(flight_time_datetime.timestamp() * 1000)

#Function to read flight data from a kafka topic, add weather data and write to another kafka topic
def produce_flight_weather_data(
    
    kafka_broker_address: str,
    kafka_input_topic_name: str,
    kafka_consumer_group: str,
    kafka_output_topic_name: str,
) -> None:
    """
    This function reads flight data from a Kafka topic, adds weather data to it, 
    and writes the enriched data to another Kafka topic.
    
    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_input_topic_name (str): The name of the input Kafka topic
        kafka_output_topic_name (str): The name of the output Kafka topic
        live_or_historical (str): Whether to fetch live or historical flight data
    
    Returns:
        None
    """
    # Initialize a quixstreams application to handle all low level Kafka operations
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='earliest',
        commit_interval=1.0 #commit the consumer offsets every 1 second
    )
    
    # specify input and output topics for this application
    input_topic = app.topic(
        name=kafka_input_topic_name,
        value_serializer='json',
        timestamp_extractor=custom_ts_extractor,
    )
    #breakpoint()
    output_topic = app.topic(name=kafka_output_topic_name, value_serializer='json')
    
    # creating a streaming dataframe
    # to apply transformations on the incoming data
    sdf = app.dataframe(input_topic)
    
    #the values in the topic are considered as dictionaries, we need the values to be of type Flight
    sdf = sdf.apply(Flight.parse_obj)
    
    # Create an instance of the LiveFlightsWeather class
    meteomatics_api = AddFlightsWeather()
        
    #apply the add_weather_inputs method to the incoming data
    #to get the weather data for the flight
    sdf  = sdf.apply(meteomatics_api.get_weather_along_route) 
        
    # let's print the data to the logs
    sdf = sdf.update(logger.info)
        
    #the output of add_weather_inputs is a FlightWeather object, 
    #quixstreams needs to be able to serialize it in json format
    #Convert FlightWeather objects to dictionaries before serialization.
    #first convert the FlightWeather object to a dictionary
    sdf = sdf.apply(lambda fw: fw.to_dict())
        
    #produce the enriched data to the output topic
    sdf = sdf.to_topic(output_topic)
        
    # Let's kick-off the streaming application
    app.run(sdf)
    

if __name__ == "__main__":
    logger.debug("Configuration:")
    logger.debug(config.model_dump())
    logger.info("Starting the enrichment of flights events with weather data...")
    try:
        # Call the function to produce flight weather data
        produce_flight_weather_data(
            kafka_broker_address=config.kafka_broker_address,
            kafka_input_topic_name=config.kafka_input_topic_name,
            kafka_consumer_group=config.kafka_consumer_group,
            kafka_output_topic_name=config.kafka_output_topic_name,
        )
    except KeyboardInterrupt:
        logger.info("Exiting...")