from typing import Optional
from quixstreams import Application
import json
from loguru import logger
from datetime import datetime, timezone
import uuid
from time import sleep

from src.config import config
from src.hopsworks_fs import HopsworksFlightsWriter, HopsworksFlightsReader

logger.debug('flights_to_feature_store service is starting...')
logger.debug(f'Config: {config.model_dump()}')

#Instantiate the HopsworksFlightsWriter class
hopsworks_flights_writer = HopsworksFlightsWriter(
    config.feature_group_name, 
    config.feature_group_version, 
    config.hopsworks_project_name, 
    config.hopsworks_api_key
)

#Instantiate the HopsworksFlightsReader class
hopsworks_flights_reader = HopsworksFlightsReader(
    hopsworks_flights_writer.get_feature_store(),
    hopsworks_flights_writer.feature_group_name,
    hopsworks_flights_writer.feature_group_version,
    config.feature_view_name,
    config.feature_view_version
)

#function to get the current UTC timestamp in seconds
def get_current_utc_ts() -> int:
    """
    Get the current UTC timestamp in seconds
    Returns:
        int: the current UTC timestamp in seconds
    """
    return int(datetime.now(timezone.utc).timestamp())

#function to push flight data to the feature store
def flights_to_feature_store(
    kafka_topic_name: str,
    kafka_broker_address: str,
    kafka_consumer_group: str,
    #feature_group_name: str,
    #feature_group_version: int,
    live_or_historical: Optional[str] = "live", #live or historical mode
    buffer_size: Optional[int] = 1, #contains the flights that we want to write to the feature store at once
    #whether to create a new consumer group or not 
    #useful for when we want to retrieve historical data from aan ohlc kafka topic
    create_new_consumer_group: Optional[bool] = False 
    )-> None:
    """
    Stream data from the ohlc Kafka topic to the hopsworks feature store in the specified feature group
    Args:
        kafka_topic_name (str): the name of the Kafka topic where the OHLC data is stored
        kafka_broker_address (str): the address of the Kafka broker
        feature_group_name (str): the name of the feature group
        feature_group_version (int): the version of the feature group
        live_or_historical (str): whether to write the data to the feature store in live or historical mode
            live: the data is written to the online feature store
            historical: the data is written to the offline feature store
        buffer_size (int): the number of messages to buffer before writing to the feature store
    Returns:
        None
    """
    save_every_n_sec: Optional[int] = 60 #force save to feature store every n seconds 
    
    # to force your application to read from the beginning of the topic
    # you need 2 things:
    # 1. Create a unique consumer group name
    # 2. Set the auto_offset_reset to 'earliest' -> offset for this new consuemr group is 0.
    # Which means that when you spin up the `kafka_to_feature_store` service again, it 
    # will re-process all the messages in the topic `kafka_topic`
    if create_new_consumer_group:
        # generate a unique consumer group name using uuid
        kafka_consumer_group = 'historical_flights_with_weather_consumer_' + str(uuid.uuid4())
        logger.debug(f'New consumer group name: {kafka_consumer_group}')

    # breakpoint()
    # Create a new application
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        # we should understand that this is relevant when the consumer group doesn't already exits
        #because if the consumer group already exists, it will start reading from the last offset
        #meaning that it will start reading from the last message that was read by the consumer group
        auto_offset_reset="earliest" if live_or_historical == 'historical' else "latest",
        commit_interval=1.0 #commit the consumer offsets every 1 second  
    )
    logger.info("Application created")
    # let's connect the app to the input topic
    topic = app.topic(name=kafka_topic_name, value_serializer='json')
    #get current UTC in seconds
    last_save_to_feature_store_ts = get_current_utc_ts()
    #initialize the buffer
    buffer = []
    #TODO: handle the case where the buffer is not full and there is nor more expected data to come in 
    # as with the current implementation we may miss the last few messages if the buffer is not full (up to buffer_size-1 messages)
    # Create a consumer and start a polling loop
    with app.get_consumer() as consumer:
        consumer.subscribe(topics=[topic.name])
        logger.info(f"Subscribed to topic {kafka_topic_name}") 
        while True:
            msg = consumer.poll(1) #how much time to wait for a message before skipping to the next iteration
            if msg is None:
                logger.debug(f"No new messages available in the input topic {kafka_topic_name}")
                # No new messages available in the input topic
                #instead of skipping we will check when was the last time we received a message
                #and if it was more than N minutes ago we will push the data to the feature store regardless of the buffer size
                #save_every_n_sec = 10 
                # check how many seconds has passed since the last message was received and compare it to the save_every_n_sec
                #breakpoint()
                if (get_current_utc_ts() - last_save_to_feature_store_ts)>= save_every_n_sec and len(buffer) > 0:
                    logger.debug("Excedeed the timer limit, pushing the data to the feature store")
                    #breakpoint()
                    if live_or_historical == "live":
                        logger.debug("Pushing the live flight data to the online feature store")
                        sleep(0.1) # wait for 0.1 seconds befor pushing the data
                        #push the available data to the feature store
                        hopsworks_flights_writer.push_flight_data_to_feature_store(
                        flight_data=buffer 
                        #feature_group_name=feature_group_name, 
                        #feature_group_version=feature_group_version,
                        #online_or_offline = "online"  if live_or_historical == "live" else "offline"
                        )
                        #breakpoint()
                        #clear the buffer
                        buffer = []
                        
                    elif live_or_historical == "historical":
                        logger.debug("Pushing the historical flight data to the offline feature store")
                        #push the available data to the feature store
                        hopsworks_flights_writer.push_flight_data_to_feature_store(
                        flight_data=buffer 
                        #feature_group_name=feature_group_name, 
                        #feature_group_version=feature_group_version,
                        #online_or_offline = "online"  if live_or_historical == "live" else "offline"
                        )
                        #breakpoint()
                        #clear the buffer
                        buffer = []
                        
                else:
                    #if the last message was received less than save_every_n_sec seconds ago we will skip to the next iteration
                    logger.debug("Timer limit not excedeed, continuing the polling from the input Kafka topic")
                    continue
            elif msg.error():
                logger.info('Kafka error:', msg.error())
                continue
            else:
                # step 1 -> parse the data from the topic into a dictionary
                try:
                    flight = json.loads(msg.value().decode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON: {e}")
                    continue
                #append the data to the buffer
                buffer.append(flight)
                logger.info(f"current buffer length: {len(buffer)}")
                
                #check if the buffer is full
                if len(buffer) >= buffer_size:
                    # step 2 -> store the data in the feature store
                    if live_or_historical == "live":
                        logger.debug("Pushing the live flight data to the online feature store")
                        sleep(0.1) # wait for 0.1 seconds befor pushing the data
                        #push the available data to the feature store
                        hopsworks_flights_writer.push_flight_data_to_feature_store(
                        flight_data=buffer 
                        #feature_group_name=feature_group_name, 
                        #feature_group_version=feature_group_version,
                        #online_or_offline = "online"  if live_or_historical == "live" else "offline"
                        )
                        #breakpoint()
                        #clear the buffer
                        buffer = []
                    elif live_or_historical == "historical":
                        logger.debug("Pushing the historical flight data to the offline feature store")
                        #push the available data to the feature store
                        hopsworks_flights_writer.push_flight_data_to_feature_store(
                        flight_data=buffer 
                        #feature_group_name=feature_group_name, 
                        #feature_group_version=feature_group_version,
                        #online_or_offline = "online"  if live_or_historical == "live" else "offline"
                        )
                        #breakpoint()
                        #clear the buffer
                        buffer = []
                # update the last_save_to_feature_store_ts
                last_save_to_feature_store_ts = get_current_utc_ts()

if __name__ == "__main__":
    try:
        flights_to_feature_store(
            kafka_topic_name=config.kafka_topic_name,
            kafka_broker_address=config.kafka_broker_address,
            kafka_consumer_group=config.kafka_consumer_group,
            #feature_group_name=config.feature_group_name,
            #feature_group_version=config.feature_group_version,
            live_or_historical=config.live_or_historical,
            buffer_size=config.buffer_size,
            create_new_consumer_group=config.create_new_consumer_group
        )
    except KeyboardInterrupt:
            logger.info("Stopping the storing...")         
            
