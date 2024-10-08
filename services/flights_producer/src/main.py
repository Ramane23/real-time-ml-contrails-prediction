from typing import List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
from quixstreams import Application
import time
from src.config import config
from src.aviation_edge_api.live_flights import LiveFlights
from src.aviation_edge_api.historical_flights import historicalFlights
from src.aviation_edge_api.flight import Flight



def produce_flights(
    kafka_broker_address: str,
    kafka_topic_name: str,
    live_or_historical: str,
) -> None:
    """
    Reads flights data from the Aviation Edge API and produces it to a Kafka topic

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_topic (str): The name of the Kafka topic
    Returns:
        None
    """
    # Intiating the application
    app = Application(broker_address=kafka_broker_address)
    # breakpoint()

    # the topic where we will save the flights data
    topic = app.topic(
        name=kafka_topic_name, 
        value_serializer="json",
        )

    logger.info("subscribing to aviation edge API to fetch data")

    if live_or_historical == "live":
        # Create an instance of the LiveFlights class to fetch live data
        aviation_edge_api = LiveFlights()
    else:
        # Create an instance of the historicalFlights class to fetch historical data
        aviation_edge_api = historicalFlights()
        # last_n_days=last_n_days,
        # n_threads=1,  # len(product_ids),
        # cache_dir=config.cache_dir_historical_data,

    logger.info("Creating the flights producer...")

    # Create a Producer instance
    with app.get_producer() as producer:
        while True:
            # check if we are done fetching historical data
            if aviation_edge_api._is_done():
                logger.info("Done fetching historical data")
                break

            # Get the flights from the Aviation Edge API
            flights: List[Flight] = aviation_edge_api.get_flights()
            if flights != {'error': 'No Record Found', 'success': False}:
                # breakpoint()
                # Challenge 1: Send a heartbeat to Prometheus to check the service is alive
                # Challenge 2: Send an event with trade latency to Prometheus, to monitor the trade latency
                # breakpoint()
                # producing the live data to the kafka topic
                for flight in flights:
                    # Serialize an event using the defined Topic
                    message = topic.serialize(
                        key=str(flight.flight_id),  # Convert flight_id to string
                        value=flight.model_dump(),
                    )

                    # Produce a message into the Kafka topic
                    producer.produce(
                        topic=topic.name,
                        value=message.value,
                        key=message.key,
                    )

                    logger.debug(f"{flight.model_dump()}")
                    # breakpoint()
                logger.info(f"Produced {len(flights)} flights to Kafka topic {topic.name}")
                # Wait for 1 minute before fetching the next batch of flights
                time.sleep(60)  # 1 minute = 60 seconds
            else:
                break

if __name__ == "__main__":
    logger.debug("Configuration:")
    logger.debug(config.model_dump())
    try:
        produce_flights(
            kafka_broker_address=config.kafka_broker_address,
            kafka_topic_name=config.kafka_topic_name,
            live_or_historical=config.live_or_historical,
        )
    except KeyboardInterrupt:
        logger.info("Exiting...")
