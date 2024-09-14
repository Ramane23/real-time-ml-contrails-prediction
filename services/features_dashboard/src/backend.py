from loguru import logger
from typing import List, Dict
import time
import hopsworks
from hsfs.client.exceptions import FeatureStoreException
import pandas as pd

from tools.infrastructures.hopsworks import HopsworksFlightsWriter, HopsworksFlightsReader
from src.config import config

logger.debug('Backend module loaded')
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


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.online and args.offline:
        raise ValueError('You cannot pass both --online and --offline')    
    online_or_offline = 'offline' if args.offline else 'online'
    
    from loguru import logger
    data = get_features_from_the_store(online_or_offline)
    
    logger.debug(f'Received {len(data)} rows of data from the Feature Store')

    logger.debug(data.head())
