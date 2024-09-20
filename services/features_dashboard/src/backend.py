from loguru import logger
from typing import List, Dict
import pandas as pd

from hopsworks_fs import  GetFeaturesFromTheStore
from config import config

if __name__ == '__main__':
        
    logger.debug('Backend module loaded')
    logger.debug(f'Config: {config.model_dump()}')
    
    #Instantiate the GetFeaturesFromTheStore class
    get_features_from_the_store = GetFeaturesFromTheStore()
        
    #retrieve the flights data from the hopsworks feature store
    flights_data = get_features_from_the_store.get_features(live_or_historical=config.live_or_historical)
    #breakpoint()     
    if flights_data is None or flights_data.empty:
        logger.error("No data retrieved from the Feature Store. The DataFrame is empty or None.")
    else:
        logger.info(f"Retrieved {len(flights_data)} rows from the Feature Store.")
        logger.info(flights_data.head())
            
