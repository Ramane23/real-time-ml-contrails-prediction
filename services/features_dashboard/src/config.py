from pydantic_settings import BaseSettings


class Config(BaseSettings):
    
    product_id: str
    
    # feature group our feature view reads data from
    feature_group_name: str
    feature_group_version: int

    # feature view name and version
    feature_view_name: str
    feature_view_version: int

    # required to authenticate with Hopsworks API
    hopsworks_project_name: str
    hopsworks_api_key: str

config = Config()