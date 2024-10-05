
def get_model_name(product_id: str) -> str:
    """
    Returns the model registry name for the given product_id.

    Args:
        - product_id: the product_id of the model we want to fetch
    
    Returns:
        - str: the model registry name for the given product_id
    """
    return f'{product_id.replace("/","_")}_price_change_predictor'