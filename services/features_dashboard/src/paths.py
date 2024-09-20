from pathlib import Path

# Create a function to return the path of the aircraft icon 
def get_aircraft_icon_path():
    """
    This function returns the path of the aircraft icon as a string.
    """
    # Get the parent directory of the current directory two levels up
    current_dir = Path(__file__).resolve().parent.parent

    # Get the path of the aircraft_icon.png file
    aircraft_icon_path = current_dir / 'aircraft_icon.png'
    
    # Return the path as a string to avoid PosixPath issues
    return str(aircraft_icon_path)
