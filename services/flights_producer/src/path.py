from pathlib import Path

#create a function to return the path of the airports.csv file
def get_airports_path():
    """
    This function returns the path of the airports.csv file
    """
    # Get the parent directory of the current directory two levels up
    current_dir = Path(__file__).resolve().parent.parent

    # Get the path of the airports.csv file
    airports_path = current_dir / 'airports.csv'

    return airports_path

if __name__ == "__main__":
    print(get_airports_path())
    # Expected output should be the correct path to the airports.csv file