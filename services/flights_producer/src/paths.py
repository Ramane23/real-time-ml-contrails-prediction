from pathlib import Path

#create a function to return the path of the airports.csv file
#def get_airports_path():
    #"""
    #This function returns the path of the airports.csv file
    #"""
    # Get the parent directory of the current directory two levels up
    #current_dir = Path(__file__).resolve().parent.parent

    # Get the path of the airports.csv file
    #airports_path = current_dir / 'airports.csv'
    
    #return airports_path 

#create a function to return the path of the airlines.csv file
#def get_airlines_path():
    #"""
    #This function returns the path of the airlines.csv file
    #"""
    # Get the parent directory of the current directory two levels up
    #current_dir = Path(__file__).resolve().parent.parent

    # Get the path of the airports.csv file
    #airlines_path = current_dir / 'airlines.csv'

    #return airlines_path

#create a function to return the path of the aircrafts characteristics database
#def get_aircrafts_path():
    #"""
    #This function returns the path of the aircrafts.xlsx file
    #"""
    # Get the parent directory of the current directory two levels up
    #current_dir = Path(__file__).resolve().parent.parent

    # Get the path of the airports.csv file
    #aircrafts_path = current_dir / 'FAA_aircraft_characteristics_database.xlsx'

    #return aircrafts_path

# Correctly define paths using the /app directory as the base
def get_airports_path():
    return Path('/app') / 'airports.csv'

def get_airlines_path():
    return Path('/app') / 'airlines.csv'

def get_aircrafts_path():
    return Path('/app') / 'FAA_aircraft_characteristics_database.xlsx'

if __name__ == "__main__":
    print(get_airports_path())
    print(get_airlines_path())
    # Expected output should be the correct path to the airports.csv file