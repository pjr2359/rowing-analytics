import pandas as pd
from sqlalchemy import create_engine

# Database connection setup (adjust username, password, host, and dbname as needed)
engine = create_engine('postgresql://postgres:Conan_Stephens27@localhost/postgres')

# Function to parse erg data from CSV
def parse_erg_data(file_path):
    """
    Parses erg data from a CSV file and returns a cleaned DataFrame.
    """
    erg_data = pd.read_csv(file_path, skiprows=1)  # Skip the first row, adjust if needed
    # Extract relevant columns
    columns = ['Side', 'Name', 'Overall Split', 'Watts/lb', 'Weight', 'Pacing 1st', 'Pacing 2nd', 'Pacing 3rd']
    erg_cleaned = erg_data.iloc[:, [0, 1, 2, 3, 4, 5, 7, 9]]  # Selecting the columns
    erg_cleaned.columns = columns  # Rename columns for clarity
    return erg_cleaned

# Function to parse water (on-water race) data from CSV
def parse_water_data(file_path):
    """
    Parses on-water race data from a CSV file and returns a cleaned DataFrame.
    """
    water_data = pd.read_csv(file_path, skiprows=1)  # Skip the first row, adjust if needed
    # Extract relevant columns with lineup and rower info
    water_cleaned = water_data.iloc[1:, 4:11]  # Selecting columns with lineup and rower info
    water_cleaned.columns = ['Name 1', 'Lineup 1', 'Name 2', 'Lineup 2', 'Name 3', 'Lineup 3', 'Name 4']  # Rename columns
    return water_cleaned

# Function to load erg data into SQL database
def load_erg_data(file_path, engine):
    """
    Loads cleaned erg data into the SQL table 'erg_data'.
    """
    # Parse the erg data
    erg_data = parse_erg_data(file_path)
    # Insert data into the erg_data table
    erg_data.to_sql('erg_data', engine, if_exists='append', index=False)
    print("Erg data successfully inserted into the database.")

# Function to load water data into SQL database
def load_water_data(file_path, engine):
    """
    Loads cleaned on-water data into the SQL table 'water_data'.
    """
    # Parse the water data
    water_data = parse_water_data(file_path)
    # Insert data into the water_data table
    water_data.to_sql('water_data', engine, if_exists='append', index=False)
    print("Water data successfully inserted into the database.")

# File paths to the CSV files
erg_file = './data/C150 2024-25 - 9_23 3x3k.csv'  # Erg data file
water_file = './data/C150 2024-25 - 9_25 4x2k.csv'  # On-water data file

# Load both CSV files into the database
load_erg_data(erg_file, engine)
load_water_data(water_file, engine)
