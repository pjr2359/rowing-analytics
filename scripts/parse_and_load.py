import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, Text, ARRAY, ForeignKey, select
from datetime import datetime
import os
import re
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("Script has started.")

# Database connection setup
engine = create_engine('postgresql://postgres:Conan_Stephens27@localhost/rowing-analytics')

def parse_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    return data

def get_or_create_rower_id(conn, rower_table, rower_name):
    sel = select(rower_table.c.rower_id).where(rower_table.c.name == rower_name)
    result = conn.execute(sel).fetchone()
    if result:
        return result[0]
    else:
        ins = rower_table.insert().values(name=rower_name)
        res = conn.execute(ins)
        return res.inserted_primary_key[0]

def parse_data(file_path, engine):
    logger.debug("Entered parse_data function.")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")
        sys.exit(1)
    else:
        logger.debug(f"File found: {file_path}")

    try:
        data = parse_csv(file_path)
        logger.info(f"Data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        traceback.print_exc()
        return

    try:
        with engine.connect() as conn:
            logger.debug("Database connection established.")
            metadata = MetaData()
            # Reflect existing tables
            rower_table = Table('rower', metadata, autoload_with=engine)
            boat_table = Table('boat', metadata, autoload_with=engine)
            event_table = Table('event', metadata, autoload_with=engine)
            piece_table = Table('piece', metadata, autoload_with=engine)
            lineup_table = Table('lineup', metadata, autoload_with=engine)
            result_table = Table('result', metadata, autoload_with=engine)
            # Define the seat_race table (if not already defined)
            seat_race_table = Table('seat_race', metadata,
                Column('seat_race_id', Integer, primary_key=True),
                Column('event_id', Integer, ForeignKey('event.event_id')),
                Column('piece_numbers', ARRAY(Integer)),
                Column('rower_id_1', Integer, ForeignKey('rower.rower_id')),
                Column('rower_id_2', Integer, ForeignKey('rower.rower_id')),
                Column('time_difference', Float),
                Column('winner_id', Integer, ForeignKey('rower.rower_id')),
                Column('notes', Text)
            )
            metadata.create_all(engine)

            # Initialize variables
            event_id = None
            boat_names = []
            boat_ids = {}
            coxswains = {}
            piece_number = 0
            seat_race_pieces = []

            for index, row in data.iterrows():
                logger.debug(f"Processing row {index}: {row.values}")

                # Skip empty rows
                if row.isnull().all():
                    continue

                # Check for event date
                if pd.notnull(row[0]) and re.match(r'\d{1,2}/\d{1,2}/\d{4}', str(row[0])):
                    event_date = datetime.strptime(row[0], '%m/%d/%Y').date()
                    event_name = 'Race on ' + str(event_date)
                    logger.info(f"Event date found: {event_date}")
                    # Insert event into database
                    ins = event_table.insert().values(event_date=event_date, event_name=event_name)
                    result = conn.execute(ins)
                    event_id = result.inserted_primary_key[0]
                    logger.info(f"Inserted event with ID: {event_id}")
                    continue

                # Detect "Pieces Switched"
                if pd.notnull(row[0]) and 'Pieces Switched' in str(row[0]):
                    pieces_switched = str(row[0]).replace('Pieces Switched', '').strip()
                    piece_numbers = [int(p.strip()) for p in pieces_switched.split('/')]
                    logger.info(f"Pieces switched: {piece_numbers}")
                    seat_race_pieces = piece_numbers  # Store for later use
                    continue

                # Detect "Result"
                if pd.notnull(row[0]) and 'over' in str(row[0]):
                    result_text = str(row[0]).strip()
                    match = re.match(r'(\w+)\s+over\s+(\w+)\s+by\s+a\s+total\s+([0-9.]+)\s+secs', result_text)
                    if match:
                        name1 = match.group(1)
                        name2 = match.group(2)
                        time_diff = float(match.group(3))
                        logger.info(f"Seat race result: {name1} over {name2} by {time_diff} seconds")

                        # Get or create rower IDs
                        rower_id_1 = get_or_create_rower_id(conn, rower_table, name1)
                        rower_id_2 = get_or_create_rower_id(conn, rower_table, name2)

                        # Determine winner
                        winner_id = rower_id_1

                        # Insert seat race data
                        ins = seat_race_table.insert().values(
                            event_id=event_id,
                            piece_numbers=seat_race_pieces,
                            rower_id_1=rower_id_1,
                            rower_id_2=rower_id_2,
                            time_difference=time_diff,
                            winner_id=winner_id,
                            notes=result_text
                        )
                        conn.execute(ins)
                        logger.info(f"Inserted seat race result between {name1} and {name2}")
                    else:
                        logger.warning(f"Could not parse seat race result: {result_text}")
                    continue

                # Check for boat names
                if pd.isnull(row[0]) and pd.notnull(row[1]):
                    potential_boat_names = [str(name).strip() for name in row[1:] if pd.notnull(name)]
                    # Check if the row likely contains boat names
                    if all(re.match(r'^[A-Za-z0-9\s\']+$', name) for name in potential_boat_names):
                        invalid_entries = {'Pieces Switched', 'Result', 'Meters', 'Split', 'Margin'}
                        if not any(name in invalid_entries for name in potential_boat_names):
                            boat_names = potential_boat_names
                            logger.info(f"Boat names found: {boat_names}")
                            # Insert boats into database
                            for boat_name in boat_names:
                                sel = select(boat_table.c.boat_id).where(boat_table.c.name == boat_name)
                                result = conn.execute(sel).fetchone()
                                if result:
                                    boat_id = result[0]
                                    logger.info(f"Boat '{boat_name}' exists with ID: {boat_id}")
                                else:
                                    ins = boat_table.insert().values(name=boat_name, boat_class='Unknown')
                                    res = conn.execute(ins)
                                    boat_id = res.inserted_primary_key[0]
                                    logger.info(f"Inserted boat '{boat_name}' with ID: {boat_id}")
                                boat_ids[boat_name] = boat_id
                            continue

                # Check for coxswains
                if pd.notnull(row[0]) and 'Coxswain' in str(row[0]):
                    coxswains = {boat_names[i]: row[i+1] for i in range(len(boat_names))}
                    logger.info(f"Coxswains found: {coxswains}")
                    continue

                # Check for rower positions
                if pd.notnull(row[0]) and str(row[0]).strip().isdigit():
                    seat_number = int(row[0])
                    for i, boat_name in enumerate(boat_names):
                        if i+1 < len(row) and pd.notnull(row[i+1]):
                            rower_name = str(row[i+1]).strip()
                            rower_id = get_or_create_rower_id(conn, rower_table, rower_name)
                            # Insert lineup
                            ins = lineup_table.insert().values(
                                piece_id=None,  # Will be updated later
                                boat_id=boat_ids[boat_name],
                                rower_id=rower_id,
                                seat_number=seat_number,
                                is_coxswain=False
                            )
                            conn.execute(ins)
                            logger.info(f"Inserted lineup for rower '{rower_name}' in boat '{boat_name}'")
                    continue

                # Handle coxswain entries after rower positions
                if coxswains:
                    for boat_name, coxswain_name in coxswains.items():
                        coxswain_name = str(coxswain_name).strip()
                        cox_id = get_or_create_rower_id(conn, rower_table, coxswain_name)
                        # Insert coxswain into lineup
                        ins = lineup_table.insert().values(
                            piece_id=None,  # Will be updated later
                            boat_id=boat_ids[boat_name],
                            rower_id=cox_id,
                            seat_number=0,
                            is_coxswain=True
                        )
                        conn.execute(ins)
                        logger.info(f"Inserted lineup for coxswain '{coxswain_name}' in boat '{boat_name}'")
                    coxswains = {}  # Reset coxswains after processing
                    continue

                # Check for pieces
                if pd.notnull(row[0]) and 'Piece' in str(row[0]):
                    piece_number += 1
                    piece_description = str(row[0]).strip()
                    logger.info(f"Found piece: {piece_description}")
                    # Insert piece into database
                    ins = piece_table.insert().values(
                        event_id=event_id,
                        piece_number=piece_number,
                        distance=None,
                        description=piece_description
                    )
                    result = conn.execute(ins)
                    piece_id = result.inserted_primary_key[0]
                    logger.info(f"Inserted piece with ID: {piece_id}")
                    # Update lineups with piece_id
                    conn.execute(
                        lineup_table.update().where(lineup_table.c.piece_id == None).values(piece_id=piece_id)
                    )
                    logger.info("Updated lineup with piece_id")
                    # Parse times for each boat
                    for i, boat_name in enumerate(boat_names):
                        if i+1 < len(row) and pd.notnull(row[i+1]):
                            time_str = str(row[i+1]).strip()
                            if time_str:
                                time_in_seconds = convert_time_to_seconds(time_str)
                                if time_in_seconds is not None:
                                    # Insert result into database
                                    ins = result_table.insert().values(
                                        piece_id=piece_id,
                                        boat_id=boat_ids[boat_name],
                                        time=time_in_seconds,
                                        split=None,
                                        margin=None  # Can be calculated later
                                    )
                                    conn.execute(ins)
                                    logger.info(f"Inserted result for boat '{boat_name}' in piece '{piece_description}'")
                                else:
                                    logger.warning(f"Invalid time format for boat '{boat_name}': '{time_str}'")
                    continue

                logger.debug(f"No matching condition for row {index}. Content: {row.values}")

            logger.debug("Finished processing data.")

    except Exception as e:
        logger.error(f"An error occurred while parsing the CSV file: {e}")
        traceback.print_exc()

def convert_time_to_seconds(time_str):
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        else:
            return float(time_str)
    except ValueError:
        logger.warning(f"Could not convert time string to seconds: '{time_str}'")
        return None

def main():
    logger.debug("Main function is running.")
    # Directory containing water data CSV files
    water_data_dir = '/home/pjreilly44/rowing-analytics/data/water data'
    water_filepath = '/home/pjreilly44/rowing-analytics/data/water data/Water Data 24_25 - Copy of 9_14 3x4k.csv'

    # Check if the file exists
    if not os.path.exists(water_filepath):
        logger.error(f"File not found: {water_filepath}")
        print(f"File not found: {water_filepath}")
        sys.exit(1)
    else:
        logger.debug(f"File found: {water_filepath}")

    # Parse a single water data file for testing
    parse_data(water_filepath, engine)

if __name__ == "__main__":
    main()
