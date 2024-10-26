import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, Text, ARRAY, ForeignKey, select, update
from datetime import datetime
import os
import re
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Database connection setup
engine = create_engine('postgresql+psycopg2://postgres:Conan_Stephens27@localhost:5432/rowing-analytics')

def parse_csv(file_path):
    # Read the CSV without headers to keep raw data
    data = pd.read_csv(file_path, header=None)
    return data

def get_or_create_rower_id(conn, rower_table, rower_name):
    if not rower_name or pd.isnull(rower_name):
        return None
    sel = select(rower_table.c.rower_id).where(rower_table.c.name == rower_name)
    result = conn.execute(sel).fetchone()
    if result:
        return result[0]
    else:
        ins = rower_table.insert().values(name=rower_name)
        res = conn.execute(ins)
        return res.inserted_primary_key[0]

def parse_data(file_path, engine):
    logger.info(f"Parsing file: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    try:
        data = parse_csv(file_path)
        logger.info(f"Data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        traceback.print_exc()
        return

    try:
        with engine.connect() as conn:
            logger.info("Database connection established.")
            metadata = MetaData()
            # Reflect existing tables
            metadata.reflect(bind=engine)
            rower_table = metadata.tables['rower']
            boat_table = metadata.tables['boat']
            event_table = metadata.tables['event']
            piece_table = metadata.tables['piece']
            lineup_table = metadata.tables['lineup']
            result_table = metadata.tables['result']

            # Ensure 'seat_race' table exists
            if 'seat_race' not in metadata.tables:
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
            else:
                seat_race_table = metadata.tables['seat_race']

            # Initialize variables
            event_id = None
            boat_names = []
            boat_ids = {}
            coxswains = {}
            piece_number = 0
            seat_race_pieces = []
            lineups = {}  # To store lineups per piece
            pieces_switched = False

            # Iterate over each row in the data
            for index, row in data.iterrows():
                row_values = row.fillna('').tolist()
                first_cell = str(row_values[0]).strip()

                # Skip empty rows
                if all(cell == '' for cell in row_values):
                    continue

                # Check for event date or name
                if index == 0 and first_cell:
                    # Try parsing date
                    try:
                        event_date = datetime.strptime(first_cell, '%m/%d/%Y').date()
                        event_name = f'Race on {event_date}'
                    except ValueError:
                        event_name = first_cell
                        event_date = None
                    logger.info(f"Event found: {event_name}, Date: {event_date}")
                    # Insert event into database
                    ins = event_table.insert().values(event_date=event_date, event_name=event_name)
                    result = conn.execute(ins)
                    event_id = result.inserted_primary_key[0]
                    continue

                # Check for 'Pieces Switched' and 'Result' in the row
                if 'Pieces Switched' in row_values or 'Result' in row_values:
                    # Extract seat-racing information
                    for cell in row_values:
                        cell = str(cell).strip()
                        if 'Pieces Switched' in cell:
                            pieces_switched = True
                            continue
                        if 'over' in cell:
                            # Example: 'Reilly over Purcea by a total 11.3 secs'
                            match = re.match(r'(\w+)\s+over\s+(\w+)\s+by\s+a\s+total\s+([0-9.]+)\s+secs', cell)
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
                                    notes=cell
                                )
                                conn.execute(ins)
                                logger.info(f"Inserted seat race result between {name1} and {name2}")
                            else:
                                logger.warning(f"Could not parse seat race result: {cell}")
                    continue

                # Check for 'Boat' header
                if first_cell == 'Boat':
                    # Extract boat names starting from the second cell
                    boat_names = [str(name).strip() for name in row_values[1:] if name]
                    # Insert boats into database
                    for boat_name in boat_names:
                        if boat_name not in boat_ids:
                            sel = select(boat_table.c.boat_id).where(boat_table.c.name == boat_name)
                            result = conn.execute(sel).fetchone()
                            if result:
                                boat_id = result[0]
                            else:
                                ins = boat_table.insert().values(name=boat_name, boat_class='Unknown')
                                res = conn.execute(ins)
                                boat_id = res.inserted_primary_key[0]
                            boat_ids[boat_name] = boat_id
                    continue

                # Check for 'Coxswain' row
                if first_cell == 'Coxswain':
                    coxswains = {}
                    for i, boat_name in enumerate(boat_names):
                        coxswain_name = row_values[i+1] if i+1 < len(row_values) else None
                        if coxswain_name:
                            coxswains[boat_name] = coxswain_name.strip()
                    # Insert coxswains into 'lineup' table
                    for boat_name, coxswain_name in coxswains.items():
                        cox_id = get_or_create_rower_id(conn, rower_table, coxswain_name)
                        lineup_entry = {
                            'piece_id': None,  # Will be updated later
                            'boat_id': boat_ids[boat_name],
                            'rower_id': cox_id,
                            'seat_number': 0,  # Seat 0 for coxswain
                            'is_coxswain': True
                        }
                        # Store in lineups dict
                        lineups.setdefault(boat_name, []).append(lineup_entry)
                    continue

                # Check for seat numbers (assuming they are digits or '8', '7', etc.)
                if first_cell.isdigit():
                    seat_number = int(first_cell)
                    for i, boat_name in enumerate(boat_names):
                        rower_name = row_values[i+1] if i+1 < len(row_values) else None
                        if rower_name:
                            rower_name = rower_name.strip()
                            rower_id = get_or_create_rower_id(conn, rower_table, rower_name)
                            lineup_entry = {
                                'piece_id': None,  # Will be updated later
                                'boat_id': boat_ids[boat_name],
                                'rower_id': rower_id,
                                'seat_number': seat_number,
                                'is_coxswain': False
                            }
                            # Store in lineups dict
                            lineups.setdefault(boat_name, []).append(lineup_entry)
                    continue

                # Check for 'Piece' entries
                if first_cell.startswith('Piece'):
                    piece_number += 1
                    piece_description = first_cell
                    # Insert piece into database
                    ins = piece_table.insert().values(
                        event_id=event_id,
                        piece_number=piece_number,
                        distance=None,
                        description=piece_description
                    )
                    result = conn.execute(ins)
                    piece_id = result.inserted_primary_key[0]

                    # Update lineups with piece_id
                    for boat_name in boat_names:
                        if boat_name in lineups:
                            for lineup_entry in lineups[boat_name]:
                                lineup_entry['piece_id'] = piece_id
                                # Insert lineup into database
                                ins = lineup_table.insert().values(**lineup_entry)
                                conn.execute(ins)
                            # Clear lineups for the next piece
                            lineups[boat_name] = []

                    # Now parse the times for each boat
                    for i, boat_name in enumerate(boat_names):
                        time_str = row_values[i+1] if i+1 < len(row_values) else ''
                        time_in_seconds = convert_time_to_seconds(time_str)
                        if time_in_seconds is not None:
                            # Insert result into database
                            ins = result_table.insert().values(
                                piece_id=piece_id,
                                boat_id=boat_ids[boat_name],
                                time=time_in_seconds,
                                split=None,
                                margin=None
                            )
                            conn.execute(ins)
                    continue

                # Check for 'Split' entries
                if first_cell == 'Split':
                    # Update splits in the result table
                    for i, boat_name in enumerate(boat_names):
                        split_str = row_values[i+1] if i+1 < len(row_values) else ''
                        split_in_seconds = convert_time_to_seconds(split_str)
                        if split_in_seconds is not None:
                            # Update the split for the last piece and boat
                            update_stmt = (
                                update(result_table)
                                .where(result_table.c.piece_id == piece_id)
                                .where(result_table.c.boat_id == boat_ids[boat_name])
                                .values(split=split_in_seconds)
                            )
                            conn.execute(update_stmt)
                    continue

                # Check for 'Margin' entries
                if first_cell == 'Margin':
                    # Update margins in the result table
                    for i, boat_name in enumerate(boat_names):
                        margin_str = row_values[i+1] if i+1 < len(row_values) else ''
                        margin_in_seconds = convert_time_to_seconds(margin_str)
                        if margin_in_seconds is not None:
                            # Update the margin for the last piece and boat
                            update_stmt = (
                                update(result_table)
                                .where(result_table.c.piece_id == piece_id)
                                .where(result_table.c.boat_id == boat_ids[boat_name])
                                .values(margin=margin_in_seconds)
                            )
                            conn.execute(update_stmt)
                    continue

                # Handle any other cases as needed

            logger.info("Finished processing data.")

    except Exception as e:
        logger.error(f"An error occurred while parsing the CSV file: {e}")
        traceback.print_exc()

def convert_time_to_seconds(time_str):
    try:
        time_str = time_str.strip()
        if not time_str:
            return None
        if ':' in time_str:
            parts = time_str.split(':')
            parts = [float(p) for p in parts]
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 3600 + minutes * 60 + seconds
        else:
            return float(time_str)
    except ValueError:
        logger.warning(f"Could not convert time string to seconds: '{time_str}'")
        return None

def main():
    logger.info("Starting the parsing process.")
    # Directory containing water data CSV files
    water_data_dir = 'C:/Users/PJRei/rowing-analytics/data'

    # Iterate over all CSV files in the directory
    for filename in os.listdir(water_data_dir):
        if filename.endswith('.csv'):
            water_filepath = os.path.join(water_data_dir, filename)
            parse_data(water_filepath, engine)

if __name__ == "__main__":
    main()
