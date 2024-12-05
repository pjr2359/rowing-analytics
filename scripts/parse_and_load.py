# parse_and_load.py

import pandas as pd
from sqlalchemy import create_engine, MetaData, select, update
from datetime import datetime
import os
import re
import logging
import sys
import traceback
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
load_dotenv()

logging.basicConfig(level=logging.DEBUG)  
logger = logging.getLogger(__name__)
password = os.getenv('PASSWORD')
# db connection setup
engine = create_engine('postgresql+psycopg2://postgres:'+password+'@localhost:5432/rowing-analytics')


def parse_csv(file_path):
    try:
        
        df = pd.read_csv(file_path, na_filter=False)
        
        # replace empty strings with None
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        logger.info(f"CSV headers: {df.columns.tolist()}")
        logger.info(f"First few rows:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return None

def get_or_create_rower_id(conn, rower_table, rower_name):
    if not rower_name or pd.isnull(rower_name):
        return None
    rower_name = rower_name.strip()
    sel = select(rower_table.c.rower_id).where(rower_table.c.name == rower_name)
    result = conn.execute(sel).fetchone()
    if result:
        return result[0]
    else:
        ins = rower_table.insert().values(name=rower_name)
        res = conn.execute(ins)
        return res.inserted_primary_key[0]

def get_or_create_boat_id(conn, boat_table, boat_name, boat_class, boat_rank):
    """Get existing boat ID or create new boat entry."""
    if not boat_name or pd.isnull(boat_name):
        return None
        
    boat_name = boat_name.strip()
    sel = select(boat_table.c.boat_id).where(
        (boat_table.c.name == boat_name) &
        (boat_table.c.boat_class == boat_class) &
        (boat_table.c.boat_rank == boat_rank)
    )
    result = conn.execute(sel).fetchone()
    
    if result:
        return result[0]
    else:
        ins = boat_table.insert().values(
            name=boat_name,
            boat_class=boat_class,
            boat_rank=boat_rank
        )
        res = conn.execute(ins)
        return res.inserted_primary_key[0]

def convert_time_to_seconds(time_str):
    try:
        time_str = str(time_str).strip()
        if not time_str or '/' in time_str:  
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
            try:
                return float(time_str)
            except ValueError:
                return None
    except (ValueError, AttributeError):
        logger.warning(f"Could not convert time string to seconds: '{time_str}'")
        return None
    
def determine_boat_configuration(row_values, boat_names,  boat_configurations):
                            """
                            Parse a row of lineup data to determine boat configurations.
                            Returns True if the row contains lineup data, False otherwise.
                            """
                            print(f"Processing row: {row_values}")
                            print(f"Boat names: {boat_names}")
                            print(f"Current boat configurations: {boat_configurations}")
                            
                            first_cell = str(row_values[0]).strip()
                            second_cell = str(row_values[1]).strip()
                            print(f"First cell: '{first_cell}'")
                            print(f"Second cell: '{second_cell}'")
                            
                            # Check if this is a lineup row (either seat number or "Coxswain")
                            if not (first_cell.isdigit() or first_cell == 'Boat'):
                                print("Not a lineup row - returning False")
                                return False
                                
                            is_cox = second_cell == 'Coxswain'
                            seat_number = 0 if is_cox else int(first_cell)
                            print(f"Is cox: {is_cox}, Seat number: {seat_number}")
                            
                            # Process each boat's data in this row
                            for i, boat_name in enumerate(boat_names):
                                print(f"\nProcessing boat: {boat_name}")
                                if i + 1 < len(row_values):
                                    rower_name = str(row_values[i + 1]).strip()
                                    print(f"Rower name: '{rower_name}'")
                                    if rower_name and rower_name.lower() != 'nan':
                                        if is_cox:
                                            boat_configurations[boat_name]['has_cox'] = True
                                            print(f"Added cox to {boat_name}")
                                        else:
                                            boat_configurations[boat_name]['seat_count'] = max(
                                                boat_configurations[boat_name]['seat_count'],
                                                seat_number
                                            )
                                            print(f"Updated seat count for {boat_name} to {boat_configurations[boat_name]['seat_count']}")
                            print(f"\nFinal boat configurations: {boat_configurations}")
                            return True

def get_boat_class(config):
    """Convert seat count and cox presence to boat class string"""
    if config['seat_count'] == 0:  # No rowers
        return None
    return f"{config['seat_count']}{'+' if config['has_cox'] else '-'}"

def insert_lineups(conn, piece_id, boat_id, rower_id, seat_number : int, is_coxswain : bool):
    """Inserting the lineups into the db"""
    

    


def get_boat_columns(data, boat_names):
    """
    Extract columns for each boat name from the DataFrame.
    
    :param data: Pandas DataFrame containing the CSV data.
    :param boat_names: List of boat names to extract columns for.
    :return: Dictionary with boat names as keys and lists of column values as values.
    """
    boat_columns = {boat_name: [] for boat_name in boat_names}
    
    for _, row in data.iterrows():
        for i, boat_name in enumerate(boat_names):
            if i + 1 < len(row):
                value = row[i + 1]
                boat_columns[boat_name].append(value)
    
    return boat_columns

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
        with engine.begin() as conn:
            logger.info("Database connection established.")
            metadata = MetaData()
            metadata.reflect(bind=engine)
            

            rower_table = metadata.tables['rower']
            boat_table = metadata.tables['boat']
            event_table = metadata.tables['event']
            piece_table = metadata.tables['piece']
            lineup_table = metadata.tables['lineup']
            result_table = metadata.tables['result']
            seat_race_table = metadata.tables.get('seat_race')

            
            try:
                first_column = data.columns[0]
                logger.info(f"Attempting to parse date from column: {first_column}")
                
                date_formats = ['%m/%d/%Y', '%m/%d/%y']
                event_date = None
                
                for date_format in date_formats:
                    try:
                        date_str = first_column.strip()
                        event_date = datetime.strptime(date_str, date_format).date()
                        logger.info(f"Successfully parsed date: {event_date}")
                        break
                    except ValueError:
                        continue
                
                if event_date:
                    
                    ins = event_table.insert().values(
                        event_date=event_date,
                        event_name=f'Practice on {event_date}'
                    )
                    result = conn.execute(ins)
                    event_id = result.inserted_primary_key[0]
                    logger.info(f"Created event with ID {event_id} for date {event_date}")
                else:
                    logger.error(f"Could not parse date from column: {first_column}")
                    return
            except Exception as e:
                logger.error(f"Error creating event: {e}")
                return

            event_id = None
            boat_names = []
            boat_ids = {}
            piece_id = None  
            piece_number = 0
            seat_race_pieces = []
            pieces = []
            lineups = {}
            has_coxswain = {}
            seat_counts = {}
            boat_configurations = {}


            for index, row in data.iterrows():
                row_values = row.fillna('').tolist()
                first_cell = str(row_values[0]).strip()

              
                if all(cell == '' for cell in row_values):
                    continue

                if index == 0 and first_cell and first_cell not in ['Boat', 'Coxswain']:
                    try:
                        logger.info(f"Attempting to parse date from: {first_cell}")
                        date_formats = ['%m/%d/%Y', '%m/%d/%y', '%m/%d/%Y ', '%m/%d/%y ']
                        event_date = None
                        for date_format in date_formats:
                            try:
                                event_date = datetime.strptime(first_cell, date_format).date()
                                logger.info(f"Successfully parsed date: {event_date}")
                                break
                            except ValueError:
                                continue
                        
                        if event_date:
                            event_name = f'Race on {event_date}'
                           
                            ins = event_table.insert().values(
                                event_date=event_date,
                                event_name=event_name
                            )
                            result = conn.execute(ins)
                            event_id = result.inserted_primary_key[0]
                            logger.info(f"Created event: {event_name} with ID {event_id}")
                        else:
                            logger.warning(f"Could not parse date from: {first_cell}")
                    except SQLAlchemyError as e:
                        logger.error(f"Database error creating event: {e}")
                    continue

                if first_cell == 'Boat':
                    boat_names = [str(name).strip() for name in row_values[1:] 
                                if name and 'over' not in name and '/' not in name]
                    boat_names = [name for name in boat_names if name]
                    logger.debug(f"Processing row: {row_values}")
                    logger.debug(f"Boat names: {boat_names}")

                    boat_configurations = {boat_name: {'seat_count': 0, 'has_cox': False} for boat_name in boat_names  }
                    logger.debug(f"Current boat configurations: {boat_configurations}")
                    logger.debug(f"First cell: {first_cell}")

                    boat_columns = get_boat_columns(data, boat_names)
                    logger.debug(f"Boat columns: {boat_columns}")

                    determine_boat_configuration(boat_columns, boat_names, boat_configurations)
                    
                    for i, boat_name in enumerate(boat_names):
                        try:
                           
                            boat_id = get_or_create_boat_id(
                                conn,
                                boat_table,
                                boat_name,
                                boat_class=boat_class,
                                boat_rank=boat_rank
                            )
                            boat_ids[(boat_name, boat_class, boat_rank)] = boat_id
                            logger.debug(f"Created/found boat {boat_name} with ID {boat_id} (rank {boat_rank})")
                        except SQLAlchemyError as e:
                            logger.error(f"Error creating boat {boat_name}: {e}")
                    
                    lineups = {boat_name: [] for boat_name in boat_names}
                    has_coxswain = {boat_name: False for boat_name in boat_names}
                    seat_counts = {boat_name: 0 for boat_name in boat_names}
                    continue

                # lineup processing
                if first_cell.isdigit() or first_cell == 'Coxswain':
                    seat_number = 0 if first_cell == 'Coxswain' else int(first_cell)
                    is_cox = first_cell == 'Coxswain'
                    
                    
                    for i, boat_name in enumerate(boat_names):
                        if i + 1 < len(row_values):
                            rower_name = str(row_values[i + 1]).strip()
                            if rower_name and rower_name.lower() != 'nan':
                                if '/' in rower_name:  # Handle paired rowers
                                    rower_names = [name.strip() for name in rower_name.split('/')]
                                else:
                                    rower_names = [rower_name]
                                
                                for name in rower_names:
                                    rower_id = get_or_create_rower_id(conn, rower_table, name)
                                    if rower_id:
                                        if is_cox:
                                            has_coxswain[boat_name] = True
                                        else:
                                            seat_counts[boat_name] = max(seat_counts[boat_name], seat_number)
                                        
                                        lineups[boat_name].append({
                                            'rower_id': rower_id,
                                            'seat_number': seat_number,
                                            'is_coxswain': is_cox
                                        })
                    continue
                
                # Add piece processing:
                if first_cell.startswith('Piece'):
                    piece_number += 1
                    piece_description = first_cell
                    
                    # Create new piece
                    ins = piece_table.insert().values(
                        event_id=event_id,
                        piece_number=piece_number,
                        description=piece_description
                    )
                    result = conn.execute(ins)
                    piece_id = result.inserted_primary_key[0]
                    logger.info(f"Created piece {piece_number}: {piece_description}")
                    
                    # Process times for each boat
                    for i, boat_name in enumerate(boat_names):
                        if i + 1 < len(row_values):
                            time_str = row_values[i + 1]
                            time_in_seconds = convert_time_to_seconds(time_str)
                            
                            if time_in_seconds is not None:
                                boat_rank = i + 1
                                num_rowers = seat_counts.get(boat_name, 8)  # Default to 8
                                has_cox = has_coxswain.get(boat_name, False)
                                boat_class = f'{num_rowers}{"+" if has_cox else "-"}'
                                
                                # Get existing boat ID
                                boat_id = None
                                for key, value in boat_ids.items():
                                    if key[0] == boat_name and key[2] == boat_rank:
                                        boat_id = value
                                        old_key = key
                                        break
                                
                                if boat_id:
                                    try:
                                        # Check if the boat class needs to be updated
                                        current_class = key[1]
                                        if current_class != boat_class:
                                            # Try to update boat class
                                            update_stmt = (
                                                update(boat_table)
                                                .where(boat_table.c.boat_id == boat_id)
                                                .values(boat_class=boat_class)
                                            )
                                            conn.execute(update_stmt)
                                            
                                            # Update boat_ids dictionary
                                            new_key = (boat_name, boat_class, boat_rank)
                                            boat_ids[new_key] = boat_ids.pop(old_key)
                                        
                                        # Insert result
                                        ins = result_table.insert().values(
                                            piece_id=piece_id,
                                            boat_id=boat_id,
                                            time=time_in_seconds
                                        )
                                        conn.execute(ins)
                                        logger.info(f"Recorded time {time_str} for boat {boat_name}")
                                    except SQLAlchemyError as e:
                                        logger.error(f"Database error updating boat {boat_name}: {e}")
                                        continue
                #this is the domain in which we will discuss the ~lineups~


                

                # Seat race handling:
                if 'Pieces Switched' in first_cell:
                    # Extract piece numbers
                    match = re.search(r'(\d+)/(\d+)', str(row_values[1]))
                    if match:
                        piece1, piece2 = int(match.group(1)), int(match.group(2))
                        seat_race_pieces.extend([piece1, piece2])
                        logger.info(f"Recorded switched pieces: {piece1} and {piece2}")
                    continue

                if 'Result' in first_cell or any('over' in str(cell) for cell in row_values):
                    for cell in row_values:
                        cell = str(cell).strip()
                        if 'over' in cell.lower():
                            # Example: "Reilly over Purcea by a total 11.3 secs"
                            match = re.search(r'(\w+)\s+over\s+(\w+)\s+by\s+(?:a\s+total\s+)?([0-9.]+)\s*secs?', cell)
                            if match:
                                winner, loser, margin = match.group(1), match.group(2), float(match.group(3))
                                
                                # Get rower IDs
                                winner_id = get_or_create_rower_id(conn, rower_table, winner)
                                loser_id = get_or_create_rower_id(conn, rower_table, loser)
                                
                                if winner_id and loser_id and seat_race_pieces:
                                    # Insert seat race result
                                    ins = seat_race_table.insert().values(
                                        event_id=event_id,
                                        piece_numbers=seat_race_pieces,
                                        rower_id_1=winner_id,
                                        rower_id_2=loser_id,
                                        time_difference=margin,
                                        winner_id=winner_id,
                                        notes=cell
                                    )
                                    conn.execute(ins)
                                    logger.info(f"Recorded seat race result: {winner} over {loser} by {margin} seconds")
                    continue

    except Exception as e:
        logger.error(f"An error occurred while parsing the file: {e}")
        traceback.print_exc()

def main():
    logger.info("Starting the parsing process.")
  
    water_data_dir = '/home/pjreilly44/rowing-analytics/data'
    
  
    if not os.path.exists(water_data_dir):
        logger.error(f"The directory {water_data_dir} does not exist.")
        return
    
    # List all files in directory
    files = os.listdir(water_data_dir)
    logger.info(f"Found files in directory: {files}")
    
    # Iterate over all CSV files in the directory
    csv_count = 0
    for filename in files:
        if filename.endswith('.csv'):
            csv_count += 1
            water_filepath = os.path.join(water_data_dir, filename)
            logger.info(f"Processing file {csv_count}: {filename}")
            parse_data(water_filepath, engine)
    
    if csv_count == 0:
        logger.warning("No CSV files found in the data directory!")

if __name__ == "__main__":
    main()
