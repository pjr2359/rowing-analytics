import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select
from datetime import datetime
import os
import re

# Database connection setup (adjust username, password, host, and dbname as needed)
engine = create_engine('postgresql://postgres:Conan_Stephens27@localhost/postgres')

# Function to parse the CSV files
def parse_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    return data

# Function to parse events, boats, lineups, pieces, and results
def parse_data(file_path, engine):
    data = parse_csv(file_path)
    conn = engine.connect()
    metadata = MetaData(bind=engine)
    metadata.reflect()

    # Tables
    rower_table = metadata.tables['rower']
    boat_table = metadata.tables['boat']
    event_table = metadata.tables['event']
    piece_table = metadata.tables['piece']
    lineup_table = metadata.tables['lineup']
    result_table = metadata.tables['result']

    # Initialize variables
    boats = []
    current_boat = None
    event_date = None
    event_name = None
    pieces = []
    piece_number = 0
    parsing_lineups = False
    parsing_pieces = False
    boat_names = []
    coxswains = {}

    for index, row in data.iterrows():
        # Check for event date
        if pd.notnull(row[0]) and re.match(r'\d{1,2}/\d{1,2}/\d{4}', str(row[0])):
            event_date = datetime.strptime(row[0], '%m/%d/%Y').date()
            event_name = 'Race on ' + str(event_date)
            # Insert event into database
            ins = event_table.insert().values(event_date=event_date, event_name=event_name)
            result = conn.execute(ins)
            event_id = result.inserted_primary_key[0]
            continue

        # Check for boat names
        if pd.notnull(row[1]) and pd.isnull(row[0]):
            boat_names = [name for name in row if pd.notnull(name)]
            # Insert boats into database
            boat_ids = {}
            for boat_name in boat_names:
                sel = select([boat_table.c.boat_id]).where(boat_table.c.name == boat_name)
                result = conn.execute(sel).fetchone()
                if result:
                    boat_id = result[0]
                else:
                    ins = boat_table.insert().values(name=boat_name, boat_class='Unknown')
                    res = conn.execute(ins)
                    boat_id = res.inserted_primary_key[0]
                boat_ids[boat_name] = boat_id
            continue

        # Check for coxswains
        if pd.notnull(row[0]) and 'Coxswain' in str(row[0]):
            coxswains = {boat_names[i]: row[i+1] for i in range(len(boat_names))}
            continue

        # Check for rower positions
        if pd.notnull(row[0]) and row[0] in [str(i) for i in range(1, 9)]:
            seat_number = int(row[0])
            for i, boat_name in enumerate(boat_names):
                rower_name = row[i+1] if i+1 < len(row) and pd.notnull(row[i+1]) else None
                if rower_name:
                    # Insert rower into database if not exists
                    sel = select([rower_table.c.rower_id]).where(rower_table.c.name == rower_name)
                    result = conn.execute(sel).fetchone()
                    if result:
                        rower_id = result[0]
                    else:
                        ins = rower_table.insert().values(name=rower_name)
                        res = conn.execute(ins)
                        rower_id = res.inserted_primary_key[0]

                    # Insert lineup
                    ins = lineup_table.insert().values(
                        piece_id=None,  # Will be updated later
                        boat_id=boat_ids[boat_name],
                        rower_id=rower_id,
                        seat_number=seat_number,
                        is_coxswain=False
                    )
                    conn.execute(ins)

        # Handle coxswain entries
        if coxswains:
            for boat_name, coxswain_name in coxswains.items():
                # Insert coxswain into database if not exists
                sel = select([rower_table.c.rower_id]).where(rower_table.c.name == coxswain_name)
                result = conn.execute(sel).fetchone()
                if result:
                    cox_id = result[0]
                else:
                    ins = rower_table.insert().values(name=coxswain_name)
                    res = conn.execute(ins)
                    cox_id = res.inserted_primary_key[0]

                # Insert coxswain into lineup
                ins = lineup_table.insert().values(
                    piece_id=None,  # Will be updated later
                    boat_id=boat_ids[boat_name],
                    rower_id=cox_id,
                    seat_number=0,
                    is_coxswain=True
                )
                conn.execute(ins)
            coxswains = {}  # Reset coxswains after processing

        # Check for pieces
        if pd.notnull(row[0]) and 'Piece' in str(row[0]):
            piece_number += 1
            piece_description = row[0]
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
            conn.execute(
                lineup_table.update().where(lineup_table.c.piece_id == None).values(piece_id=piece_id)
            )

            # Parse times for each boat
            times = {}
            for i, boat_name in enumerate(boat_names):
                time_str = row[i+1] if i+1 < len(row) and pd.notnull(row[i+1]) else None
                if time_str:
                    time_in_seconds = convert_time_to_seconds(time_str)
                    times[boat_name] = time_in_seconds
                    # Insert result into database
                    ins = result_table.insert().values(
                        piece_id=piece_id,
                        boat_id=boat_ids[boat_name],
                        time=time_in_seconds,
                        split=None,
                        margin=None  # Can be calculated later
                    )
                    conn.execute(ins)

    conn.close()

def convert_time_to_seconds(time_str):
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
    return None

# Function to load erg data into SQL database
def load_erg_data(file_path, engine):
    """
    Loads cleaned erg data into the SQL table 'erg_data'.
    """
    erg_data = parse_erg_csv(file_path)
    conn = engine.connect()
    metadata = MetaData(bind=engine)
    metadata.reflect()
    rower_table = metadata.tables['rower']
    erg_table = metadata.tables['erg_data']

    for index, row in erg_data.iterrows():
        rower_name = row['Name']
        # Insert rower if not exists
        sel = select([rower_table.c.rower_id]).where(rower_table.c.name == rower_name)
        result = conn.execute(sel).fetchone()
        if result:
            rower_id = result[0]
        else:
            ins = rower_table.insert().values(name=rower_name, weight=row['Weight'])
            res = conn.execute(ins)
            rower_id = res.inserted_primary_key[0]
        # Insert erg data
        ins = erg_table.insert().values(
            rower_id=rower_id,
            test_date=row['Test Date'],
            overall_split=row['Overall Split'],
            watts_per_lb=row['Watts/lb'],
            weight=row['Weight'],
            pacing=row['Pacing']
        )
        conn.execute(ins)
    conn.close()

def parse_erg_csv(file_path):
    erg_data = pd.read_csv(file_path)
    # Process erg data as per your CSV structure
    # Return a DataFrame with columns matching the erg_data table
    return erg_data

if __name__ == "__main__":
    # Directory containing water data CSV files
    water_data_dir = '/home/pjreilly44/rowing-analytics/data/water data'

    # Load water data
    for file_name in os.listdir(water_data_dir):
        if file_name.endswith('.csv'):
            water_file_path = os.path.join(water_data_dir, file_name)
            parse_data(water_file_path, engine)

    # Load erg data (if applicable)
    erg_data_dir = '/home/pjreilly44/rowing-analytics/data/erg data'
    for file_name in os.listdir(erg_data_dir):
        if file_name.endswith('.csv'):
            erg_file_path = os.path.join(erg_data_dir, file_name)
            load_erg_data(erg_file_path, engine)
