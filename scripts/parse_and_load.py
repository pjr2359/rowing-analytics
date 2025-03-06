# parse_and_load.py
import pandas as pd
import re
import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, MetaData, select, update, insert
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)



class RowingDataParser:
    def __init__(self, db_engine):
        self.engine = db_engine
        self.metadata = MetaData()
        self.metadata.reflect(bind=engine)
        self.tables = {
            'rower': self.metadata.tables['rower'],
            'boat': self.metadata.tables['boat'],
            'event': self.metadata.tables['event'],
            'piece': self.metadata.tables['piece'],
            'lineup': self.metadata.tables['lineup'],
            'result': self.metadata.tables['result'],
            'seat_race': self.metadata.tables['seat_race']
        }
        
    def parse_csv_file(self, file_path):
        """Parse a rowing data CSV file with proper structure detection"""
        logger.info(f"Parsing file: {file_path}")
        
        try:
            # Read the raw CSV without headers first
            raw_data = pd.read_csv(file_path, header=None)
            logger.debug(f"Raw data shape: {raw_data.shape}")
            
            # Extract date from first cell
            event_date = self._extract_date(raw_data.iloc[0, 0])
            logger.info(f"Extracted event date: {event_date}")
            
            # Find the row that contains "Boat" as the first cell
            boat_row_idx = raw_data[raw_data[0] == "Boat"].index
            if len(boat_row_idx) == 0:
                logger.error("Could not find 'Boat' row in CSV")
                return None
                
            boat_row_idx = boat_row_idx[0]
            boat_names = [str(name).strip() for name in raw_data.iloc[boat_row_idx, 1:] 
                         if pd.notna(name) and str(name).strip()]
            logger.info(f"Found boats: {boat_names}")
            
            # Extract boat information
            boat_info = self._extract_boat_info(raw_data, boat_row_idx, boat_names)
            
            # Extract lineup information
            lineup_info = self._extract_lineup_info(raw_data, boat_row_idx, boat_names)
            
            # Extract piece information
            piece_info = self._extract_piece_info(raw_data, boat_row_idx, boat_names)
            
            # Extract seat race results if any
            seat_race_info = self._extract_seat_race_info(raw_data)
            
            return {
                'event_date': event_date,
                'boats': boat_info,
                'lineups': lineup_info,
                'pieces': piece_info,
                'seat_races': seat_race_info
            }
            
        except Exception as e:
            logger.error(f"Error parsing CSV file: {str(e)}", exc_info=True)
            return None
            
    def _extract_date(self, date_cell):
        """Extract and parse date from the first cell"""
        if pd.isna(date_cell):
            return None
            
        date_str = str(date_cell).strip()
        date_patterns = ['%m/%d/%Y', '%m/%d/%y']
        
        for pattern in date_patterns:
            try:
                return datetime.strptime(date_str, pattern).date()
            except ValueError:
                continue
                
        logger.warning(f"Could not parse date from: {date_str}")
        return None
        
    def _extract_boat_info(self, data, boat_row_idx, boat_names):
        """Extract information about each boat"""
        boat_info = {}
        
        # Check where "Coxswain" row is
        cox_row_idx = data[data[0] == "Coxswain"].index
        has_cox = len(cox_row_idx) > 0
        
        # Determine the highest seat number for each boat
        seat_rows = []
        for idx, row in data.iterrows():
            if idx > boat_row_idx and str(row[0]).strip().isdigit():
                seat_rows.append(idx)
        
        max_seat = 0
        if seat_rows:
            max_seat = max([int(str(data.iloc[idx, 0]).strip()) for idx in seat_rows])
            
        for i, boat_name in enumerate(boat_names):
            boat_info[boat_name] = {
                'boat_class': f"{max_seat}{'+' if has_cox else '-'}",
                'boat_rank': i + 1  # Assign rank based on order in CSV
            }
            
        return boat_info
        
    def _extract_lineup_info(self, data, boat_row_idx, boat_names):
        """Extract lineup information (who sits where in each boat)"""
        lineups = {boat_name: [] for boat_name in boat_names}
        
        # Look for coxswain row
        cox_row_idx = data[data[0] == "Coxswain"].index
        if cox_row_idx.size > 0:
            cox_row_idx = cox_row_idx[0]
            for i, boat_name in enumerate(boat_names):
                col_idx = i + 1
                if col_idx < data.shape[1]:
                    cox_name = str(data.iloc[cox_row_idx, col_idx]).strip()
                    if cox_name and cox_name.lower() not in ('nan', ''):
                        lineups[boat_name].append({
                            'rower_name': cox_name,
                            'seat_number': 0,
                            'is_coxswain': True
                        })
        
        # Look for numbered seats (1-8)
        for idx, row in data.iterrows():
            first_cell = str(row[0]).strip()
            if first_cell.isdigit() and int(first_cell) > 0:
                seat_number = int(first_cell)
                for i, boat_name in enumerate(boat_names):
                    col_idx = i + 1
                    if col_idx < data.shape[1]:
                        rower_cell = str(data.iloc[idx, col_idx]).strip()
                        if rower_cell and rower_cell.lower() not in ('nan', ''):
                            # Handle paired rowers (with slash)
                            if '/' in rower_cell:
                                rower_names = [name.strip() for name in rower_cell.split('/')]
                                for rower_name in rower_names:
                                    lineups[boat_name].append({
                                        'rower_name': rower_name,
                                        'seat_number': seat_number,
                                        'is_coxswain': False
                                    })
                            else:
                                lineups[boat_name].append({
                                    'rower_name': rower_cell,
                                    'seat_number': seat_number,
                                    'is_coxswain': False
                                })
                                
        return lineups
        
    def _extract_piece_info(self, data, boat_row_idx, boat_names):
        """Extract piece (race segment) information and times"""
        pieces = []
        
        for idx, row in data.iterrows():
            first_cell = str(row[0]).strip()
            
            # Check for any "Piece" text
            match = re.search(r'[Pp]iece\s*(\d+)', first_cell)
            if match:
                piece_number = int(match.group(1))
                piece_times = {}
                
                # Process boat times
                for i, boat_name in enumerate(boat_names):
                    col_idx = i + 1
                    if col_idx < data.shape[1]:
                        time_str = str(data.iloc[idx, col_idx]).strip()
                        if time_str and time_str.lower() not in ('nan', ''):
                            piece_times[boat_name] = self._convert_time_to_seconds(time_str)
                
                # Process split times and margins (existing code)
                # ...
                
                pieces.append({
                    'piece_number': piece_number,
                    'results': piece_times
                })
        
        return pieces
        
    def _extract_seat_race_info(self, data):
        """Extract seat race results (if any)"""
        seat_races = []
        
        # Look for 'Pieces Switched' row
        for idx, row in data.iterrows():
            first_cell = str(row[0]).strip()
            if 'Pieces Switched' in first_cell:
                # Look for results in the next row
                if idx + 1 < data.shape[0]:
                    result_row = data.iloc[idx + 1]
                    if str(result_row[0]).strip() == 'Result':
                        for i in range(1, data.shape[1]):
                            result_text = str(result_row[i]).strip()
                            if result_text and 'over' in result_text.lower():
                                # Get pieces_switched for this column
                                pieces_switched = None
                                switch_info = str(data.iloc[idx, i]).strip()
                                match = re.search(r'(\d+)/(\d+)', switch_info)
                                if match:
                                    pieces_switched = [int(match.group(1)), int(match.group(2))]
                                
                                # Improved regex to match more formats
                                match = re.search(r'(\w+)\s+over\s+(\w+)\s+by\s+(?:a\s+total\s+)?([0-9.]+)\s*(?:seconds|secs)', result_text, re.IGNORECASE)
                                if match:
                                    winner, loser, margin = match.group(1), match.group(2), float(match.group(3))
                                    logger.debug(f"Found seat race: {winner} over {loser} by {margin}")
                                    seat_races.append({
                                        'pieces_switched': pieces_switched,
                                        'winner': winner,
                                        'loser': loser,
                                        'margin': margin,
                                        'notes': result_text
                                    })
                                else:
                                    logger.warning(f"Could not parse seat race result: '{result_text}'")
        
        logger.info(f"Extracted {len(seat_races)} seat races")
        return seat_races
            
    def _convert_time_to_seconds(self, time_str):
        """Convert time string (MM:SS.ss) to seconds"""
        try:
            time_str = str(time_str).strip()
            if not time_str or '/' in time_str:
                return None
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = float(parts[0]), float(parts[1])
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
            else:
                try:
                    return float(time_str)
                except ValueError:
                    return None
        except (ValueError, AttributeError):
            logger.warning(f"Could not convert time string to seconds: '{time_str}'")
            return None
            
    def load_data_to_db(self, parsed_data):
        """Load parsed data into the database"""
        if not parsed_data:
            logger.error("No data to load")
            return
            
        try:
            with self.engine.begin() as conn:
                # Create event
                event_id = self._create_event(conn, parsed_data['event_date'])
                if not event_id:
                    logger.error("Failed to create event")
                    return
                    
                # Create boats
                boat_ids = {}
                for boat_name, boat_data in parsed_data['boats'].items():
                    boat_id = self._get_or_create_boat(
                        conn, 
                        boat_name, 
                        boat_data['boat_class'],
                        boat_data['boat_rank']
                    )
                    boat_ids[boat_name] = boat_id
                    
                # Create pieces and results
                piece_ids = {}
                for piece_data in parsed_data['pieces']:
                    piece_id = self._create_piece(
                        conn,
                        event_id,
                        piece_data['piece_number']
                    )
                    piece_ids[piece_data['piece_number']] = piece_id
                    
                    # Add results for each boat
                    for boat_name, result_data in piece_data['results'].items():
                        if boat_name in boat_ids:
                            boat_id = boat_ids[boat_name]
                            self._create_result(
                                conn,
                                piece_id,
                                boat_id,
                                result_data
                            )
                
                # Create rowers and lineups
                for boat_name, lineup_data in parsed_data['lineups'].items():
                    if boat_name in boat_ids:
                        boat_id = boat_ids[boat_name]
                        for rower_data in lineup_data:
                            rower_id = self._get_or_create_rower(
                                conn,
                                rower_data['rower_name']
                            )
                            
                            # Add to lineup for each piece
                            for piece_number, piece_id in piece_ids.items():
                                self._create_lineup(
                                    conn,
                                    piece_id,
                                    boat_id,
                                    rower_id,
                                    rower_data['seat_number'],
                                    rower_data['is_coxswain']
                                )
                
                # Create seat races
                for seat_race in parsed_data['seat_races']:
                    self._create_seat_race(conn, event_id, seat_race)
                    
                logger.info(f"Successfully loaded data for event {event_id}")
                
        except Exception as e:
            logger.error(f"Error loading data to database: {str(e)}", exc_info=True)
    
    def _create_event(self, conn, event_date):
        """Create an event in the database"""
        if not event_date:
            return None
            
        event_name = f'Race on {event_date}'
        ins = self.tables['event'].insert().values(
            event_date=event_date,
            event_name=event_name
        )
        result = conn.execute(ins)
        event_id = result.inserted_primary_key[0]
        logger.info(f"Created event: {event_name} with ID {event_id}")
        return event_id
        
    def _get_or_create_boat(self, conn, boat_name, boat_class, boat_rank):
        """Get or create a boat in the database"""
        sel = select(self.tables['boat'].c.boat_id).where(
            (self.tables['boat'].c.name == boat_name) &
            (self.tables['boat'].c.boat_class == boat_class)
        )
        result = conn.execute(sel).fetchone()
        if result:
            boat_id = result[0]
            # Update boat rank if needed
            upd = update(self.tables['boat']).where(
                self.tables['boat'].c.boat_id == boat_id
            ).values(
                boat_rank=boat_rank
            )
            conn.execute(upd)
            return boat_id
        else:
            ins = self.tables['boat'].insert().values(
                name=boat_name,
                boat_class=boat_class,
                boat_rank=boat_rank
            )
            result = conn.execute(ins)
            boat_id = result.inserted_primary_key[0]
            logger.info(f"Created boat: {boat_name} (class: {boat_class}, rank: {boat_rank})")
            return boat_id
            
    def _get_or_create_rower(self, conn, rower_name):
        """Get or create a rower in the database"""
        if not rower_name:
            return None
            
        sel = select(self.tables['rower'].c.rower_id).where(
            self.tables['rower'].c.name == rower_name
        )
        result = conn.execute(sel).fetchone()
        if result:
            return result[0]
        else:
            ins = self.tables['rower'].insert().values(
                name=rower_name
            )
            result = conn.execute(ins)
            rower_id = result.inserted_primary_key[0]
            logger.info(f"Created rower: {rower_name}")
            return rower_id
            
    def _create_piece(self, conn, event_id, piece_number):
        """Create a piece in the database"""
        description = f"Piece {piece_number}"
        ins = self.tables['piece'].insert().values(
            event_id=event_id,
            piece_number=piece_number,
            description=description
        )
        result = conn.execute(ins)
        piece_id = result.inserted_primary_key[0]
        logger.info(f"Created piece: {description}")
        return piece_id
        
    def _create_result(self, conn, piece_id, boat_id, result_data):
        """Create a result in the database"""
        if isinstance(result_data, dict):
            time_value = result_data.get('time')
            split_value = result_data.get('split')
            margin_value = result_data.get('margin')
        else:
            time_value = result_data
            split_value = None
            margin_value = None
            
        ins = self.tables['result'].insert().values(
            piece_id=piece_id,
            boat_id=boat_id,
            time=time_value,
            split=split_value,
            margin=margin_value
        )
        conn.execute(ins)
        logger.debug(f"Created result for boat {boat_id}, piece {piece_id}")
        
    def _create_lineup(self, conn, piece_id, boat_id, rower_id, seat_number, is_coxswain):
        """Create a lineup entry in the database"""
        if not rower_id:
            return
            
        ins = self.tables['lineup'].insert().values(
            piece_id=piece_id,
            boat_id=boat_id,
            rower_id=rower_id,
            seat_number=seat_number,
            is_coxswain=is_coxswain
        )
        conn.execute(ins)
        logger.debug(f"Added rower {rower_id} to boat {boat_id}, piece {piece_id}")
        
    def _create_seat_race(self, conn, event_id, seat_race_data):
        """Create a seat race entry in the database"""
        winner_id = self._get_or_create_rower(conn, seat_race_data['winner'])
        loser_id = self._get_or_create_rower(conn, seat_race_data['loser'])
        
        ins = self.tables['seat_race'].insert().values(
            event_id=event_id,
            piece_numbers=seat_race_data['pieces_switched'],
            rower_id_1=winner_id,
            rower_id_2=loser_id,
            time_difference=seat_race_data['margin'],
            winner_id=winner_id,
            notes=seat_race_data['notes']
        )
        conn.execute(ins)
        logger.info(f"Created seat race result: {seat_race_data['notes']}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    load_dotenv()
    password = os.getenv('PASSWORD')
    engine = create_engine(f'postgresql://postgres:{password}@localhost/rowing-analytics')
    
    parser = RowingDataParser(engine)
    data_dir = './data'
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv') and not filename.endswith('.csv:Zone.Identifier'):
            file_path = os.path.join(data_dir, filename)
            parsed_data = parser.parse_csv_file(file_path)
            if parsed_data:
                parser.load_data_to_db(parsed_data)