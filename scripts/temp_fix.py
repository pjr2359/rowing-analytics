import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
password = os.getenv("PASSWORD")
engine = create_engine(f"postgresql://postgres:{password}@localhost:5432/rowing-analytics")

# Default Gold Medal Standard times (in seconds)
DEFAULT_GMS_TIMES = {
    '8+': 330.0,  # ~5:30 for 2k
    '4+': 365.0,  # ~6:05 for 2k
    '4x': 355.0,  # ~5:55 for 2k
    '2-': 395.0,  # ~6:35 for 2k
    '2x': 385.0,  # ~6:25 for 2k
    '1x': 420.0,  # ~7:00 for 2k
    '4-': 360.0,  # ~6:00 for 2k
    '2+': 400.0,  # ~6:40 for 2k
    'Unknown': 400.0
}

def fix_erg_data_processor():
    """Fix the erg data processor to handle the CSV formatting and schema mismatches"""
    with open('scripts/erg_data_processor.py', 'r') as file:
        content = file.read()
    
    # Add DEFAULT_GMS_TIMES after imports
    import_section_end = "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')"
    gms_times_def = """
# Default Gold Medal Standard times (in seconds)
DEFAULT_GMS_TIMES = {
    '8+': 330.0,  # ~5:30 for 2k
    '4+': 365.0,  # ~6:05 for 2k
    '4x': 355.0,  # ~5:55 for 2k
    '2-': 395.0,  # ~6:35 for 2k
    '2x': 385.0,  # ~6:25 for 2k
    '1x': 420.0,  # ~7:00 for 2k
    '4-': 360.0,  # ~6:00 for 2k
    '2+': 400.0,  # ~6:40 for 2k
    'Unknown': 400.0
}
"""
    if "DEFAULT_GMS_TIMES" not in content:
        content = content.replace(import_section_end, import_section_end + "\n" + gms_times_def)
    
    # Add DEFAULT_GMS_TIMES to ErgDataProcessor class
    class_init_end = "        self.erg_data = None\n        self.performance_data = None\n        self.combined_data = None"
    class_init_updated = class_init_end + "\n        self.DEFAULT_GMS_TIMES = DEFAULT_GMS_TIMES"
    content = content.replace(class_init_end, class_init_updated)
    
    # 1. Fix _time_to_seconds to handle various time formats
    old_time_function = '    def _time_to_seconds(self, time_str):\n        """Convert time string format (MM:SS.s) to seconds"""\n        if pd.isna(time_str):\n            return None\n            \n        # Handle different formats\n        if isinstance(time_str, (int, float)):\n            return float(time_str)\n            \n        # For format like \'1:45.2\'\n        try:\n            parts = time_str.replace(\',\', \'.\').split(\':\')\n            if len(parts) == 2:\n                mins, secs = parts\n                return float(mins) * 60 + float(secs)\n            elif len(parts) == 3:\n                hours, mins, secs = parts\n                return float(hours) * 3600 + float(mins) * 60 + float(secs)\n            else:\n                return float(time_str)\n        except:\n            # If all else fails, try direct conversion\n            try:\n                return float(time_str)\n            except:\n                return None'
    
    new_time_function = '    def _time_to_seconds(self, time_str):\n        """Convert time string format (MM:SS.s) to seconds"""\n        if pd.isna(time_str):\n            return None\n            \n        # Handle different formats\n        if isinstance(time_str, (int, float)):\n            return float(time_str)\n        \n        # Clean the string    \n        if isinstance(time_str, str):\n            # Handle formats like 6:29.1\n            time_str = time_str.replace(\',\', \'.\').strip()\n            \n            # Try parsing different formats\n            try:\n                parts = time_str.split(\':\')\n                if len(parts) == 2:\n                    mins, secs = parts\n                    return float(mins) * 60 + float(secs)\n                elif len(parts) == 3:\n                    hours, mins, secs = parts\n                    return float(hours) * 3600 + float(mins) * 60 + float(secs)\n                else:\n                    # Try direct conversion for single numbers\n                    return float(time_str)\n            except:\n                # If all else fails, try direct conversion or return None\n                try:\n                    return float(time_str)\n                except:\n                    logger.warning(f"Could not convert time: {time_str}")\n                    return None\n        return None'
    
    content = content.replace(old_time_function, new_time_function)
    
    # 2. Fix CSV parsing and test type detection
    old_load_erg = '    def load_erg_data(self, directory_path=\'./erg_data\'):\n        """Load all erg data files from the specified directory"""\n        all_data = []\n        \n        try:\n            for filename in os.listdir(directory_path):\n                if filename.endswith(\'.csv\'):\n                    file_path = os.path.join(directory_path, filename)\n                    logger.info(f"Processing erg file: {filename}")\n                    \n                    # Try to determine test type from filename\n                    test_type = \'unknown\'\n                    if \'2k\' in filename.lower():\n                        test_type = \'2k\'\n                    elif \'6k\' in filename.lower():\n                        test_type = \'6k\'\n                    elif \'30min\' in filename.lower() or \'30m\' in filename.lower():\n                        test_type = \'30min\''
    
    new_load_erg = '    def load_erg_data(self, directory_path=\'./erg_data\'):\n        """Load all erg data files from the specified directory"""\n        all_data = []\n        \n        try:\n            if not os.path.exists(directory_path):\n                logger.warning(f"Erg data directory \'{directory_path}\' does not exist")\n                self.erg_data = pd.DataFrame()\n                return self.erg_data\n                \n            self.erg_data_dir = directory_path\n        \n            # Define test type detection patterns    \n            test_type_patterns = {\n                \'2k\': [\'2k\', \'2000m\', \'2000\'],\n                \'5k\': [\'5k\', \'5000m\', \'5000\'],\n                \'6k\': [\'6k\', \'6000m\', \'6000\', \'2x6k\'],\n                \'30min\': [\'30min\', \'30m\', \'30\\\'\'],\n                \'4x10min\': [\'4x10\', \'4x10m\', \'4x10min\'],\n                \'3x3k\': [\'3x3k\', \'3x3000\'],\n                \'3x4k\': [\'3x4k\', \'3x4000\'],\n                \'4x2k\': [\'4x2k\', \'4x2000\']\n            }\n            \n            for filename in os.listdir(directory_path):\n                if filename.endswith(\'.csv\'):\n                    file_path = os.path.join(directory_path, filename)\n                    logger.info(f"Processing erg file: {filename}")\n                    \n                    # Try to determine test type from filename using patterns\n                    test_type = \'unknown\'\n                    for t_type, patterns in test_type_patterns.items():\n                        if any(pattern in filename.lower() for pattern in patterns):\n                            test_type = t_type\n                            break\n                            \n                    # Extract date from filename more robustly\n                    date_match = re.search(r\'(\\d{1,2})_(\\d{1,2})\', filename)\n                    if date_match:\n                        month, day = date_match.groups()\n                        # Determine year based on month (assuming school year)\n                        current_month = datetime.now().month\n                        year = datetime.now().year\n                        \n                        # If current month < 7 (July) and file month > 7, it\'s previous year\n                        if current_month < 7 and int(month) > 7:\n                            year -= 1\n                        # If current month > 7 and file month < 7, it\'s next year\n                        elif current_month > 7 and int(month) < 7:\n                            year += 1\n                            \n                        test_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"\n                    else:\n                        # Fallback to file modification date\n                        mod_time = os.path.getmtime(file_path)\n                        test_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")'
    
    content = content.replace(old_load_erg, new_load_erg)
    
    # 3. Fix CSV parsing to extract the right columns
    old_read_csv = '                    # Read the CSV file\n                    df = pd.read_csv(file_path)\n                    \n                    # Basic validation and cleanup\n                    if \'name\' in df.columns:\n                        name_col = \'name\'\n                    elif \'rower\' in df.columns:\n                        name_col = \'rower\'\n                    elif \'rower_name\' in df.columns:\n                        name_col = \'rower_name\'\n                    else:\n                        # Try to find a name column\n                        for col in df.columns:\n                            if \'name\' in col.lower():\n                                name_col = col\n                                break\n                        else:\n                            logger.warning(f"Could not find name column in {filename}")\n                            continue'
                    
    new_read_csv = '                    # Read the CSV file with flexible parsing\n                    try:\n                        df = pd.read_csv(file_path, na_values=[\'\', \'NA\', \'DNS\', \'#REF!\', \'DNF\'])\n                    except Exception as e:\n                        logger.warning(f"Error reading {filename}: {e}")\n                        continue\n                    \n                    # Clean up column names (lowercase, strip whitespace)\n                    df.columns = [col.lower().strip() for col in df.columns]\n                    \n                    # Basic validation and cleanup\n                    # Look for name column with multiple possible names\n                    name_col = None\n                    for possible_name in [\'name\', \'rower\', \'rower_name\']:\n                        if possible_name in df.columns:\n                            name_col = possible_name\n                            break\n                    \n                    # If not found by exact match, try contains\n                    if name_col is None:\n                        for col in df.columns:\n                            if \'name\' in col.lower():\n                                name_col = col\n                                break\n                                \n                    # If still no name column, skip this file\n                    if name_col is None:\n                        logger.warning(f"Could not find name column in {filename}")\n                        continue\n                        \n                    # Filter out rows with missing names or special rows like \'BIKE\', \'RX\'\n                    df = df[df[name_col].notna()]\n                    df = df[~df[name_col].str.contains(\'BIKE|RX|Mgr|Cox\', case=False, na=False)]'
                    
    content = content.replace(old_read_csv, new_read_csv)
    
    # 4. Fix time/split column detection and standardize test_type
    old_time_col = '                    # Standardize time columns\n                    time_col = None\n                    for col in df.columns:\n                        if \'time\' in col.lower() or \'split\' in col.lower():\n                            time_col = col\n                            break\n                            \n                    if time_col is not None:\n                        df = df.rename(columns={time_col: \'erg_time\'})\n                        \n                        # Convert time strings to seconds if not already\n                        if df[\'erg_time\'].dtype == object:\n                            df[\'erg_time_seconds\'] = df[\'erg_time\'].apply(self._time_to_seconds)\n                        else:\n                            df[\'erg_time_seconds\'] = df[\'erg_time\']'
    
    new_time_col = '                    # Standardize time columns - search for split or time\n                    time_col = None\n                    for col_pattern in [\'time\', \'split\', \'overall\']:\n                        for col in df.columns:\n                            if col_pattern in col.lower():\n                                time_col = col\n                                break\n                        if time_col:\n                            break\n                            \n                    if time_col is not None:\n                        df = df.rename(columns={time_col: \'erg_time\'})\n                        \n                        # Convert time strings to seconds\n                        df[\'erg_time_seconds\'] = df[\'erg_time\'].apply(self._time_to_seconds)\n                    else:\n                        logger.warning(f"No time/split column found in {filename}")\n                        continue\n                        \n                    # Look for weight column\n                    weight_col = None\n                    for col in df.columns:\n                        if \'weight\' in col.lower():\n                            weight_col = col\n                            break\n                            \n                    if weight_col:\n                        df[\'weight\'] = pd.to_numeric(df[weight_col], errors=\'coerce\')\n                        \n                    # Look for power-to-weight ratio column\n                    p2w_col = None\n                    for col in df.columns:\n                        if \'watt\' in col.lower() and \'lb\' in col.lower():\n                            p2w_col = col\n                            break\n                            \n                    if p2w_col:\n                        df[\'watts_per_lb\'] = pd.to_numeric(df[p2w_col], errors=\'coerce\')\n                        \n                    # Add side preference if available (P/S)\n                    side_col = None\n                    for col in df.columns:\n                        if col.lower() == \'side\':\n                            side_col = col\n                            break\n                            \n                    if side_col:\n                        df[\'side\'] = df[side_col]'
    
    content = content.replace(old_time_col, new_time_col)
    
    # 5. Fix date handling in test_date
    additional_date_fix = '                    # Add test date column with proper datetime format\n                    df[\'test_date\'] = pd.to_datetime(test_date)\n                    \n                    # Add test type column\n                    df[\'test_type\'] = test_type\n                    \n                    # Add source filename\n                    df[\'source_file\'] = filename\n                    \n                    # Filter out rows with missing erg times\n                    df = df[df[\'erg_time_seconds\'].notna()]\n                    \n                    all_data.append(df)'
                    
    content = content.replace("                    # Add test type column\n                    df['test_type'] = test_type", additional_date_fix)
    
    # 6. Fix the _get_recent_erg_scores method to safely handle missing data
    old_get_recent = '    def _get_recent_erg_scores(self, rower_names):\n      """Get the most recent erg scores for a list of rowers"""\n      lookup = {}\n      \n      if self.erg_data is None:\n          self.load_erg_data()\n          \n      if self.erg_data is None or self.erg_data.empty:\n          return lookup\n          \n      for rower in rower_names:\n          # Find all erg tests for this rower\n          rower_ergs = self.erg_data[self.erg_data[\'rower_name\'] == rower]\n          \n          if rower_ergs.empty:\n              logger.warning(f"No erg data found for {rower}")\n              continue\n              \n          # Group by test type\n          erg_history = []\n          for _, row in rower_ergs.iterrows():\n              erg_history.append({\n                  \'date\': row[\'test_date\'],\n                  \'score\': row[\'erg_time_seconds\'],\n                  \'weight\': row.get(\'weight\', None),\n                  \'power_to_weight\': row.get(\'watts_per_lb\', None),\n                  \'test_type\': row[\'test_type\']\n              })\n              \n          lookup[rower] = sorted(erg_history, key=lambda x: x[\'date\'], reverse=True)\n          \n      return lookup'
      
    new_get_recent = '    def _get_recent_erg_scores(self, rower_names):\n        """Get the most recent erg scores for a list of rowers"""\n        lookup = {}\n        \n        if self.erg_data is None:\n            self.load_erg_data()\n            \n        if self.erg_data is None or self.erg_data.empty:\n            return lookup\n            \n        for rower in rower_names:\n            # Find all erg tests for this rower\n            rower_ergs = self.erg_data[self.erg_data[\'rower_name\'] == rower]\n            \n            if rower_ergs.empty:\n                logger.warning(f"No erg data found for {rower}")\n                continue\n                \n            # Group by test type\n            erg_history = []\n            for _, row in rower_ergs.iterrows():\n                # Ensure test_date is properly converted to datetime\n                try:\n                    test_date = pd.to_datetime(row[\'test_date\'])\n                except:\n                    logger.warning(f"Invalid test date for {rower}: {row[\'test_date\']}")\n                    test_date = pd.Timestamp.now()\n                    \n                # Safely extract values with defaults\n                erg_time = row[\'erg_time_seconds\'] if \'erg_time_seconds\' in row and pd.notna(row[\'erg_time_seconds\']) else None\n                weight = row[\'weight\'] if \'weight\' in row and pd.notna(row[\'weight\']) else None\n                p2w = row[\'watts_per_lb\'] if \'watts_per_lb\' in row and pd.notna(row[\'watts_per_lb\']) else None\n                test_type = row[\'test_type\'] if \'test_type\' in row else \'unknown\'\n                \n                if erg_time is not None:\n                    erg_history.append({\n                        \'date\': test_date,\n                        \'score\': float(erg_time),\n                        \'weight\': float(weight) if weight is not None else None,\n                        \'power_to_weight\': float(p2w) if p2w is not None else None,\n                        \'test_type\': test_type\n                    })\n                \n            # Sort by date with most recent first    \n            lookup[rower] = sorted(erg_history, key=lambda x: x[\'date\'], reverse=True)\n            \n        return lookup'
        
    content = content.replace(old_get_recent, new_get_recent)
    
    # 7. Fix the combine_data_sources method to properly handle missing data
    old_combine = '    def combine_data_sources(self, erg_days_lookback=180):\n        """Combine erg data with on-water performance data"""\n        if self.erg_data is None:\n            self.load_erg_data()\n            \n        if self.performance_data is None:\n            self.load_on_water_data()\n\n        if self.erg_data is None or self.performance_data is None:\n            logger.error("Failed to load one or both data sources")\n            return pd.DataFrame()\n            \n        if self.erg_data.empty or self.performance_data.empty:\n            logger.error("Cannot combine data: one or both sources are empty")\n            return pd.DataFrame()'
            
    new_combine = '    def combine_data_sources(self, erg_days_lookback=180):\n        """Combine erg data with on-water performance data"""\n        if self.erg_data is None:\n            self.load_erg_data()\n            \n        if self.performance_data is None:\n            self.load_on_water_data()\n\n        if self.erg_data is None or self.performance_data is None:\n            logger.error("Failed to load one or both data sources")\n            return None\n            \n        if self.erg_data.empty or self.performance_data.empty:\n            logger.error("Cannot combine data: one or both sources are empty")\n            return self.performance_data.copy() if not self.performance_data.empty else None'
            
    content = content.replace(old_combine, new_combine)
    
    # 8. Fix analyze_seat_race_with_erg_context to handle None values
    old_seat_race = '                    result_row[\'rower1_erg\'] = erg1[\'score\']\n                    result_row[\'rower1_erg_date\'] = erg1[\'date\']\n                    result_row[\'rower2_erg\'] = erg2[\'score\']\n                    result_row[\'rower2_erg_date\'] = erg2[\'date\']\n                    result_row[\'erg_difference\'] = erg2[\'score\'] - erg1[\'score\']  # Positive = rower1 faster\n                    \n                    # Add physiological context\n                    if \'weight\' in erg1 and \'weight\' in erg2:\n                        result_row[\'rower1_weight\'] = erg1[\'weight\']\n                        result_row[\'rower2_weight\'] = erg2[\'weight\']\n                        result_row[\'weight_difference\'] = erg2[\'weight\'] - erg1[\'weight\']'
    
    new_seat_race = '                    result_row[\'rower1_erg\'] = erg1[\'score\']\n                    result_row[\'rower1_erg_date\'] = erg1[\'date\']\n                    result_row[\'rower2_erg\'] = erg2[\'score\']\n                    result_row[\'rower2_erg_date\'] = erg2[\'date\']\n                    result_row[\'erg_difference\'] = erg2[\'score\'] - erg1[\'score\']  # Positive = rower1 faster\n                    \n                    # Add physiological context with safe handling of None values\n                    result_row[\'rower1_weight\'] = erg1.get(\'weight\')\n                    result_row[\'rower2_weight\'] = erg2.get(\'weight\')\n                    \n                    # Only calculate differences if both weights are available\n                    if erg1.get(\'weight\') is not None and erg2.get(\'weight\') is not None:\n                        result_row[\'weight_difference\'] = erg2[\'weight\'] - erg1[\'weight\']\n                    else:\n                        result_row[\'weight_difference\'] = None'
    
    content = content.replace(old_seat_race, new_seat_race)
    
    # Add GMS fallback in analyze_boat_class_specialists
    old_gms = '        try:\n            from rowing_analysis import GMS_TIMES\n        except ImportError:\n            # Define basic GMS times if not available\n            GMS_TIMES = {'
            
    new_gms = '        try:\n            from rowing_analysis import GMS_TIMES\n        except ImportError:\n            # Use default GMS times if not available\n            GMS_TIMES = self.DEFAULT_GMS_TIMES\n        except AttributeError:\n            # Use default GMS times if not available\n            GMS_TIMES = self.DEFAULT_GMS_TIMES'
            
    content = content.replace(old_gms, new_gms)
    
    # Write the updated file
    with open('scripts/erg_data_processor.py', 'w') as file:
        file.write(content)
    
    logger.info("Fixed erg_data_processor.py")
    
if __name__ == "__main__":
    fix_erg_data_processor()
    print("Data mismatches fixed. Now run your analysis again.")