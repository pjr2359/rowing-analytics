import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
password = os.getenv("PASSWORD")
engine = create_engine(f"postgresql://postgres:{password}@localhost:5432/rowing-analytics")

def fix_erg_data_processor():
    # Update erg_data_processor.py to use correct schema
    with open('scripts/erg_data_processor.py', 'r') as file:
        content = file.read()
        
    # Fix 1: Update rower name standardization query
    content = content.replace(
        "SELECT rower_id, first_name, last_name FROM rower;",
        "SELECT rower_id, name FROM rower;"
    )
    
    # Fix 2: Update the rower name matching logic
    content = content.replace(
        "                full_name = f\"{rower['first_name']} {rower['last_name']}\".lower()\n                last_name = rower['last_name'].lower()",
        "                full_name = rower['name'].lower()\n                last_name = rower['name'].split()[-1].lower() if ' ' in rower['name'] else rower['name'].lower()"
    )
    
    # Fix 3: Update return value in rower name matching
    content = content.replace(
        "                    return db_rowers.loc[db_rowers['rower_id'] == rower_lookup[clean_name], \n                                        'last_name'].values[0]",
        "                    return db_rowers.loc[db_rowers['rower_id'] == rower_lookup[clean_name], \n                                        'name'].values[0]"
    )
    
    # Fix 4: Update the water data query JOIN condition
    content = content.replace(
        "                JOIN lineup l ON r.result_id = l.result_id",
        "                JOIN lineup l ON r.boat_id = l.boat_id AND r.piece_id = l.piece_id"
    )
    
    # Fix 5: Prevent NoneType errors - use regular strings instead of triple quotes to avoid nesting issues
    search_pattern = "    def combine_data_sources(self):\n        \"\"\"Combine erg and on-water data sources\"\"\"\n        self._standardize_rower_names()\n        self._load_performance_data()\n        \n        if self.erg_data.empty or self.performance_data.empty:"
    replace_pattern = "    def combine_data_sources(self):\n        \"\"\"Combine erg and on-water data sources\"\"\"\n        self._standardize_rower_names()\n        self._load_performance_data()\n        \n        if self.erg_data is None or self.performance_data is None:\n            return None\n            \n        if self.erg_data.empty or self.performance_data.empty:"
    
    content = content.replace(search_pattern, replace_pattern)
    
    # Fix 6: Update on-water data query to use 'name' instead of 'last_name'
    content = content.replace(
        "                rwr.last_name AS rower_name,",
        "                rwr.name AS rower_name,"
    )
    
    # Fix 7: Fix compatibility_scores function
    content = content.replace(
        "        r1.last_name as rower1_name,",
        "        r1.name as rower1_name,"
    )
    content = content.replace(
        "        r2.last_name as rower2_name,",
        "        r2.name as rower2_name,"
    )
    content = content.replace(
        "        r1.last_name IN :rowers AND r2.last_name IN :rowers",
        "        r1.name IN :rowers AND r2.name IN :rowers"
    )
    
    # Fix 8: Update event table column name from 'date' to 'event_date'
    content = content.replace(
        "                evt.date AS event_date,",
        "                evt.event_date AS event_date,"
    )
    content = content.replace(
        "                evt.date DESC,",
        "                evt.event_date DESC,"
    )
    
    # Fix 9: Update seat race query
    content = content.replace(
        "            e.date AS race_date,",
        "            e.event_date AS race_date,"
    )
    
    # Fix 10: Update any other references to last_name in seat_race analysis
    content = content.replace(
        "            r1.last_name AS rower1_name,",
        "            r1.name AS rower1_name,"
    )
    content = content.replace(
        "            r2.last_name AS rower2_name,",
        "            r2.name AS rower2_name,"
    )
    content = content.replace(
        "            winner.last_name AS winner_name,",
        "            winner.name AS winner_name,"
    )
    
    # Fix 11: Update the _load_performance_data method with robust error handling
    search_pattern = "    def _load_performance_data(self):\n        \"\"\"Load on-water performance data from database\"\"\"\n        if self.performance_data is None:\n            self.performance_data = self.load_on_water_data()\n        return self.performance_data"
    replace_pattern = "    def _load_performance_data(self):\n        \"\"\"Load on-water performance data from database\"\"\"\n        if self.performance_data is None:\n            try:\n                self.performance_data = self.load_on_water_data()\n            except Exception as e:\n                logger.error(f\"Error loading on-water data: {e}\")\n                self.performance_data = None\n        return self.performance_data"
    
    content = content.replace(search_pattern, replace_pattern)
    
    # Write updated file
    with open('scripts/erg_data_processor.py', 'w') as file:
        file.write(content)
    
    print("Updated erg_data_processor.py to match database schema")

def fix_rowing_analysis():
    # Now update rowing_analysis.py too
    with open('scripts/rowing_analysis.py', 'r') as file:
        content = file.read()
        
    # Fix event.date references
    content = content.replace(
        "            e.event_date >= '{cutoff_date}'",
        "            e.event_date >= '{cutoff_date}'"
    )
    content = content.replace(
        "            e.date >= '{cutoff_date}'",
        "            e.event_date >= '{cutoff_date}'"
    )
    content = content.replace(
        "ORDER BY e.date DESC",
        "ORDER BY e.event_date DESC"
    )
    
    # Write updated file
    with open('scripts/rowing_analysis.py', 'w') as file:
        file.write(content)
    
    print("Updated rowing_analysis.py to match database schema")

if __name__ == "__main__":
    fix_erg_data_processor()
    fix_rowing_analysis()