Rowing Analytics Project
The Rowing Analytics Project is designed to parse rowing performance data from CSV files, load it into a PostgreSQL database, and analyze the data to rank rowers based on their performance relative to Gold Medal Standard (GMS) times.


This project helps coaches and athletes by:

Parsing CSV files containing race results and lineup information.
Loading parsed data into a structured PostgreSQL database.
Analyzing stored data to compute average percentages off GMS times for each rower.
Visualizing rankings through graphical representations.
Prerequisites
Python 3.7+
PostgreSQL 12+
Git (optional, for cloning the repository)
Setup Instructions
Clone the Repository


git clone https://github.com/yourusername/rowing-analytics.git
cd rowing-analytics
Set Up a Python Virtual Environment


python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install Dependencies


pip install -r requirements.txt
Install PostgreSQL

Download and install PostgreSQL from the official website.

Configure the Database

Create Database


psql -U postgres
CREATE DATABASE "rowing-analytics";
\q
Set Up Database Schema


psql -U postgres -d "rowing-analytics" -f scripts/database_setup.sql
Set Up Environment Variables

Create a .env file in the scripts/ directory:


cd scripts
touch .env
Add your PostgreSQL password to .env:

makefile
Copy code
PASSWORD=YourPostgresPassword
Prepare Data Files

Place your CSV data files in the data/ directory.
Running the Project
1. Parse and Load Data
Navigate to the scripts/ directory and run:


python parse_and_load.py
Parses CSV files in data/ and loads data into the database.
2. Analyze the Data
From the scripts/ directory, run:


python rowing_analysis.py
Retrieves data from the database.
Computes average percentage off GMS for each rower.
Displays a bar plot of the rankings.
Script Overview
parse_and_load.py
Purpose: Parses CSV files and loads data into the database.
Key Functions:
parse_csv(): Reads CSV files.
get_or_create_rower_id(): Manages rower IDs.
convert_time_to_seconds(): Converts time formats.
parse_data(): Processes and inserts data.
Usage: Run to parse all CSV files in data/ and populate the database.
rowing_analysis.py
Purpose: Analyzes performance data and ranks rowers.
Key Functions:
get_rower_performance(): Retrieves data from the database.
compute_percentage_off_gms(): Calculates average percentages off GMS.
visualize_rankings(): Generates a bar plot of rankings.
Usage: Run to perform analysis and display rankings.
Dependencies
psycopg2-binary
pandas
sqlalchemy
python-dotenv
matplotlib
seaborn
Install all dependencies using:


pip install -r requirements.txt
Troubleshooting
No Data Loaded: Ensure CSV files are correctly formatted and placed in data/.
Database Connection Errors: Confirm PostgreSQL is running and credentials in .env are correct.
Missing Dependencies: Install via pip install -r requirements.txt.
Visualization Issues: Ensure matplotlib and seaborn are installed.
License
This project is licensed under the MIT License.
