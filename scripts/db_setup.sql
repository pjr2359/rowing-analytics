-- database_setup.sql

-- Drop existing tables if they exist (be cautious with this in production)
DROP TABLE IF EXISTS seat_race CASCADE;
DROP TABLE IF EXISTS erg_data CASCADE;
DROP TABLE IF EXISTS result CASCADE;
DROP TABLE IF EXISTS lineup CASCADE;
DROP TABLE IF EXISTS piece CASCADE;
DROP TABLE IF EXISTS event CASCADE;
DROP TABLE IF EXISTS boat CASCADE;
DROP TABLE IF EXISTS rower CASCADE;

-- Create the 'rower' table
CREATE TABLE rower (
    rower_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    weight FLOAT,
    side VARCHAR(10)
);

-- Create the updated 'boat' table
CREATE TABLE boat (
    boat_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    boat_class VARCHAR(10), -- e.g., '8+', '4+', '4-'
    boat_rank INTEGER,      -- e.g., 1 for 1v, 2 for 2v
    UNIQUE(name, boat_class, boat_rank)
);

-- Create the 'event' table
CREATE TABLE event (
    event_id SERIAL PRIMARY KEY,
    event_date DATE ,
    event_name VARCHAR(100)
);

-- Create the 'piece' table
CREATE TABLE piece (
    piece_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES event(event_id),
    piece_number INTEGER,
    distance INTEGER, -- in meters
    description VARCHAR(255)
);

-- Create the 'lineup' table
CREATE TABLE lineup (
    lineup_id SERIAL PRIMARY KEY,
    piece_id INTEGER REFERENCES piece(piece_id),
    boat_id INTEGER REFERENCES boat(boat_id),
    rower_id INTEGER REFERENCES rower(rower_id),
    seat_number INTEGER,
    is_coxswain BOOLEAN DEFAULT FALSE
);

-- Create the 'result' table
CREATE TABLE result (
    result_id SERIAL PRIMARY KEY,
    piece_id INTEGER REFERENCES piece(piece_id),
    boat_id INTEGER REFERENCES boat(boat_id),
    time FLOAT,   -- total time in seconds
    split FLOAT,  -- split time in seconds
    margin FLOAT  -- margin in seconds
);

-- Create the 'erg_data' table
CREATE TABLE erg_data (
    erg_data_id SERIAL PRIMARY KEY,
    rower_id INTEGER REFERENCES rower(rower_id),
    test_date DATE,
    overall_split FLOAT,
    watts_per_lb FLOAT,
    weight FLOAT,
    pacing FLOAT[] -- Array of pacing intervals
);

-- Create the 'seat_race' table
CREATE TABLE seat_race (
    seat_race_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES event(event_id),
    piece_numbers INTEGER[],
    rower_id_1 INTEGER REFERENCES rower(rower_id),
    rower_id_2 INTEGER REFERENCES rower(rower_id),
    time_difference FLOAT,
    winner_id INTEGER REFERENCES rower(rower_id),
    notes TEXT
);
