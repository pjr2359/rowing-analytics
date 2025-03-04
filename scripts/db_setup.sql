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

CREATE TABLE rower (
    rower_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    weight FLOAT,
    side VARCHAR(10)
);

CREATE TABLE boat (
    boat_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    boat_class VARCHAR(10),
    boat_rank INTEGER
);

CREATE TABLE event (
    event_id SERIAL PRIMARY KEY,
    event_date DATE,
    event_name VARCHAR(100)
);

CREATE TABLE piece (
    piece_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES event(event_id),
    piece_number INTEGER,
    distance INTEGER,
    description VARCHAR(255)
);

CREATE TABLE lineup (
    lineup_id SERIAL PRIMARY KEY,
    piece_id INTEGER REFERENCES piece(piece_id),
    boat_id INTEGER REFERENCES boat(boat_id),
    rower_id INTEGER REFERENCES rower(rower_id),
    seat_number INTEGER,
    is_coxswain BOOLEAN DEFAULT FALSE
);

CREATE TABLE result (
    result_id SERIAL PRIMARY KEY,
    piece_id INTEGER REFERENCES piece(piece_id),
    boat_id INTEGER REFERENCES boat(boat_id),
    time FLOAT,
    split FLOAT,
    margin FLOAT
);

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