-- Create the 'rower' table
CREATE TABLE rower (
    rower_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    weight FLOAT,
    side VARCHAR(10)
);

-- Create the 'boat' table
CREATE TABLE boat (
    boat_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    boat_class VARCHAR(10) -- e.g., '8+', '4+', '4-'
);

-- Create the 'event' table
CREATE TABLE event (
    event_id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
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
    time FLOAT, -- total time in seconds
    split FLOAT, -- split time in seconds
    margin FLOAT -- margin in seconds
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
