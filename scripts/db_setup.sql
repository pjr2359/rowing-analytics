-- db_setup.sql

-- Drop existing tables if they exist (be cautious with this in production)
DROP TABLE IF EXISTS seat_race CASCADE;
DROP TABLE IF EXISTS erg_data CASCADE;
DROP TABLE IF EXISTS result CASCADE;
DROP TABLE IF EXISTS lineup CASCADE;
DROP TABLE IF EXISTS piece CASCADE;
DROP TABLE IF EXISTS event CASCADE;
DROP TABLE IF EXISTS boat CASCADE;
DROP TABLE IF EXISTS rower CASCADE;

-- Create tables with improved schema

-- Rower table: Store information about individual rowers
CREATE TABLE rower (
    rower_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    weight FLOAT,
    side VARCHAR(10),  -- Port/Starboard preference
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Boat table: Store information about boats
CREATE TABLE boat (
    boat_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    boat_class VARCHAR(10),  -- Format: 8+, 4-, etc.
    boat_rank INTEGER,       -- Relative rank of boat (1st varsity, 2nd varsity, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Event table: Store information about rowing events/practices
CREATE TABLE event (
    event_id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    event_name VARCHAR(100) NOT NULL,
    event_type VARCHAR(50), -- Race, Practice, etc.
    location VARCHAR(100),
    distance INTEGER,        -- Standard distance if applicable
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Piece table: Store information about individual pieces within an event
CREATE TABLE piece (
    piece_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES event(event_id) ON DELETE CASCADE,
    piece_number INTEGER NOT NULL,
    distance INTEGER,        -- Distance of this specific piece
    description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lineup table: Store which rowers were in which boats for which pieces
CREATE TABLE lineup (
    lineup_id SERIAL PRIMARY KEY,
    piece_id INTEGER REFERENCES piece(piece_id) ON DELETE CASCADE,
    boat_id INTEGER REFERENCES boat(boat_id) ON DELETE CASCADE,
    rower_id INTEGER REFERENCES rower(rower_id) ON DELETE CASCADE,
    seat_number INTEGER,     -- 0 for coxswain, 1-8 for rowers
    is_coxswain BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Result table: Store performance data for each boat in each piece
CREATE TABLE result (
    result_id SERIAL PRIMARY KEY,
    piece_id INTEGER REFERENCES piece(piece_id) ON DELETE CASCADE,
    boat_id INTEGER REFERENCES boat(boat_id) ON DELETE CASCADE,
    time FLOAT,              -- Time in seconds
    split FLOAT,             -- Split time in seconds
    margin FLOAT,            -- Margin compared to leading boat
    meters INTEGER,          -- Distance completed if variable
    ranking INTEGER,         -- Finishing position
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seat Race table: Store results of seat racing
CREATE TABLE seat_race (
    seat_race_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES event(event_id) ON DELETE CASCADE,
    piece_numbers INTEGER[], -- Array of piece numbers involved in seat race
    rower_id_1 INTEGER REFERENCES rower(rower_id) ON DELETE CASCADE,
    rower_id_2 INTEGER REFERENCES rower(rower_id) ON DELETE CASCADE,
    time_difference FLOAT,   -- Time difference in seconds
    winner_id INTEGER REFERENCES rower(rower_id) ON DELETE CASCADE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Erg Data table: Store ergometer test results
CREATE TABLE erg_data (
    erg_data_id SERIAL PRIMARY KEY,
    rower_id INTEGER REFERENCES rower(rower_id) ON DELETE CASCADE,
    test_date DATE NOT NULL,
    test_type VARCHAR(50),   -- 2k, 6k, etc.
    overall_split FLOAT,     -- Average split time in seconds
    watts_per_lb FLOAT,
    weight FLOAT,            -- Rower's weight at time of test
    pacing FLOAT[],          -- Array of split times for each segment
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for improved query performance
CREATE INDEX idx_lineup_piece_boat ON lineup(piece_id, boat_id);
CREATE INDEX idx_lineup_rower ON lineup(rower_id);
CREATE INDEX idx_result_piece ON result(piece_id);
CREATE INDEX idx_result_boat ON result(boat_id);
CREATE INDEX idx_piece_event ON piece(event_id);
CREATE INDEX idx_seat_race_event ON seat_race(event_id);
CREATE INDEX idx_erg_data_rower ON erg_data(rower_id);

-- Add update trigger function to maintain updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for each table
CREATE TRIGGER update_rower_timestamp BEFORE UPDATE ON rower FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_boat_timestamp BEFORE UPDATE ON boat FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_event_timestamp BEFORE UPDATE ON event FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_piece_timestamp BEFORE UPDATE ON piece FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_lineup_timestamp BEFORE UPDATE ON lineup FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_result_timestamp BEFORE UPDATE ON result FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_seat_race_timestamp BEFORE UPDATE ON seat_race FOR EACH ROW EXECUTE FUNCTION update_timestamp();
CREATE TRIGGER update_erg_data_timestamp BEFORE UPDATE ON erg_data FOR EACH ROW EXECUTE FUNCTION update_timestamp();