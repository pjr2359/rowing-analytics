-- Table to store erg data
CREATE TABLE erg_data (
    id SERIAL PRIMARY KEY,
    side VARCHAR(10),
    name VARCHAR(100),
    overall_split FLOAT,
    watts_per_lb FLOAT,
    weight FLOAT,
    pacing_1st FLOAT,
    pacing_2nd FLOAT,
    pacing_3rd FLOAT
);

-- Table to store on-water race data
CREATE TABLE water_data (
    id SERIAL PRIMARY KEY,
    name_1 VARCHAR(100),
    lineup_1 VARCHAR(100),
    name_2 VARCHAR(100),
    lineup_2 VARCHAR(100),
    name_3 VARCHAR(100),
    lineup_3 VARCHAR(100),
    name_4 VARCHAR(100)
);

-- Example JOIN: If you want to query both datasets by rower name:
SELECT * FROM erg_data
JOIN water_data ON erg_data.name = water_data.name_1;
