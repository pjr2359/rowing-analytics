# rowing_analysis.py

import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Database connection setup
engine = create_engine('postgresql://postgres:Conan_Stephens27@localhost/rowing-analytics')

# GMS times for boat classes and rankings (in seconds)
GMS_TIMES = {
    # 8+ Boats
    '1v 8+': 5 * 60 + 40,  # 5:40 -> 340 seconds
    '2v 8+': 5 * 60 + 46,  # 5:46 -> 346 seconds
    '3v 8+': 5 * 60 + 52,  # 5:52 -> 352 seconds
    '4v 8+': 6 * 60 + 0,   # 6:00 -> 360 seconds

    # 4+ Boats
    '1v 4+': 6 * 60 + 8,   # 6:08 -> 368 seconds
    '2v 4+': 6 * 60 + 14,  # 6:14 -> 374 seconds
    '3v 4+': 6 * 60 + 20,  # 6:20 -> 380 seconds

    # 4- Boats
    '1v 4-': 6 * 60 + 3,   # 6:03 -> 363 seconds
    '2v 4-': 6 * 60 + 9,   # 6:09 -> 369 seconds
    '3v 4-': 6 * 60 + 15,  # 6:15 -> 375 seconds
}

def get_rower_performance(engine):
    """
    Retrieves performance data for each rower by joining relevant tables.
    """
    query = """
    SELECT
        r.rower_id,
        r.name AS rower_name,
        res.time,
        res.margin,
        b.boat_class,
        b.boat_rank,
        p.piece_id,
        p.piece_number,
        e.event_id,
        e.event_name,
        e.event_date
    FROM
        lineup l
    JOIN
        rower r ON l.rower_id = r.rower_id
    JOIN
        boat b ON l.boat_id = b.boat_id
    JOIN
        result res ON l.piece_id = res.piece_id AND l.boat_id = res.boat_id
    JOIN
        piece p ON res.piece_id = p.piece_id
    JOIN
        event e ON p.event_id = e.event_id
    WHERE
        res.time IS NOT NULL
    ORDER BY
        r.rower_id, p.piece_number;
    """
    df = pd.read_sql_query(query, engine)
    return df

def compute_percentage_off_gms(df):
    """
    Computes the average percentage off GMS for each rower based on boat class and rank.
    """
    # Create a boat identifier combining boat_class and boat_rank
    df['boat_identifier'] = df['boat_rank'].astype(str) + 'v ' + df['boat_class']

    # Add GMS times to the dataframe
    df['GMS_time'] = df['boat_identifier'].map(GMS_TIMES)

    # Drop rows where GMS_time is missing
    df = df.dropna(subset=['GMS_time'])

    # Calculate percentage off GMS for each row
    df['pct_off_gms'] = ((df['time'] - df['GMS_time']) / df['GMS_time']) * 100

    # Group by rower and calculate the mean percentage off GMS
    avg_pct_off = df.groupby(['rower_id', 'rower_name'])['pct_off_gms'].mean().reset_index()
    # Rank rowers based on average percentage off GMS (lower is better)
    avg_pct_off['rank'] = avg_pct_off['pct_off_gms'].rank(method='min')
    avg_pct_off.sort_values('rank', inplace=True)
    return avg_pct_off

def visualize_rankings(avg_pct_off):
    """
    Visualizes the rankings of rowers based on average percentage off GMS.
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='pct_off_gms',
        y='rower_name',
        data=avg_pct_off,
        palette='viridis',
        order=avg_pct_off.sort_values('pct_off_gms')['rower_name']
    )
    plt.xlabel('Average Percentage Off GMS (%)')
    plt.ylabel('Rower Name')
    plt.title('Rower Rankings Based on Average Percentage Off GMS')
    plt.tight_layout()
    plt.show()

def main():
    # Step 1: Retrieve data
    df_performance = get_rower_performance(engine)
    if df_performance.empty:
        print("No performance data found.")
        return

    # Step 2: Compute average percentage off GMS
    avg_pct_off = compute_percentage_off_gms(df_performance)
    print("Average Percentage Off GMS per Rower:")
    print(avg_pct_off)

    # Step 3: Visualize the rankings
    visualize_rankings(avg_pct_off)

if __name__ == "__main__":
    main()
