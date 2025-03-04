# rowing_analysis.py

import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

password = os.getenv('PASSWORD')
engine = create_engine('postgresql://postgres:'+password+'@localhost/rowing-analytics')

GMS_TIMES = {
    '1v 8+': 5 * 60 + 40,
    '2v 8+': 5 * 60 + 46,
    '3v 8+': 5 * 60 + 52,
    '4v 8+': 6 * 60 + 0,
    '1v 4+': 6 * 60 + 8,
    '2v 4+': 6 * 60 + 14,
    '3v 4+': 6 * 60 + 20,
    '4v 4+': 6 * 60 + 26,
    '5v 4+': 6 * 60 + 32,
    '1v 4-': 6 * 60 + 3,
    '2v 4-': 6 * 60 + 9,
    '3v 4-': 6 * 60 + 15,
    '4v 4-': 6 * 60 + 21,
    '5v 4-': 6 * 60 + 27,
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
    try:
        logger.debug("Executing SQL query to fetch performance data.")
        df = pd.read_sql_query(query, engine)
        logger.info(f"Number of performance records fetched: {len(df)}")
        logger.debug(f"Performance data:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Error fetching performance data: {e}")
        return pd.DataFrame()

def compute_percentage_off_gms(df):
    """
    Computes the average percentage off GMS for each rower based on boat class and rank.
    """

    df['boat_identifier'] = df['boat_rank'].astype(str) + 'v ' + df['boat_class']

    df['GMS_time'] = df['boat_identifier'].map(GMS_TIMES)


    df = df.dropna(subset=['GMS_time'])

    df['pct_off_gms'] = ((df['time'] - df['GMS_time']) / df['GMS_time']) * 100

    avg_pct_off = df.groupby(['rower_id', 'rower_name'])['pct_off_gms'].mean().reset_index()

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

    df_performance = get_rower_performance(engine)
    if df_performance.empty:
        print("No performance data found.")
        return

    avg_pct_off = compute_percentage_off_gms(df_performance)
    print("Average Percentage Off GMS per Rower:")
    print(avg_pct_off)

    visualize_rankings(avg_pct_off)

if __name__ == "__main__":
    main()
