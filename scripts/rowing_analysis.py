# improved_rowing_analysis.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
password = os.getenv('PASSWORD')
os.makedirs('analysis_output', exist_ok=True)

# Gold Medal Standard (GMS) times in seconds for different boat classes and ranks
GMS_TIMES = {
    '1v 8+': 5 * 60 + 40,  # 5:40
    '2v 8+': 5 * 60 + 46,  # 5:46
    '3v 8+': 5 * 60 + 52,  # 5:52
    '4v 8+': 6 * 60 + 0,   # 6:00
    '1v 4+': 6 * 60 + 8,   # 6:08
    '2v 4+': 6 * 60 + 14,  # 6:14
    '3v 4+': 6 * 60 + 20,  # 6:20
    '4v 4+': 6 * 60 + 26,  # 6:26
    '5v 4+': 6 * 60 + 32,  # 6:32
    '1v 4-': 6 * 60 + 3,   # 6:03
    '2v 4-': 6 * 60 + 9,   # 6:09
    '3v 4-': 6 * 60 + 15,  # 6:15
    '4v 4-': 6 * 60 + 21,  # 6:21
    '5v 4-': 6 * 60 + 27,  # 6:27
}

class RowingAnalysis:
    def __init__(self, db_connection_string=None):
        """Initialize the analysis with database connection"""
        if db_connection_string is None:
            password = os.getenv('PASSWORD')
            if not password:
                raise ValueError("Database password not found in environment variables")
            db_connection_string = f'postgresql://postgres:{password}@localhost/rowing-analytics'
            
        try:
            self.engine = create_engine(db_connection_string)
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.fetchone()[0] != 1:
                    raise ConnectionError("Database connection test failed")
                logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
            
        # Initialize data frames to None (will be loaded on demand)
        self.rower_data = None
        self.performance_data = None
        self.seat_race_data = None
        self.gms_comparison_data = None
        
    def load_performance_data(self, days_back=None):
        """
        Load performance data from database.
        Optional: Filter for events within the last X days.
        """
        query = """
        SELECT
            r.rower_id,
            r.name AS rower_name,
            r.weight,
            r.side,
            res.time,
            res.split,
            res.margin,
            b.boat_id,
            b.name AS boat_name,
            b.boat_class,
            b.boat_rank,
            p.piece_id,
            p.piece_number,
            p.distance,
            e.event_id,
            e.event_name,
            e.event_date,
            l.seat_number,
            l.is_coxswain
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
        """
        
        # Add date filter if requested
        if days_back is not None:
            cutoff_date = datetime.now().date() - timedelta(days=days_back)
            query += f"\nAND e.event_date >= '{cutoff_date}'"
            
        query += "\nORDER BY e.event_date DESC, p.piece_number, r.rower_id"
        
        try:
            logger.info("Loading performance data from database")
            self.performance_data = pd.read_sql_query(query, self.engine)
            
            # Add useful derived columns
            if not self.performance_data.empty:
                # Create boat identifier that matches GMS_TIMES keys
                self.performance_data['boat_identifier'] = (
                    self.performance_data['boat_rank'].astype(str) + 'v ' + 
                    self.performance_data['boat_class']
                )
                
                # Convert time to minutes:seconds format for display
                self.performance_data['time_formatted'] = self.performance_data['time'].apply(
                    lambda x: f"{int(x // 60)}:{x % 60:05.2f}" if pd.notnull(x) else None
                )
                
                # Convert split to minutes:seconds format for display
                self.performance_data['split_formatted'] = self.performance_data['split'].apply(
                    lambda x: f"{int(x // 60)}:{x % 60:05.2f}" if pd.notnull(x) else None
                )
                
            logger.info(f"Loaded {len(self.performance_data)} performance records")
            return self.performance_data
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            return pd.DataFrame()
    
    def load_seat_race_data(self):
        """Load seat race data from database"""
        query = """
        SELECT
            sr.seat_race_id,
            sr.event_id,
            e.event_date,
            e.event_name,
            sr.piece_numbers,
            r1.rower_id AS rower_id_1,
            r1.name AS rower1_name,
            r2.rower_id AS rower_id_2,
            r2.name AS rower2_name,
            sr.time_difference,
            w.rower_id AS winner_id,
            w.name AS winner_name,
            sr.notes
        FROM
            seat_race sr
        JOIN
            event e ON sr.event_id = e.event_id
        JOIN
            rower r1 ON sr.rower_id_1 = r1.rower_id
        JOIN
            rower r2 ON sr.rower_id_2 = r2.rower_id
        JOIN
            rower w ON sr.winner_id = w.rower_id
        ORDER BY
            e.event_date DESC
        """
        
        try:
            logger.info("Loading seat race data from database")
            self.seat_race_data = pd.read_sql_query(query, self.engine)
            logger.info(f"Loaded {len(self.seat_race_data)} seat race records")
            return self.seat_race_data
        except Exception as e:
            logger.error(f"Error loading seat race data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_gms_comparison(self):
        """
        Calculate each rower's performance relative to Gold Medal Standard times
        """
        if self.performance_data is None:
            self.load_performance_data()
            
        if self.performance_data.empty:
            logger.warning("No performance data available for GMS comparison")
            return pd.DataFrame()
            
        # Create a copy of performance data for analysis
        analysis_df = self.performance_data.copy()
        
        # Add GMS time based on boat identifier
        analysis_df['gms_time'] = analysis_df['boat_identifier'].map(GMS_TIMES)
        
        # Calculate percentage off GMS
        analysis_df['pct_off_gms'] = (
            (analysis_df['time'] - analysis_df['gms_time']) / analysis_df['gms_time'] * 100
        )
        
        # Drop rows where GMS time is not available
        analysis_df = analysis_df.dropna(subset=['gms_time', 'pct_off_gms'])
        
        if analysis_df.empty:
            logger.warning("No matching boat configurations found for GMS comparison")
            return pd.DataFrame()
            
        # Calculate average percentage off GMS for each rower
        rower_gms = analysis_df.groupby(['rower_id', 'rower_name']).agg({
            'pct_off_gms': ['mean', 'std', 'count'],
            'time': 'count',
            'event_id': 'nunique'
        }).reset_index()
        
        # Flatten multi-level columns
        rower_gms.columns = ['rower_id', 'rower_name', 'avg_pct_off_gms', 
                            'std_pct_off_gms', 'gms_count', 'total_pieces', 'event_count']
        
        # Add rank based on GMS performance
        rower_gms['rank'] = rower_gms['avg_pct_off_gms'].rank(method='min')
        
        # Sort by rank
        rower_gms = rower_gms.sort_values('rank')
        
        self.gms_comparison_data = rower_gms
        logger.info(f"Calculated GMS comparison for {len(rower_gms)} rowers")
        
        return rower_gms
    
    def visualize_gms_rankings(self, top_n=None, min_pieces=3, show_plot=True, save_path=None):
        """
        Visualize the rankings of rowers based on average percentage off GMS.
        
        Parameters:
        - top_n: Limit visualization to top N rowers (optional)
        - min_pieces: Minimum number of pieces required for inclusion
        - show_plot: Whether to display the plot
        - save_path: Path to save the visualization (optional)
        """
        if self.gms_comparison_data is None:
            self.calculate_gms_comparison()
            
        if self.gms_comparison_data.empty:
            logger.warning("No GMS comparison data available for visualization")
            return
            
        # Filter based on minimum pieces
        plot_data = self.gms_comparison_data[self.gms_comparison_data['gms_count'] >= min_pieces].copy()
        
        if plot_data.empty:
            logger.warning(f"No rowers with at least {min_pieces} pieces for GMS comparison")
            return
            
        # Limit to top N if specified
        if top_n is not None and len(plot_data) > top_n:
            plot_data = plot_data.head(top_n)
            
        # Create plot
        plt.figure(figsize=(12, max(8, len(plot_data) * 0.4)))
        
        # Create bar plot
        ax = sns.barplot(
            x='avg_pct_off_gms',
            y='rower_name',
            data=plot_data,
            palette='viridis',
            order=plot_data.sort_values('avg_pct_off_gms')['rower_name']
        )
        
        # Add error bars
        for i, row in plot_data.iterrows():
            idx = plot_data.sort_values('avg_pct_off_gms')['rower_name'].tolist().index(row['rower_name'])
            plt.errorbar(
                x=row['avg_pct_off_gms'],
                y=idx,
                xerr=row['std_pct_off_gms'],
                fmt='none',
                ecolor='gray',
                capsize=3
            )
        
        # Customize plot
        plt.xlabel('Average Percentage Off GMS (%)')
        plt.ylabel('Rower Name')
        plt.title('Rower Rankings Based on Average Percentage Off GMS')
        
        # Add count of pieces to labels
        ax2 = ax.twinx()
        ax2.set_yticks(range(len(plot_data)))
        ax2.set_yticklabels([f"({row['gms_count']} pieces)" for _, row in 
                             plot_data.sort_values('avg_pct_off_gms').iterrows()])
        ax2.set_ylabel('Piece Count')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved GMS rankings visualization to {save_path}")
            
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_rower_progress(self, rower_name=None, rower_id=None, days_back=90, 
                              show_plot=True, save_path=None):
        """
        Analyze a rower's progress over time
        
        Parameters:
        - rower_name: Name of the rower to analyze
        - rower_id: ID of the rower to analyze (alternative to name)
        - days_back: Number of days to look back
        - show_plot: Whether to display the plot
        - save_path: Path to save the visualization (optional)
        """
        if self.performance_data is None:
            self.load_performance_data()
            
        if self.performance_data.empty:
            logger.warning("No performance data available for rower progress analysis")
            return
            
        # Get rower ID if name provided
        if rower_id is None and rower_name is not None:
            rower_matches = self.performance_data[
                self.performance_data['rower_name'].str.lower() == rower_name.lower()
            ]
            if rower_matches.empty:
                logger.warning(f"No rower found with name '{rower_name}'")
                return
            rower_id = rower_matches['rower_id'].iloc[0]
            rower_name = rower_matches['rower_name'].iloc[0]  # Get exact case
            
        # Ensure we have a rower ID
        if rower_id is None:
            logger.error("Either rower_name or rower_id must be provided")
            return
            
        # Filter data for this rower
        rower_data = self.performance_data[self.performance_data['rower_id'] == rower_id].copy()
        
        if rower_data.empty:
            logger.warning(f"No performance data found for rower ID {rower_id}")
            return
            
        # Get exact rower name if not already known
        if rower_name is None:
            rower_name = rower_data['rower_name'].iloc[0]
            
        # Calculate GMS comparisons
        rower_data['gms_time'] = rower_data['boat_identifier'].map(GMS_TIMES)
        rower_data['pct_off_gms'] = (
            (rower_data['time'] - rower_data['gms_time']) / rower_data['gms_time'] * 100
        )
        
        # Drop rows without GMS times
        rower_data = rower_data.dropna(subset=['gms_time'])
        
        if rower_data.empty:
            logger.warning(f"No GMS comparison data available for rower {rower_name}")
            return
            
        # Sort by date
        rower_data = rower_data.sort_values('event_date')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot GMS comparison over time
        sns.lineplot(
            x='event_date',
            y='pct_off_gms',
            data=rower_data,
            marker='o',
            label='% Off GMS'
        )
        
        # Add trend line
        z = np.polyfit(
            pd.to_datetime(rower_data['event_date']).map(datetime.toordinal),
            rower_data['pct_off_gms'],
            1
        )
        p = np.poly1d(z)
        x_ord = pd.to_datetime(rower_data['event_date']).map(datetime.toordinal)
        plt.plot(
            pd.to_datetime(rower_data['event_date']),
            p(x_ord),
            "r--",
            label=f"Trend: {z[0]:.2f} per day"
        )
        
        # Customize plot
        plt.xlabel('Date')
        plt.ylabel('Percentage Off GMS (%)')
        plt.title(f'Performance Trend for {rower_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # X-axis date formatting
        plt.gcf().autofmt_xdate()
        
        # Add annotations for boat classes
        for i, row in rower_data.iterrows():
            plt.annotate(
                row['boat_identifier'],
                (pd.to_datetime(row['event_date']), row['pct_off_gms']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )
            
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved rower progress visualization to {save_path}")
            
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        # Return statistical summary
        summary = {
            'rower_name': rower_name,
            'rower_id': rower_id,
            'num_pieces': len(rower_data),
            'avg_pct_off_gms': rower_data['pct_off_gms'].mean(),
            'trend_per_day': z[0],
            'trend_per_month': z[0] * 30,
            'first_date': rower_data['event_date'].min(),
            'last_date': rower_data['event_date'].max(),
            'boat_classes': rower_data['boat_identifier'].unique().tolist()
        }
        
        return summary
    
    def analyze_seat_races(self, rower_name=None, show_plot=True, save_path=None):
        """
        Analyze seat race results, optionally filtered for a specific rower
        
        Parameters:
        - rower_name: Name of the rower to analyze (optional)
        - show_plot: Whether to display the plot
        - save_path: Path to save the visualization (optional)
        """
        if self.seat_race_data is None:
            self.load_seat_race_data()
            
        if self.seat_race_data.empty:
            logger.warning("No seat race data available for analysis")
            return
            
        # Filter for specific rower if requested
        if rower_name is not None:
            rower_races = self.seat_race_data[
                (self.seat_race_data['rower1_name'].str.lower() == rower_name.lower()) |
                (self.seat_race_data['rower2_name'].str.lower() == rower_name.lower())
            ].copy()
            
            if rower_races.empty:
                logger.warning(f"No seat races found for rower '{rower_name}'")
                return
                
            # Calculate result from perspective of the specified rower
            rower_races['rower_won'] = rower_races.apply(
                lambda row: (row['rower1_name'].lower() == rower_name.lower() and 
                            row['winner_name'].lower() == rower_name.lower()) or
                           (row['rower2_name'].lower() == rower_name.lower() and 
                            row['winner_name'].lower() == rower_name.lower()),
                axis=1
            )
            
            rower_races['time_diff_for_rower'] = rower_races.apply(
                lambda row: row['time_difference'] if row['rower_won'] else -row['time_difference'],
                axis=1
            )
            
            rower_races['opponent'] = rower_races.apply(
                lambda row: row['rower2_name'] if row['rower1_name'].lower() == rower_name.lower() 
                else row['rower1_name'],
                axis=1
            )
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot seat race results
            ax = sns.barplot(
                x='time_diff_for_rower',
                y='opponent',
                data=rower_races,
                palette=['green' if won else 'red' for won in rower_races['rower_won']],
                orient='h'
            )
            
            # Add event dates
            for i, row in rower_races.iterrows():
                idx = rower_races['opponent'].tolist().index(row['opponent'])
                plt.text(
                    x=0,
                    y=idx,
                    s=f"{row['event_date']}",
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold'
                )
                
            # Customize plot
            plt.xlabel('Time Difference (seconds)')
            plt.ylabel('Opponent')
            plt.title(f'Seat Race Results for {rower_name.title()}')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(True, axis='x', alpha=0.3)
            
            # Add win/loss count
            wins = rower_races['rower_won'].sum()
            losses = len(rower_races) - wins
            plt.figtext(
                0.02, 0.02,
                f"Wins: {wins}, Losses: {losses}",
                ha='left',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved seat race visualization to {save_path}")
                
            # Show if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            # Return summary statistics
            summary = {
                'rower_name': rower_name,
                'seat_races': len(rower_races),
                'wins': wins,
                'losses': losses,
                'win_percentage': wins / len(rower_races) * 100 if len(rower_races) > 0 else 0,
                'avg_margin': rower_races['time_diff_for_rower'].mean(),
                'opponents': rower_races['opponent'].tolist(),
                'dates': rower_races['event_date'].tolist()
            }
            
            return summary
        else:
            # Analyze overall seat race statistics
            
            # Create a DataFrame of all rowers involved in seat races
            all_rowers = pd.DataFrame({
                'rower_name': pd.concat([
                    self.seat_race_data['rower1_name'],
                    self.seat_race_data['rower2_name']
                ]).unique()
            })
            
            # Calculate wins and losses
            all_rowers['wins'] = all_rowers['rower_name'].apply(
                lambda name: sum(self.seat_race_data['winner_name'] == name)
            )
            
            all_rowers['races'] = all_rowers['rower_name'].apply(
                lambda name: sum(
                    (self.seat_race_data['rower1_name'] == name) | 
                    (self.seat_race_data['rower2_name'] == name)
                )
            )
            
            all_rowers['losses'] = all_rowers['races'] - all_rowers['wins']
            all_rowers['win_percentage'] = all_rowers['wins'] / all_rowers['races'] * 100
            
            # Sort by win percentage
            all_rowers = all_rowers.sort_values('win_percentage', ascending=False)
            
            # Create plot
            plt.figure(figsize=(12, max(8, len(all_rowers) * 0.4)))
            
            # Plot win percentages
            sns.barplot(
                x='win_percentage',
                y='rower_name',
                data=all_rowers,
                palette='viridis',
                order=all_rowers['rower_name']
            )
            
            # Customize plot
            plt.xlabel('Win Percentage (%)')
            plt.ylabel('Rower Name')
            plt.title('Seat Race Win Percentages')
            plt.grid(True, axis='x', alpha=0.3)
            
            # Add win-loss record
            ax2 = plt.gca().twinx()
            ax2.set_yticks(range(len(all_rowers)))
            ax2.set_yticklabels([
                f"({row['wins']}-{row['losses']})" 
                for _, row in all_rowers.iterrows()
            ])
            ax2.set_ylabel('Win-Loss Record')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved seat race statistics visualization to {save_path}")
                
            # Show if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return all_rowers

if __name__ == "__main__":
    analyzer = RowingAnalysis()
    
    analyzer.load_performance_data()
    analyzer.load_seat_race_data()
    
    # GMS Rankings
    gms_data = analyzer.calculate_gms_comparison()
    if not gms_data.empty:
        analyzer.visualize_gms_rankings(min_pieces=1, show_plot=False, save_path='analysis_output/gms_rankings.png')
        
    # Individual rower progress (example)
    rower_name = "Reilly" 
    analyzer.analyze_rower_progress(rower_name=rower_name, show_plot=False, save_path=f'analysis_output/{rower_name}_progress.png')
    
    # Seat race analysis
    analyzer.analyze_seat_races(show_plot=False, save_path='analysis_output/seat_races.png')
    analyzer.analyze_seat_races(rower_name=rower_name)