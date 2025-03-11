import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ErgDataProcessor:
    def __init__(self, db_engine=None):
        """Initialize the erg data processor with database connection"""
        if db_engine is None:
            password = os.getenv('PASSWORD')
            self.engine = create_engine(f'postgresql://postgres:{password}@localhost/rowing-analytics')
        else:
            self.engine = db_engine
            
        self.erg_data = None
        self.performance_data = None
        self.combined_data = None
        
    def load_erg_data(self, directory_path='./erg_data'):
        """Load all erg data files from the specified directory"""
        all_data = []
        
        try:
            for filename in os.listdir(directory_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(directory_path, filename)
                    logger.info(f"Processing erg file: {filename}")
                    
                    # Try to determine test type from filename
                    test_type = 'unknown'
                    if '2k' in filename.lower():
                        test_type = '2k'
                    elif '6k' in filename.lower():
                        test_type = '6k'
                    elif '30min' in filename.lower() or '30m' in filename.lower():
                        test_type = '30min'
                        
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Basic validation and cleanup
                    if 'name' in df.columns:
                        name_col = 'name'
                    elif 'rower' in df.columns:
                        name_col = 'rower'
                    elif 'rower_name' in df.columns:
                        name_col = 'rower_name'
                    else:
                        # Try to find a name column
                        for col in df.columns:
                            if 'name' in col.lower():
                                name_col = col
                                break
                        else:
                            logger.warning(f"Could not find name column in {filename}")
                            continue
                    
                    # Standardize column names
                    df = df.rename(columns={name_col: 'rower_name'})
                    
                    # Try to find date column
                    date_col = None
                    for col in df.columns:
                        if 'date' in col.lower():
                            date_col = col
                            break
                            
                    # Parse date if available
                    if date_col is not None:
                        try:
                            df['test_date'] = pd.to_datetime(df[date_col])
                        except:
                            logger.warning(f"Could not parse dates in {filename}")
                            # Try to extract date from filename
                            date_match = re.search(r'(\d{1,2}[-_/]\d{1,2}[-_/]\d{2,4})', filename)
                            if date_match:
                                try:
                                    test_date = pd.to_datetime(date_match.group(1))
                                    df['test_date'] = test_date
                                except:
                                    # Use file modification date as fallback
                                    mod_time = os.path.getmtime(file_path)
                                    df['test_date'] = pd.to_datetime(mod_time, unit='s')
                            else:
                                # Use file modification date as fallback
                                mod_time = os.path.getmtime(file_path)
                                df['test_date'] = pd.to_datetime(mod_time, unit='s')
                    else:
                        # Use file modification date
                        mod_time = os.path.getmtime(file_path)
                        df['test_date'] = pd.to_datetime(mod_time, unit='s')
                    
                    # Add test type column
                    df['test_type'] = test_type
                    
                    # Standardize time columns
                    time_col = None
                    for col in df.columns:
                        if 'time' in col.lower() or 'split' in col.lower():
                            time_col = col
                            break
                            
                    if time_col is not None:
                        df = df.rename(columns={time_col: 'erg_time'})
                        
                        # Convert time strings to seconds if not already
                        if df['erg_time'].dtype == object:
                            df['erg_time_seconds'] = df['erg_time'].apply(self._time_to_seconds)
                        else:
                            df['erg_time_seconds'] = df['erg_time']
                    
                    # Add source filename
                    df['source_file'] = filename
                    
                    all_data.append(df)
                    
            if all_data:
                self.erg_data = pd.concat(all_data, ignore_index=True)
                logger.info(f"Loaded {len(self.erg_data)} erg records")
                
                # Standardize rower names
                self._standardize_rower_names()
                
                return self.erg_data
            else:
                logger.warning("No erg data files found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading erg data: {e}")
            return pd.DataFrame()
            
    def _time_to_seconds(self, time_str):
        """Convert time string format (MM:SS.s) to seconds"""
        try:
            if isinstance(time_str, (int, float)):
                return float(time_str)
                
            if pd.isna(time_str) or time_str == '':
                return np.nan
                
            parts = str(time_str).strip().split(':')
            
            if len(parts) == 1:  # Already in seconds or single number
                return float(parts[0])
            elif len(parts) == 2:  # MM:SS.s
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:  # HH:MM:SS.s
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            else:
                return np.nan
        except:
            return np.nan
            
    def _standardize_rower_names(self):
        """Standardize rower names to match database"""
        if self.erg_data is None:
            return
            
        # Get rower names from database
        try:
            query = "SELECT rower_id, name FROM rower;"
            db_rowers = pd.read_sql(query, self.engine)
            
            # Create lookup dictionary for matching
            rower_lookup = {}
            for _, rower in db_rowers.iterrows():
                full_name = rower['name'].lower()
                last_name = rower['name'].split()[-1].lower() if ' ' in rower['name'] else rower['name'].lower()
                rower_lookup[full_name] = rower['rower_id']
                rower_lookup[last_name] = rower['rower_id']  # Also match by last name
                
            # Apply matching to standardize names
            def match_rower_name(name):
                if pd.isna(name):
                    return name
                    
                clean_name = str(name).strip().lower()
                
                if clean_name in rower_lookup:
                    return db_rowers.loc[db_rowers['rower_id'] == rower_lookup[clean_name], 
                                        'name'].values[0]
                
                # Try matching on just the last name part
                if ' ' in clean_name:
                    last = clean_name.split(' ')[-1]
                    if last in rower_lookup:
                        return db_rowers.loc[db_rowers['rower_id'] == rower_lookup[last], 
                                            'last_name'].values[0]
                
                return name  # Keep original if no match
                
            self.erg_data['rower_name'] = self.erg_data['rower_name'].apply(match_rower_name)
            
        except Exception as e:
            logger.error(f"Error standardizing rower names: {e}")

    def load_on_water_data(self):
        """Load on-water performance data from database"""
        try:
            query = """
            SELECT 
                rwr.rower_id,
                rwr.name AS rower_name,
                evt.event_date AS event_date,
                p.piece_number,
                b.name AS boat_name,
                b.boat_class,
                r.time,
                r.split,
                p.distance
            FROM 
                result r
                JOIN lineup l ON r.boat_id = l.boat_id AND r.piece_id = l.piece_id
                JOIN piece p ON r.piece_id = p.piece_id
                JOIN event evt ON p.event_id = evt.event_id
                JOIN boat b ON r.boat_id = b.boat_id
                JOIN rower rwr ON l.rower_id = rwr.rower_id
            ORDER BY
                evt.event_date DESC,
                p.piece_number;
            """
            
            self.performance_data = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(self.performance_data)} on-water performance records")
            
            return self.performance_data
            
        except Exception as e:
            logger.error(f"Error loading on-water data: {e}")
            return pd.DataFrame()

    def combine_data_sources(self, erg_days_lookback=180):
        """Combine erg data with on-water performance data"""
        if self.erg_data is None:
            self.load_erg_data()
            
        if self.performance_data is None:
            self.load_on_water_data()

        if self.erg_data is None or self.performance_data is None:
            logger.error("Failed to load one or both data sources")
            return pd.DataFrame()
            
        if self.erg_data.empty or self.performance_data.empty:
            logger.error("Cannot combine data: one or both sources are empty")
            return pd.DataFrame()
            
        try:
            # Filter erg data to recent results only
            if 'test_date' in self.erg_data.columns:
                cutoff_date = datetime.now() - timedelta(days=erg_days_lookback)
                recent_erg = self.erg_data[self.erg_data['test_date'] >= cutoff_date].copy()
            else:
                recent_erg = self.erg_data.copy()
                
            # Get most recent erg score for each rower by test type
            if not recent_erg.empty and 'test_date' in recent_erg.columns:
                recent_erg = recent_erg.sort_values('test_date', ascending=False)
                recent_erg = recent_erg.drop_duplicates(subset=['rower_name', 'test_type'])
            
            # Pivot to create columns for different test types
            if not recent_erg.empty:
                erg_pivot = recent_erg.pivot_table(
                    index='rower_name',
                    columns='test_type',
                    values='erg_time_seconds',
                    aggfunc='first'
                ).reset_index()
                
                # Rename columns for clarity
                erg_columns = {col: f'erg_{col}' for col in erg_pivot.columns if col != 'rower_name'}
                erg_pivot = erg_pivot.rename(columns=erg_columns)
                
                # Merge with performance data
                self.combined_data = pd.merge(
                    self.performance_data,
                    erg_pivot,
                    on='rower_name',
                    how='left'
                )
                
                logger.info(f"Combined data contains {len(self.combined_data)} records")
                return self.combined_data
                
            else:
                logger.warning("No recent erg data available for combining")
                return self.performance_data
                
        except Exception as e:
            logger.error(f"Error combining data sources: {e}")
            return pd.DataFrame()

    def calculate_erg_to_water_ratio(self, test_type='2k', min_pieces=3, recency_weight=True):
        """
        Calculate the ratio between erg performance and on-water performance
        
        Parameters:
        - test_type: The erg test type to use ('2k', '6k', etc.)
        - min_pieces: Minimum on-water pieces required
        - recency_weight: Whether to weight recent performances more heavily
        """
        if self.combined_data is None:
            self.combine_data_sources()
            
        if self.combined_data.empty:
            logger.error("No combined data available for analysis")
            return pd.DataFrame()
            
        # Column name for the specific test type
        erg_col = f'erg_{test_type}'
        
        if erg_col not in self.combined_data.columns:
            logger.error(f"No data for erg test type: {test_type}")
            return pd.DataFrame()
            
        # Filter for rows with both water and erg data
        filtered_data = self.combined_data.dropna(subset=[erg_col, 'time']).copy()
        
        if filtered_data.empty:
            logger.error("No overlapping data between erg and water performance")
            return pd.DataFrame()
            
        # Group by rower and calculate metrics
        rower_data = filtered_data.groupby('rower_name').apply(
            lambda group: self._calculate_rower_metrics(group, erg_col, recency_weight)
        ).reset_index()
        
        # Filter for minimum number of pieces
        rower_data = rower_data[rower_data['piece_count'] >= min_pieces]
        
        # Calculate ratios
        rower_data['erg_to_water_ratio'] = rower_data['erg_time'] / rower_data['weighted_water_time']
        
        # Sort by ratio (higher is better - maximizes water speed relative to erg)
        rower_data = rower_data.sort_values('erg_to_water_ratio', ascending=False)
        
        return rower_data
        
    def _calculate_rower_metrics(self, group, erg_col, recency_weight):
        """Calculate metrics for an individual rower"""
        # Get most recent erg score
        erg_time = group[erg_col].iloc[0]  # All rows for a rower have same erg score
        
        # Calculate weighted average of water times
        if recency_weight:
            # Convert dates to numeric for weighting
            group = group.copy()
            group['days_ago'] = (datetime.now() - pd.to_datetime(group['event_date'])).dt.days
            
            # Calculate weights (more recent = higher weight)
            max_days = group['days_ago'].max()
            group['weight'] = 1 - (group['days_ago'] / (max_days * 2))  # Linear decay
            
            # Calculate weighted average
            weighted_time = (group['time'] * group['weight']).sum() / group['weight'].sum()
        else:
            # Simple average
            weighted_time = group['time'].mean()
            
        return pd.Series({
            'erg_time': erg_time,
            'weighted_water_time': weighted_time,
            'piece_count': len(group),
            'latest_event_date': group['event_date'].max(),
            'boat_classes': ', '.join(group['boat_class'].unique())
        })

    def predict_lineup_performance(self, rowers, boat_class='8+', show_plot=False, save_path=None):
        """
        Predict performance of a theoretical lineup based on individual metrics
        
        Parameters:
        - rowers: List of rower names
        - boat_class: Boat class for the lineup
        - show_plot: Whether to display the plot
        - save_path: Path to save the visualization
        """
        if self.combined_data is None:
            self.combine_data_sources()
            
        if self.combined_data.empty:
            logger.error("No combined data available for prediction")
            return None
            
        # Get strength metrics for each rower
        rower_metrics = {}
        for rower in rowers:
            # Try to find rower in the data
            rower_data = self.combined_data[self.combined_data['rower_name'] == rower]
            
            if rower_data.empty:
                logger.warning(f"No data found for rower: {rower}")
                continue
                
            # Get 2k erg time if available
            if 'erg_2k' in rower_data.columns and not rower_data['erg_2k'].isna().all():
                erg_time = rower_data['erg_2k'].iloc[0]
            else:
                # Try other erg columns
                erg_cols = [col for col in rower_data.columns if col.startswith('erg_')]
                for col in erg_cols:
                    if not rower_data[col].isna().all():
                        erg_time = rower_data[col].iloc[0]
                        break
                else:
                    logger.warning(f"No erg data found for {rower}")
                    erg_time = None
            
            # Filter for specified boat class if possible
            boat_specific = rower_data[rower_data['boat_class'] == boat_class]
            
            if not boat_specific.empty:
                # Use boat-class specific data
                avg_water_time = boat_specific['time'].mean()
                boat_count = len(boat_specific)
            else:
                # Use all data
                avg_water_time = rower_data['time'].mean()
                boat_count = 0
                
            rower_metrics[rower] = {
                'erg_time': erg_time,
                'avg_water_time': avg_water_time,
                'boat_specific_count': boat_count,
                'total_pieces': len(rower_data)
            }
        
        # Get historical boat performances for comparison
        historical_query = f"""
        SELECT 
            r.time,
            count(distinct l.rower_id) as crew_size
        FROM 
            result r
            JOIN boat b ON r.boat_id = b.boat_id
            JOIN lineup l ON r.result_id = l.result_id
        WHERE 
            b.boat_class = '{boat_class}'
        GROUP BY 
            r.time
        HAVING
            count(distinct l.rower_id) >= 4
        ORDER BY
            r.time;
        """
        
        try:
            historical = pd.read_sql(historical_query, self.engine)
            
            if not historical.empty:
                best_time = historical['time'].min()
                avg_time = historical['time'].mean()
                logger.info(f"Historical {boat_class} - Best: {best_time}s, Avg: {avg_time}s")
            else:
                best_time = None
                avg_time = None
                
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            historical = pd.DataFrame()
            best_time = None
            avg_time = None
        
        # Predict performance using combination of erg and water metrics
        valid_rowers = [r for r in rower_metrics if rower_metrics[r]['erg_time'] is not None]
        
        if len(valid_rowers) < 2:
            logger.error("Not enough rowers with erg data for prediction")
            return None
            
        # Calculate predicted time based on erg scores and water performance history
        erg_sum = sum(rower_metrics[r]['erg_time'] for r in valid_rowers)
        erg_avg = erg_sum / len(valid_rowers)
        
        # Calculate average water performance factor
        water_factors = []
        for r in valid_rowers:
            if rower_metrics[r]['avg_water_time'] is not None:
                water_factors.append(rower_metrics[r]['avg_water_time'] / rower_metrics[r]['erg_time'])
                
        if water_factors:
            avg_water_factor = sum(water_factors) / len(water_factors)
        else:
            # Fallback to typical 8+ conversion
            avg_water_factor = 3.5 if boat_class == '8+' else 3.8
            
        # Predict time
        predicted_time = erg_avg * avg_water_factor
        
        # Create comparison visualization
        if show_plot or save_path:
            plt.figure(figsize=(12, 6))
            
            # Rower erg scores
            plt.subplot(1, 2, 1)
            rowers_to_plot = [r for r in valid_rowers if rower_metrics[r]['erg_time'] is not None]
            times = [rower_metrics[r]['erg_time'] for r in rowers_to_plot]
            
            bars = plt.bar(rowers_to_plot, times)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('2k Erg Time (seconds)')
            plt.title('Individual Erg Scores')
            
            # Add time labels on bars
            for bar, time_val in zip(bars, times):
                mins = int(time_val // 60)
                secs = time_val % 60
                label = f"{mins}:{secs:04.1f}"
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() - 10,
                    label,
                    ha='center',
                    va='bottom',
                    color='white',
                    fontweight='bold'
                )
                
            # Predicted vs historical
            plt.subplot(1, 2, 2)
            
            comparison_data = []
            if predicted_time is not None:
                comparison_data.append(('Predicted', predicted_time))
                
            if best_time is not None:
                comparison_data.append(('Best Historical', best_time))
                
            if avg_time is not None:
                comparison_data.append(('Avg Historical', avg_time))
                
            if comparison_data:
                labels, values = zip(*comparison_data)
                bars = plt.bar(labels, values, color=['blue', 'green', 'orange'][:len(comparison_data)])
                
                # Add time labels
                for bar, time_val in zip(bars, values):
                    mins = int(time_val // 60)
                    secs = time_val % 60
                    label = f"{mins}:{secs:04.1f}"
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() - 10,
                        label,
                        ha='center', 
                        va='bottom',
                        color='white',
                        fontweight='bold'
                    )
                    
                plt.ylabel('Time (seconds)')
                plt.title(f'Predicted {boat_class} Performance')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved lineup prediction to {save_path}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        # Return prediction details
        return {
            'lineup': rowers,
            'valid_rowers': valid_rowers,
            'boat_class': boat_class,
            'predicted_time': predicted_time,
            'best_historical': best_time,
            'avg_historical': avg_time,
            'rower_metrics': rower_metrics
        }

    def visualize_erg_to_water_correlation(self, test_type='2k', show_plot=True, save_path=None):
        """
        Visualize the correlation between erg scores and on-water performance
        """
        if self.combined_data is None:
            self.combine_data_sources()
            
        if self.combined_data.empty:
            logger.error("No combined data available for visualization")
            return
            
        erg_col = f'erg_{test_type}'
        
        if erg_col not in self.combined_data.columns:
            logger.error(f"No data for erg test type: {test_type}")
            return
            
        # Filter for rows with both water and erg data
        filtered = self.combined_data.dropna(subset=[erg_col, 'time']).copy()
        
        if filtered.empty:
            logger.error("No overlapping data between erg and water performance")
            return
            
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Color by boat class if available
        if 'boat_class' in filtered.columns:
            boat_classes = filtered['boat_class'].unique()
            colors = sns.color_palette("husl", len(boat_classes))
            
            for i, boat_class in enumerate(boat_classes):
                subset = filtered[filtered['boat_class'] == boat_class]
                plt.scatter(
                    subset[erg_col], 
                    subset['time'],
                    label=boat_class,
                    color=colors[i],
                    alpha=0.7
                )
                
            plt.legend(title='Boat Class')
        else:
            plt.scatter(filtered[erg_col], filtered['time'], alpha=0.7)
            
        # Add labels for points
        for _, row in filtered.iterrows():
            plt.annotate(
                row['rower_name'],
                (row[erg_col], row['time']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8
            )
            
        # Calculate and plot regression line
        x = filtered[erg_col]
        y = filtered['time']
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color='red', linestyle='--')
        
        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]
        
        plt.title(f'Correlation between {test_type} Erg Time and On-Water Performance (r={corr:.2f})')
        plt.xlabel(f'{test_type} Erg Time (seconds)')
        plt.ylabel('On-Water Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved correlation visualization to {save_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return corr

    def analyze_boat_class_specialists(self, min_pieces=3, show_plot=True, save_path=None):
        """
        Identify rowers who perform better in specific boat classes
        """
        if self.performance_data is None:
            self.load_on_water_data()
            
        if self.performance_data.empty:
            logger.error("No performance data available for analysis")
            return
            
        # Ensure we have boat class data
        if 'boat_class' not in self.performance_data.columns:
            logger.error("No boat class information in performance data")
            return
            
        # Calculate GMS comparison (assuming appropriate boat_identifier column)
        if 'boat_identifier' not in self.performance_data.columns:
            # Create boat identifier from boat_class
            self.performance_data['boat_identifier'] = self.performance_data['boat_class']
            
        # Import GMS times from rowing_analysis.py
        try:
            from rowing_analysis import GMS_TIMES
        except ImportError:
            # Define basic GMS times if not available
            GMS_TIMES = {
                '8+': 350.0,  # Estimated 2k time
                '4+': 380.0,
                '4x': 370.0,
                '2-': 410.0,
                '2x': 400.0,
                '1x': 430.0,
            }
            
        # Calculate percentage off GMS
        self.performance_data['gms_time'] = self.performance_data['boat_identifier'].map(GMS_TIMES)
        self.performance_data['pct_off_gms'] = (
            (self.performance_data['time'] - self.performance_data['gms_time']) / 
            self.performance_data['gms_time'] * 100
        )
        
        # Drop rows without GMS times
        perf_data = self.performance_data.dropna(subset=['gms_time', 'pct_off_gms']).copy()
        
        if perf_data.empty:
            logger.error("No data with valid GMS comparisons")
            return
            
        # Group by rower and boat class
        rower_boat_perf = perf_data.groupby(['rower_name', 'boat_class']).agg({
            'pct_off_gms': ['mean', 'std', 'count'],
            'event_date': 'max'
        }).reset_index()
        
        # Flatten multi-index columns
        rower_boat_perf.columns = [
            '_'.join(col).strip('_') for col in rower_boat_perf.columns.values
        ]
        
        # Filter for minimum number of pieces
        rower_boat_perf = rower_boat_perf[rower_boat_perf['pct_off_gms_count'] >= min_pieces]
        
        # Calculate overall average for each rower
        rower_overall = perf_data.groupby('rower_name').agg({
            'pct_off_gms': 'mean'
        }).reset_index()
        rower_overall = rower_overall.rename(columns={'pct_off_gms': 'overall_pct_off_gms'})
        
        # Merge boat class performance with overall
        rower_boat_perf = rower_boat_perf.merge(rower_overall, on='rower_name', how='left')
        
        # Calculate difference from overall average
        rower_boat_perf['diff_from_overall'] = (
            rower_boat_perf['pct_off_gms_mean'] - rower_boat_perf['overall_pct_off_gms']
        )
        
        # Sort by difference (positive = better in this boat class)
        rower_boat_perf = rower_boat_perf.sort_values('diff_from_overall', ascending=False)
        
        # Create visualization
        if show_plot or save_path:
            plt.figure(figsize=(14, 10))
            
            # Plot performance by boat class
            plt.subplot(2, 1, 1)
            for boat_class in rower_boat_perf['boat_class'].unique():
                subset = rower_boat_perf[rower_boat_perf['boat_class'] == boat_class]
                plt.scatter(
                    subset['overall_pct_off_gms'],
                    subset['pct_off_gms_mean'],
                    label=boat_class,
                    alpha=0.7
                )
                
            plt.plot(
                [rower_boat_perf['overall_pct_off_gms'].min(), rower_boat_perf['overall_pct_off_gms'].max()],
                [rower_boat_perf['overall_pct_off_gms'].min(), rower_boat_perf['overall_pct_off_gms'].max()],
                '--',
                color='gray'
            )
            
            plt.xlabel('Overall % Off GMS')
            plt.ylabel('Boat Class % Off GMS')
            plt.legend(title='Boat Class')
            plt.title('Performance by Boat Class')
            plt.grid(True, alpha=0.3)
            
            # Plot difference from overall
            plt.subplot(2, 1, 2)
            bars = plt.bar(
                rower_boat_perf['rower_name'] + ' (' + rower_boat_perf['boat_class'] + ')',
                rower_boat_perf['diff_from_overall'],
                color=plt.cm.RdYlGn((rower_boat_perf['diff_from_overall'] - rower_boat_perf['diff_from_overall'].min()) / 
                                    (rower_boat_perf['diff_from_overall'].max() - rower_boat_perf['diff_from_overall'].min()))
            )
            
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Difference from Overall % Off GMS')
            plt.title('Specialist Performance by Boat Class')
            plt.grid(True, alpha=0.3)
            
            # Add piece count as text
            for bar, count in zip(bars, rower_boat_perf['pct_off_gms_count']):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() - 5 if bar.get_height() > 0 else bar.get_height() + 5,
                    f"{int(count)} pieces",
                    ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                    color='white'
                )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        return rower_boat_perf

    def analyze_erg_water_efficiency_ratio(self, min_water_pieces=3, test_type='2k', show_plot=True, save_path=None):
        """
        Analyzes how efficiently rowers convert erg power to boat speed
        
        Returns a DataFrame of rowers ranked by their efficiency ratio
        Higher ratio = better on-water performance relative to erg score
        """
        if self.combined_data is None:
            self.combine_data_sources()
        
        erg_col = f'erg_{test_type}'
        
        # Check if the requested column exists in the data
        if erg_col not in self.combined_data.columns:
            logger.error(f"No {test_type} erg data found in the combined dataset")
            logger.info(f"Available columns: {', '.join(self.combined_data.columns)}")
            
            # Try to find an alternative erg test type
            available_erg_cols = [col for col in self.combined_data.columns if col.startswith('erg_')]
            if available_erg_cols:
                alt_col = available_erg_cols[0]
                logger.info(f"Using alternative erg data: {alt_col}")
                erg_col = alt_col
            else:
                return pd.DataFrame()
        
        # Filter data where both erg and water times exist
        filtered_data = self.combined_data.dropna(subset=[erg_col, 'time']).copy()
        
        # Group by rower, calculate metrics with recency weighting
        results = []
        for rower_name, group in filtered_data.groupby('rower_name'):
            if len(group) < min_water_pieces:
                continue
                
            # Calculate recency-weighted water performance
            group = group.sort_values('event_date', ascending=False)
            weights = np.linspace(1.0, 0.5, len(group))[:len(group)]  # Linear decay weights
            weighted_water_time = np.average(group['time'], weights=weights)
            
            # Get erg time (same for all entries of this rower)
            erg_time = group[erg_col].iloc[0]
            
            # Calculate efficiency (higher = better water performance relative to erg)
            efficiency_ratio = erg_time / weighted_water_time
            
            results.append({
                'rower_name': rower_name,
                'erg_time': erg_time,
                'weighted_water_time': weighted_water_time,
                'efficiency_ratio': efficiency_ratio,
                'power_to_weight': group['watts_per_lb'].mean() if 'watts_per_lb' in group.columns else None,
                'weight': group['weight'].mean() if 'weight' in group.columns else None,
                'piece_count': len(group)
            })
        
        # Convert to DataFrame and sort
        df_results = pd.DataFrame(results).sort_values('efficiency_ratio', ascending=False)
        
        # Create visualization
        if show_plot or save_path:
            plt.figure(figsize=(14, 8))
            
            # Create scatter plot
            plt.scatter(
                df_results['erg_time'],
                df_results['weighted_water_time'],
                s=df_results['piece_count']*10,  # Size by piece count
                alpha=0.7
            )
            
            # Add labels to each point
            for _, row in df_results.iterrows():
                plt.annotate(
                    row['rower_name'],
                    (row['erg_time'], row['weighted_water_time']),
                    xytext=(5, 0),
                    textcoords='offset points'
                )
            
            # Add diagonal lines showing different efficiency ratios
            x_min, x_max = df_results['erg_time'].min()*0.95, df_results['erg_time'].max()*1.05
            for ratio in [0.25, 0.3, 0.35, 0.4, 0.45]:
                plt.plot(
                    [x_min, x_max],
                    [x_min/ratio, x_max/ratio],
                    '--',
                    color='gray',
                    alpha=0.5,
                    label=f'Ratio: {ratio:.2f}'
                )
            
            plt.xlabel(f'{test_type} Erg Time (seconds)')
            plt.ylabel('Weighted Water Time (seconds)')
            plt.title('Erg-to-Water Performance Efficiency')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        return df_results


    def predict_optimal_lineup(self, available_rowers, boat_class='8+', seats=8, 
                            balance_weight=True, balance_sides=True, show_plot=True, save_path=None):
        """
        Predicts the optimal lineup from available rowers by analyzing:
        - Recent erg performance (weighted by recency)
        - On-water efficiency (how well erg translates to boat speed)
        - Boat class specialization
        - Weight distribution
        - Side preference (P/S balance)
        
        Parameters:
        - available_rowers: List of rower names
        - boat_class: Target boat class ('8+', '4+', '4x', '2-', etc)
        - seats: Number of rowing seats (excluding cox)
        - balance_weight: Whether to balance weight distribution
        - balance_sides: Whether to account for port/starboard preferences
        - show_plot: Whether to show visualization
        - save_path: Path to save visualization
        
        Returns: Dict with lineup details and predicted performance
        """
        if self.erg_data is None:
            self.load_erg_data()
            
        if self.combined_data is None:
            self.combine_data_sources()
            
        # Build comprehensive metrics for each rower
        rower_metrics = {}
        for rower in available_rowers:
            # Get erg data
            rower_erg = self.erg_data[self.erg_data['rower_name'] == rower].copy()
            
            # Get on-water data
            rower_water = self.combined_data[self.combined_data['rower_name'] == rower].copy()
            
            # Skip if no data
            if rower_erg.empty and rower_water.empty:
                logger.warning(f"No data found for rower: {rower}")
                continue
                
            # --- Erg metrics ---
            erg_scores = {}
            if not rower_erg.empty:
                # Sort by recency
                rower_erg = rower_erg.sort_values('test_date', ascending=False)
                
                # Get latest erg scores by test type
                for test_type in ['2k', '6k', '5k', '30min']:
                    test_data = rower_erg[rower_erg['test_type'] == test_type]
                    if not test_data.empty:
                        latest_score = test_data['erg_time_seconds'].iloc[0]
                        latest_date = test_data['test_date'].iloc[0]
                        erg_scores[test_type] = {
                            'time': latest_score,
                            'date': latest_date
                        }
                
                # Get side preference (P/S)
                if 'side' in rower_erg.columns:
                    sides = rower_erg['side'].value_counts()
                    preferred_side = sides.idxmax() if not sides.empty else None
                else:
                    preferred_side = None
                    
                # Get weight
                if 'weight' in rower_erg.columns:
                    weight = rower_erg['weight'].mean()
                else:
                    weight = None
                    
                # Get power-to-weight
                if 'watts_per_lb' in rower_erg.columns:
                    power_to_weight = rower_erg['watts_per_lb'].mean()
                else:
                    power_to_weight = None
            else:
                preferred_side = None
                weight = None
                power_to_weight = None
                
            # --- On-water metrics ---
            water_metrics = {}
            if not rower_water.empty:
                # Calculate average performance by boat class
                for boat_type, group in rower_water.groupby('boat_class'):
                    if len(group) < 2:
                        continue
                        
                    # Calculate recency-weighted performance
                    group = group.sort_values('event_date', ascending=False)
                    group['days_ago'] = (datetime.now() - pd.to_datetime(group['event_date'])).dt.days
                    group['weight'] = np.exp(-0.023 * group['days_ago'])  # Half-weight every 30 days
                    
                    weighted_time = np.average(group['time'], weights=group['weight'])
                    
                    water_metrics[boat_type] = {
                        'count': len(group),
                        'avg_time': group['time'].mean(),
                        'weighted_time': weighted_time,
                        'latest_date': group['event_date'].max()
                    }
            
            # --- Calculate efficiency ratios ---
            efficiency_ratios = {}
            for boat_type, metrics in water_metrics.items():
                if '2k' in erg_scores:
                    erg_time = erg_scores['2k']['time']
                    efficiency_ratios[boat_type] = erg_time / metrics['weighted_time']
                elif '6k' in erg_scores:
                    # Approximate 2k from 6k
                    erg_time = erg_scores['6k']['time'] * 0.85
                    efficiency_ratios[boat_type] = erg_time / metrics['weighted_time']
                    
            # Store all metrics
            rower_metrics[rower] = {
                'erg': erg_scores,
                'water': water_metrics,
                'efficiency': efficiency_ratios,
                'side': preferred_side,
                'weight': weight,
                'power_to_weight': power_to_weight,
            }
        
        # --- Score each rower for this boat class ---
        rower_scores = {}
        for rower, metrics in rower_metrics.items():
            # Base score from 2k erg (normalize so higher is better)
            if '2k' in metrics['erg']:
                # Convert erg time to "speed" (higher = better)
                erg_score = 500 / (metrics['erg']['2k']['time'] / 4)
            elif '6k' in metrics['erg']:
                # Approximate 2k from 6k
                erg_time = metrics['erg']['6k']['time'] * 0.85
                erg_score = 500 / (erg_time / 4)
            else:
                # No suitable erg score
                continue
                
            # Weight by recency (more recent = stronger weight)
            if '2k' in metrics['erg']:
                days_old = (datetime.now() - pd.to_datetime(metrics['erg']['2k']['date'])).days
            elif '6k' in metrics['erg']:
                days_old = (datetime.now() - pd.to_datetime(metrics['erg']['6k']['date'])).days
            else:
                days_old = 180
                
            recency_factor = np.exp(-0.0115 * days_old)  # Half-weight every 60 days
            erg_score *= recency_factor
            
            # Boost for boat class experience
            experience_boost = 1.0
            if boat_class in metrics['water']:
                pieces = metrics['water'][boat_class]['count']
                # Logarithmic increase (diminishing returns)
                experience_boost = 1 + min(0.2 * np.log(1 + pieces/5), 0.3)  # Up to 30% bonus
                
            # Boost for boat class efficiency
            efficiency_boost = 1.0
            if boat_class in metrics['efficiency']:
                # Higher efficiency = larger boost
                # Normalize around typical value of ~0.35
                efficiency = metrics['efficiency'][boat_class]
                efficiency_boost = (efficiency / 0.35) ** 1.5
            
            # Calculate final score
            final_score = erg_score * experience_boost * efficiency_boost
            
            # Add the scores to our dictionary
            rower_scores[rower] = {
                'total': final_score,
                'erg_score': erg_score,
                'experience': experience_boost,
                'efficiency': efficiency_boost
            }
        
        # --- Select optimal lineup ---
        # First, select based on score
        selected_rowers = sorted(rower_scores.keys(), key=lambda x: rower_scores[x]['total'], reverse=True)[:min(len(rower_scores), seats)]
        
        # Optimize for side preference if requested
        if balance_sides:
            port_rowers = []
            starboard_rowers = []
            unknown_side_rowers = []
            
            # Categorize selected rowers by side preference
            for rower in selected_rowers:
                side = rower_metrics[rower]['side']
                if side == 'P':
                    port_rowers.append(rower)
                elif side == 'S':
                    starboard_rowers.append(rower)
                else:
                    unknown_side_rowers.append(rower)
            
            # Check if we need to balance sides
            port_count = len(port_rowers)
            starboard_count = len(starboard_rowers)
            target_count = seats // 2
            
            # If we have imbalanced sides, try to fix
            if port_count != starboard_count:
                # First try to assign unknown side rowers
                for rower in unknown_side_rowers[:]:
                    if port_count < starboard_count:
                        port_rowers.append(rower)
                        port_count += 1
                        unknown_side_rowers.remove(rower)
                    elif starboard_count < port_count:
                        starboard_rowers.append(rower)
                        starboard_count += 1
                        unknown_side_rowers.remove(rower)
                    else:
                        break
                        
                # If still imbalanced, need to swap out rowers
                if port_count != starboard_count:
                    # Get all available rowers not already selected
                    remaining = [r for r in rower_scores.keys() if r not in selected_rowers]
                    remaining = sorted(remaining, key=lambda x: rower_scores[x]['total'], reverse=True)
                    
                    # Try to find rowers with needed side preference
                    needed_side = 'P' if port_count < starboard_count else 'S'
                    
                    for rower in remaining:
                        if rower_metrics[rower]['side'] == needed_side:
                            # Find weakest opposite side to replace
                            replace_side = 'S' if needed_side == 'P' else 'P'
                            replace_list = starboard_rowers if needed_side == 'P' else port_rowers
                            
                            if replace_list:
                                # Find weakest to replace
                                replace_rower = min(replace_list, key=lambda x: rower_scores[x]['total'])
                                
                                # Only replace if new rower is close enough in performance
                                if rower_scores[rower]['total'] > rower_scores[replace_rower]['total'] * 0.85:
                                    replace_list.remove(replace_rower)
                                    if needed_side == 'P':
                                        port_rowers.append(rower)
                                    else:
                                        starboard_rowers.append(rower)
                                    selected_rowers.remove(replace_rower)
                                    selected_rowers.append(rower)
                                    
                                    # Update counts
                                    if needed_side == 'P':
                                        port_count += 1
                                        starboard_count -= 1
                                    else:
                                        starboard_count += 1
                                        port_count -= 1
                                        
                                    # Check if balanced
                                    if port_count == starboard_count:
                                        break
            
            # Add any remaining unknown side rowers
            selected_rowers = port_rowers + starboard_rowers + unknown_side_rowers
            
            # Sort within sides by score
            port_rowers = sorted(port_rowers, key=lambda x: rower_scores[x]['total'], reverse=True)
            starboard_rowers = sorted(starboard_rowers, key=lambda x: rower_scores[x]['total'], reverse=True)
        
        # Optimize weight balance if requested
        if balance_weight:
            # Only proceed if we have weights
            if all(rower_metrics[r]['weight'] is not None for r in selected_rowers):
                # If we balanced sides, balance weights between sides
                if balance_sides and len(port_rowers) > 0 and len(starboard_rowers) > 0:
                    # Calculate side weights
                    port_weight = sum(rower_metrics[r]['weight'] for r in port_rowers)
                    starboard_weight = sum(rower_metrics[r]['weight'] for r in starboard_rowers)
                    
                    # Try to swap rowers to balance weight
                    attempts = 0
                    max_attempts = 10
                    
                    while abs(port_weight - starboard_weight) > 20 and attempts < max_attempts:
                        attempts += 1
                        
                        if port_weight > starboard_weight:
                            # Find heaviest port and lightest starboard
                            heaviest_port = max(port_rowers, key=lambda r: rower_metrics[r]['weight'])
                            lightest_starboard = min(starboard_rowers, key=lambda r: rower_metrics[r]['weight'])
                            
                            # Calculate weight change if swapped
                            weight_change = rower_metrics[heaviest_port]['weight'] - rower_metrics[lightest_starboard]['weight']
                            
                            # Only swap if it improves balance
                            if weight_change > 0:
                                # Swap
                                port_rowers.remove(heaviest_port)
                                starboard_rowers.remove(lightest_starboard)
                                port_rowers.append(lightest_starboard)
                                starboard_rowers.append(heaviest_port)
                                
                                # Update weights
                                port_weight = sum(rower_metrics[r]['weight'] for r in port_rowers)
                                starboard_weight = sum(rower_metrics[r]['weight'] for r in starboard_rowers)
                            else:
                                break
                        else:
                            # Find heaviest starboard and lightest port
                            heaviest_starboard = max(starboard_rowers, key=lambda r: rower_metrics[r]['weight'])
                            lightest_port = min(port_rowers, key=lambda r: rower_metrics[r]['weight'])
                            
                            # Calculate weight change if swapped
                            weight_change = rower_metrics[heaviest_starboard]['weight'] - rower_metrics[lightest_port]['weight']
                            
                            # Only swap if it improves balance
                            if weight_change > 0:
                                # Swap
                                starboard_rowers.remove(heaviest_starboard)
                                port_rowers.remove(lightest_port)
                                starboard_rowers.append(lightest_port)
                                port_rowers.append(heaviest_starboard)
                                
                                # Update weights
                                port_weight = sum(rower_metrics[r]['weight'] for r in port_rowers)
                                starboard_weight = sum(rower_metrics[r]['weight'] for r in starboard_rowers)
                            else:
                                break
                    
                    # Re-sort within sides by score after weight balancing
                    port_rowers = sorted(port_rowers, key=lambda x: rower_scores[x]['total'], reverse=True)
                    starboard_rowers = sorted(starboard_rowers, key=lambda x: rower_scores[x]['total'], reverse=True)
                    
                    # Update selected rowers list to reflect changes
                    selected_rowers = []
                    for i in range(max(len(port_rowers), len(starboard_rowers))):
                        if i < len(port_rowers):
                            selected_rowers.append(port_rowers[i])
                        if i < len(starboard_rowers):
                            selected_rowers.append(starboard_rowers[i])

        # --- Predict performance ---
        # Get 2k scores for selected rowers
        erg_times = []
        for rower in selected_rowers:
            if '2k' in rower_metrics[rower]['erg']:
                erg_times.append(rower_metrics[rower]['erg']['2k']['time'])
            elif '6k' in rower_metrics[rower]['erg']:
                erg_times.append(rower_metrics[rower]['erg']['6k']['time'] * 0.85)
        
        if not erg_times:
            logger.warning("No erg times available for prediction")
            erg_avg = None
            predicted_time = None
        else:
            # Calculate average erg score
            erg_avg = sum(erg_times) / len(erg_times)
            
            # Calculate average efficiency for this boat class
            efficiencies = [
                rower_metrics[r]['efficiency'][boat_class] 
                for r in selected_rowers
                if boat_class in rower_metrics[r]['efficiency']
            ]
            
            if efficiencies:
                avg_efficiency = sum(efficiencies) / len(efficiencies)
            else:
                avg_efficiency = 0.35  # Default efficiency if no data
            
            # Predict time
            predicted_time = erg_avg / avg_efficiency
        
        # Create visualization
        if show_plot or save_path:
            plt.figure(figsize=(14, 8))
            
            # Plot erg scores
            plt.subplot(2, 1, 1)
            erg_scores = [rower_metrics[r]['erg']['2k']['time'] if '2k' in rower_metrics[r]['erg'] else rower_metrics[r]['erg']['6k']['time'] * 0.85 for r in selected_rowers]
            plt.bar(selected_rowers, erg_scores, color='skyblue')
            plt.ylabel('Erg Time (seconds)')
            plt.title('Selected Rower Erg Scores')
            plt.xticks(rotation=45, ha='right')
            
            # Plot predicted performance
            plt.subplot(2, 1, 2)
            plt.bar(['Predicted Time'], [predicted_time], color='green')
            plt.ylabel('Time (seconds)')
            plt.title('Predicted Boat Performance')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        # Return results
        return {
            'lineup': selected_rowers,
            'predicted_time': predicted_time,
            'rower_metrics': {r: rower_metrics[r] for r in selected_rowers},
            'rower_scores': {r: rower_scores[r] for r in selected_rowers}
        }

    def _calculate_compatibility_scores(self, rowers):
        """Calculate how well rowers perform together based on past lineups"""
        query = """
        SELECT 
            l1.rower_id as rower1_id, 
            r1.name as rower1_name,
            l2.rower_id as rower2_id,
            r2.name as rower2_name,
            r.time,
            p.piece_number,
            e.date as event_date
        FROM 
            lineup l1
            JOIN lineup l2 ON l1.result_id = l2.result_id AND l1.rower_id < l2.rower_id
            JOIN result r ON l1.result_id = r.result_id
            JOIN piece p ON r.piece_id = p.piece_id
            JOIN event e ON p.event_id = e.event_id
            JOIN rower r1 ON l1.rower_id = r1.rower_id
            JOIN rower r2 ON l2.rower_id = r2.rower_id
        WHERE 
            r1.name IN :rowers AND r2.name IN :rowers
        ORDER BY 
            e.date DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.engine, params={'rowers': tuple(rowers)})
            pairs = {}
            
            for _, row in df.iterrows():
                pair = (row['rower1_name'], row['rower2_name'])
                if pair not in pairs:
                    pairs[pair] = []
                pairs[pair].append({
                    'time': row['time'],
                    'date': row['event_date'],
                    'piece': row['piece_number']
                })
            
            # Calculate compatibility scores with recency weighting
            compatibility = {}
            for pair, results in pairs.items():
                # Sort by date (newest first)
                sorted_results = sorted(results, key=lambda x: x['date'], reverse=True)
                
                # Calculate weighted average time (newer = more weight)
                weights = np.linspace(1.0, 0.5, len(sorted_results))[:len(sorted_results)]
                times = [r['time'] for r in sorted_results]
                avg_time = np.average(times, weights=weights)
                
                # Store compatibility (lower time = better compatibility)
                compatibility[pair] = {
                    'avg_time': avg_time,
                    'count': len(results)
                }
            
            return compatibility
        
        except Exception as e:
            logger.error(f"Error calculating compatibility scores: {e}")
            return {}

    def analyze_seat_race_with_erg_context(self, show_plot=True, save_path=None):
        """
        Analyzes seat race outcomes in context of erg performance
        """
        # Load seat race data from database
        try:
            query = """
            SELECT 
                sr.seat_race_id,
                sr.time_difference,
                e.event_date AS race_date,
                r1.rower_id AS rower1_id,
                r1.name AS rower1_name,
                r2.rower_id AS rower2_id, 
                r2.name AS rower2_name,
                winner.name AS winner_name,
                sr.notes
            FROM 
                seat_race sr
                JOIN event e ON sr.event_id = e.event_id
                JOIN rower r1 ON sr.rower_id_1 = r1.rower_id
                JOIN rower r2 ON sr.rower_id_2 = r2.rower_id
                JOIN rower winner ON sr.winner_id = winner.rower_id
            ORDER BY
                e.event_date DESC;
            """
            seat_races = pd.read_sql(query, self.engine)
            
            if seat_races.empty:
                logger.warning("No seat race data found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading seat race data: {e}")
            return pd.DataFrame()
        
        # Load erg data if needed
        if self.erg_data is None:
            self.load_erg_data()
        
        # Get most recent erg scores for each rower
        all_rowers = set(seat_races['rower1_name'].tolist() + seat_races['rower2_name'].tolist())
        erg_lookup = self._get_recent_erg_scores(all_rowers)
        
        # Add erg differential to seat race data
        results = []
        for _, race in seat_races.iterrows():
            rower1, rower2 = race['rower1_name'], race['rower2_name']
            winner = race['winner_name']
            
            result_row = {
                'seat_race_id': race['seat_race_id'],
                'race_date': race['race_date'],
                'rower1_name': rower1,
                'rower2_name': rower2,
                'winner_name': winner,
                'time_difference': race['time_difference'],
                'notes': race['notes']
            }
            
            # Add erg data with date context
            if rower1 in erg_lookup and rower2 in erg_lookup:
                erg1 = self._get_closest_erg_score(erg_lookup[rower1], race['race_date'])
                erg2 = self._get_closest_erg_score(erg_lookup[rower2], race['race_date'])
                
                if erg1 and erg2:
                    result_row['rower1_erg'] = erg1['score']
                    result_row['rower1_erg_date'] = erg1['date']
                    result_row['rower2_erg'] = erg2['score']
                    result_row['rower2_erg_date'] = erg2['date']
                    result_row['erg_difference'] = erg2['score'] - erg1['score']  # Positive = rower1 faster
                    
                    # Add physiological context
                    if 'weight' in erg1 and 'weight' in erg2:
                        result_row['rower1_weight'] = erg1['weight']
                        result_row['rower2_weight'] = erg2['weight']
                        result_row['weight_difference'] = erg2['weight'] - erg1['weight']
                    
                    if 'power_to_weight' in erg1 and 'power_to_weight' in erg2:
                        result_row['rower1_p2w'] = erg1['power_to_weight']
                        result_row['rower2_p2w'] = erg2['power_to_weight']
                        result_row['p2w_difference'] = erg1['power_to_weight'] - erg2['power_to_weight']
                    
                    # Calculate predicted margin from erg difference (adjusted by piece distance)
                    # Extract distance from notes if available
                    distance = 2000  # Default to 2k
                    if race['notes'] and 'meters' in str(race['notes']).lower():
                        match = re.search(r'(\d+)\s*meters', str(race['notes']).lower())
                        if match:
                            distance = int(match.group(1))
                    
                    predicted_margin = (erg2['score'] - erg1['score']) * 3.0 * (distance/2000)
                    result_row['predicted_margin'] = predicted_margin
                    
                    # Calculate actual margin with sign
                    actual_margin = race['time_difference']
                    if winner == rower1:
                        actual_signed_margin = actual_margin
                    else:
                        actual_signed_margin = -actual_margin
                        
                    result_row['actual_margin'] = actual_signed_margin
                    result_row['margin_difference'] = actual_signed_margin - predicted_margin
                    result_row['over_performance_ratio'] = actual_signed_margin / predicted_margin if predicted_margin != 0 else 0
            
            results.append(result_row)
        
        df_results = pd.DataFrame(results)
        
        # Calculate performance metrics by rower with recency weighting
        df_performance = self._calculate_seat_race_metrics(df_results)
        
        # Create enhanced visualizations
        if show_plot or save_path:
            self._visualize_seat_race_metrics(df_results, df_performance, show_plot, save_path)
        
        return {
            'seat_races': df_results,
            'rower_performance': df_performance
        }

    def _get_closest_erg_score(self, erg_history, race_date):
        """Get the erg score closest to the race date"""
        # Sort by date difference
        erg_history = sorted(erg_history, 
                            key=lambda x: abs((x['date'] - pd.to_datetime(race_date)).days))
        
        if erg_history:
            return erg_history[0]
        return None

    def analyze_water_erg_correlation_matrix(self, test_type='2k', 
                                            days_lookback=180, min_pieces=3,
                                            show_plot=True, save_path=None):
        """
        Create a correlation matrix between erg scores and on-water performance
        across different boat classes and piece types
        """
        if self.combined_data is None:
            self.combine_data_sources(erg_days_lookback=days_lookback)
            
        if self.combined_data.empty:
            logger.error("No combined data available")
            return
            
        erg_col = f'erg_{test_type}'
        if erg_col not in self.combined_data.columns:
            logger.error(f"No {test_type} erg data available")
            return
            
        # Filter for rows with both erg and water data
        filtered_data = self.combined_data.dropna(subset=[erg_col, 'time']).copy()
        
        if filtered_data.empty:
            logger.error("No matching erg and water data")
            return
            
        # Add recency weighting column
        filtered_data['days_ago'] = (datetime.now() - pd.to_datetime(filtered_data['event_date'])).dt.days
        max_days = filtered_data['days_ago'].max()
        filtered_data['recency_weight'] = 1 - (filtered_data['days_ago'] / (max_days * 1.5))
        
        # Calculate rower-specific metrics
        rower_metrics = []
        
        for rower_name, group in filtered_data.groupby('rower_name'):
            if len(group) < min_pieces:
                continue
                
            # Get erg time and weight stats
            erg_time = group[erg_col].iloc[0]  # Same for all rows
            
            # Calculate boat class specific performance
            boat_classes = group['boat_class'].unique()
            for boat_class in boat_classes:
                boat_subset = group[group['boat_class'] == boat_class]
                
                if len(boat_subset) < 2:
                    continue
                    
                # Calculate weighted average time for this boat class
                weighted_time = np.average(boat_subset['time'], 
                                        weights=boat_subset['recency_weight'])
                
                # Efficiency ratio (how erg translates to water)
                efficiency = erg_time / weighted_time
                
                # Store metrics
                rower_metrics.append({
                    'rower_name': rower_name,
                    'boat_class': boat_class,
                    'erg_time': erg_time,
                    'water_time': weighted_time,
                    'efficiency_ratio': efficiency,
                    'piece_count': len(boat_subset),
                    'power_to_weight': boat_subset['watts_per_lb'].mean() if 'watts_per_lb' in boat_subset.columns else None,
                    'weight': boat_subset['weight'].mean() if 'weight' in boat_subset.columns else None
                })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(rower_metrics)
        
        if metrics_df.empty:
            logger.error("No metrics could be calculated")
            return
            
        # Create interactive heatmap visualization showing correlation structure
        if show_plot or save_path:
            plt.figure(figsize=(16, 12))
            
            # Create subplots
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
            
            # Scatter plot of erg vs water performance by boat class
            ax1 = plt.subplot(gs[0, 0])
            
            boat_classes = metrics_df['boat_class'].unique()
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
            
            for i, boat_class in enumerate(boat_classes):
                subset = metrics_df[metrics_df['boat_class'] == boat_class]
                ax1.scatter(
                    subset['erg_time'],
                    subset['water_time'],
                    label=boat_class,
                    marker=markers[i % len(markers)],
                    s=subset['piece_count']*20,
                    alpha=0.7
                )
                
                # Add rower names
                for _, row in subset.iterrows():
                    ax1.annotate(
                        row['rower_name'],
                        (row['erg_time'], row['water_time']),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=8
                    )
            
            # Add trend lines for each boat class
            for boat_class in boat_classes:
                subset = metrics_df[metrics_df['boat_class'] == boat_class]
                if len(subset) >= 3:  # Need at least 3 points for meaningful trend
                    x = subset['erg_time']
                    y = subset['water_time']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax1.plot(x, p(x), '--', alpha=0.7)
            
            ax1.set_xlabel(f'{test_type} Erg Time (seconds)')
            ax1.set_ylabel('Water Time (seconds, weighted by recency)')
            ax1.set_title('Erg to Water Performance by Boat Class')
            ax1.legend(title='Boat Class')
            ax1.grid(True, alpha=0.3)
            
            # Top performers in each boat class
            ax2 = plt.subplot(gs[0, 1])
            
            # Get top 3 rowers by efficiency in each boat class
            top_performers = []
            for boat_class in boat_classes:
                subset = metrics_df[metrics_df['boat_class'] == boat_class]
                if not subset.empty:
                    top = subset.nlargest(min(3, len(subset)), 'efficiency_ratio')
                    for _, row in top.iterrows():
                        top_performers.append({
                            'rower': row['rower_name'],
                            'boat_class': boat_class,
                            'efficiency': row['efficiency_ratio']
                        })
            
            # Create horizontal bar chart of top performers
            top_df = pd.DataFrame(top_performers)
            if not top_df.empty:
                # Sort by efficiency
                top_df = top_df.sort_values('efficiency', ascending=True)
                
                # Create labels with boat class
                labels = [f"{row['rower']} ({row['boat_class']})" for _, row in top_df.iterrows()]
                
                bars = ax2.barh(labels, top_df['efficiency'], color='skyblue')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax2.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                            f'{width:.3f}', ha='left', va='center')
            
            ax2.set_xlabel('Efficiency Ratio')
            ax2.set_title('Top Performers by Boat Class')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Create weight vs efficiency plot
            ax3 = plt.subplot(gs[1, 0])
            
            # Only include rows with weight data
            weight_data = metrics_df.dropna(subset=['weight'])
            if not weight_data.empty:
                scatter = ax3.scatter(
                    weight_data['weight'],
                    weight_data['efficiency_ratio'],
                    c=weight_data['erg_time'],
                    s=weight_data['piece_count']*15,
                    alpha=0.7,
                    cmap='viridis_r'  # Reversed viridis (faster ergs are darker)
                )
                
                # Add rower names
                for _, row in weight_data.iterrows():
                    ax3.annotate(
                        row['rower_name'],
                        (row['weight'], row['efficiency_ratio']),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=8
                    )
                
                # Add trend line
                x = weight_data['weight']
                y = weight_data['efficiency_ratio']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax3.plot(x, p(x), '--', color='red', alpha=0.7)
                
                # Add color bar for erg time
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label(f'{test_type} Erg Time')
            
            ax3.set_xlabel('Weight (lbs)')
            ax3.set_ylabel('Efficiency Ratio')
            ax3.set_title('Weight vs Water Efficiency')
            ax3.grid(True, alpha=0.3)
            
            # Specialty boat class analysis
            ax4 = plt.subplot(gs[1, 1])
            
            # Calculate each rower's best boat class
            rower_specialty = {}
            for rower_name, group in metrics_df.groupby('rower_name'):
                if len(group) < 2:  # Need at least 2 boat classes to compare
                    continue
                    
                # Find boat class with best efficiency
                best_idx = group['efficiency_ratio'].idxmax()
                best_boat = group.loc[best_idx]
                
                # Calculate how much better than average
                avg_efficiency = group['efficiency_ratio'].mean()
                specialty_score = (best_boat['efficiency_ratio'] / avg_efficiency - 1) * 100
                
                rower_specialty[rower_name] = {
                    'best_boat': best_boat['boat_class'],
                    'best_efficiency': best_boat['efficiency_ratio'],
                    'specialty_score': specialty_score
                }
            
            specialty_df = pd.DataFrame.from_dict(rower_specialty, orient='index').reset_index()
            specialty_df = specialty_df.rename(columns={'index': 'rower_name'})
            
            if not specialty_df.empty:
                specialty_df = specialty_df.sort_values('specialty_score', ascending=False)
                
                bars = ax4.barh(
                    specialty_df['rower_name'],
                    specialty_df['specialty_score'],
                    color=[plt.cm.tab10(i % 10) for i in range(len(specialty_df))]
                )
                
                # Add boat class labels
                for i, (_, row) in enumerate(specialty_df.iterrows()):
                    ax4.text(
                        row['specialty_score'] * 1.01,
                        i,
                        row['best_boat'],
                        va='center',
                        fontsize=8
                    )
            
            ax4.set_xlabel('Specialty Score (%)')
            ax4.set_title('Boat Class Specialization')
            ax4.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved water-erg correlation matrix to {save_path}")
                
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        return metrics_df

    def analyze_erg_water_correlation_matrix(self, test_types=['2k', '6k'], min_pieces=3, show_plot=True, save_path=None):
        """
        Creates a comprehensive correlation matrix between erg performance and on-water results
        accounting for recency, boat class specialization, and weight factors
        """
        if self.combined_data is None:
            self.combine_data_sources()
            
        # Create multi-factor correlation dataset
        results = []
        for rower_name, group in self.combined_data.groupby('rower_name'):
            if len(group) < min_pieces:
                continue
                
            # Get most recent performance data with exponential time decay
            group = group.sort_values('event_date', ascending=False)
            
            # Calculate days since most recent piece
            most_recent = pd.to_datetime(group['event_date'].max())
            group['days_ago'] = (most_recent - pd.to_datetime(group['event_date'])).dt.days
            
            # Apply exponential decay weight (half-weight every 30 days)
            group['recency_weight'] = np.exp(-0.023 * group['days_ago'])  # ln(2)/30 ≈ 0.023
            
            # Get weighted average water performance by boat class
            boat_classes = {}
            for boat_class, boat_group in group.groupby('boat_class'):
                if len(boat_group) < 2:
                    continue
                    
                weighted_time = np.average(boat_group['time'], weights=boat_group['recency_weight'])
                boat_classes[boat_class] = {
                    'count': len(boat_group),
                    'weighted_time': weighted_time
                }
            
            # Get erg metrics
            erg_metrics = {}
            for test_type in test_types:
                erg_col = f'erg_{test_type}'
                if erg_col in group.columns and not group[erg_col].isna().all():
                    erg_metrics[test_type] = group[erg_col].iloc[0]
            
            # Skip if no erg data
            if not erg_metrics:
                continue
                
            # Calculate weight-normalized metrics if available
            if 'weight' in group.columns and not group['weight'].isna().all():
                weight = group['weight'].iloc[0]
                
                # Weight-normalized erg (accounts for heavyweight/lightweight differences)
                for test_type, time in erg_metrics.items():
                    erg_metrics[f"{test_type}_weight_norm"] = time * (170/weight)**0.33  # Allometric scaling
            
            # Calculate efficiency ratios for each boat class
            efficiency_ratios = {}
            for boat_class, metrics in boat_classes.items():
                if '2k' in erg_metrics:
                    efficiency_ratios[f"{boat_class}_eff"] = erg_metrics['2k'] / metrics['weighted_time']
                elif '6k' in erg_metrics:
                    # Approximate 2k from 6k
                    est_2k = erg_metrics['6k'] * 0.85
                    efficiency_ratios[f"{boat_class}_eff"] = est_2k / metrics['weighted_time']
                    
            # Store all metrics
            result = {
                'rower_name': rower_name,
                'piece_count': len(group),
                'most_recent': most_recent,
                'avg_water_time': group['time'].mean(),
                'weighted_water_time': np.average(group['time'], weights=group['recency_weight']),
                **erg_metrics,
                **{f"{k}_pieces": v['count'] for k, v in boat_classes.items()},
                **{f"{k}_time": v['weighted_time'] for k, v in boat_classes.items()},
                **efficiency_ratios
            }
            
            results.append(result)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Create correlation heatmap
        if show_plot or save_path:
            plt.figure(figsize=(14, 12))
            
            # Select numeric columns for correlation
            numeric_cols = df_results.select_dtypes(include=['float64', 'int64']).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != 'most_recent']
            
            # Calculate correlation matrix
            corr = df_results[numeric_cols].corr()
            
            # Plot heatmap
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            plt.title('Correlation Matrix of Erg and On-Water Performance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        return df_results

    def analyze_rower_progression(self, rower_name, test_type='2k', days=365, show_plot=True, save_path=None):
        """
        Analyzes a rower's progression over time, comparing erg and on-water performance trends
        """
        # Load data if not already loaded
        if self.erg_data is None:
            self.load_erg_data()
        
        if self.performance_data is None:
            self.load_on_water_data()
            
        # Filter for selected rower
        rower_erg = self.erg_data[self.erg_data['rower_name'] == rower_name].copy()
        rower_water = self.performance_data[self.performance_data['rower_name'] == rower_name].copy()
        
        if rower_erg.empty and rower_water.empty:
            logger.warning(f"No data found for rower: {rower_name}")
            return None
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter for date range
        if 'test_date' in rower_erg.columns:
            rower_erg = rower_erg[(rower_erg['test_date'] >= start_date) & 
                                (rower_erg['test_date'] <= end_date)]
                                
        rower_water = rower_water[(rower_water['event_date'] >= start_date) & 
                                (rower_water['event_date'] <= end_date)]
        
        # Create visualization
        if show_plot or save_path:
            plt.figure(figsize=(14, 10))
            
            # Plot 1: Erg progression
            plt.subplot(2, 1, 1)
            
            if not rower_erg.empty:
                # Filter for test type if specified
                if test_type != 'all':
                    test_erg = rower_erg[rower_erg['test_type'] == test_type]
                else:
                    test_erg = rower_erg
                    
                if not test_erg.empty and 'test_date' in test_erg.columns:
                    # Sort by date
                    test_erg = test_erg.sort_values('test_date')
                    
                    # Plot erg times
                    plt.plot(
                        test_erg['test_date'], 
                        test_erg['erg_time_seconds'], 
                        'o-', 
                        label=f"{test_type} Erg"
                    )
                    
                    # Add trendline
                    dates_numeric = pd.to_datetime(test_erg['test_date']).map(datetime.toordinal)
                    z = np.polyfit(dates_numeric, test_erg['erg_time_seconds'], 1)
                    p = np.poly1d(z)
                    plt.plot(
                        test_erg['test_date'], 
                        p(dates_numeric), 
                        'r--',
                        label=f"Trend: {z[0]*30:.2f} sec/month"
                    )
                    
                    plt.ylabel(f"{test_type} Time (seconds)")
                    plt.title(f"{rower_name}'s Erg Progression")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Add split times as text
                    for i, row in test_erg.iterrows():
                        mins = int(row['erg_time_seconds'] // 60)
                        secs = row['erg_time_seconds'] % 60
                        plt.text(
                            row['test_date'], 
                            row['erg_time_seconds'] + 2,
                            f"{mins}:{secs:05.2f}",
                            ha='center'
                        )
            
            # Plot 2: On-water progression with GMS comparison
            plt.subplot(2, 1, 2)
            
            if not rower_water.empty:
                # Calculate GMS percentages if possible
                if 'boat_identifier' in rower_water.columns:
                    from rowing_analysis import GMS_TIMES
                    
                    rower_water['gms_time'] = rower_water['boat_identifier'].map(GMS_TIMES)
                    rower_water['pct_off_gms'] = (
                        (rower_water['time'] - rower_water['gms_time']) / 
                        rower_water['gms_time'] * 100
                    )
                    
                    valid_gms = rower_water.dropna(subset=['gms_time', 'pct_off_gms'])
                    
                    if not valid_gms.empty:
                        # Group by date and boat class
                        for name, group in valid_gms.groupby('boat_class'):
                            group = group.sort_values('event_date')
                            plt.scatter(
                                group['event_date'], 
                                group['pct_off_gms'],
                                label=name,
                                s=50,
                                alpha=0.7
                            )
                        
                        # Add overall trendline
                        all_data = valid_gms.sort_values('event_date')
                        dates_numeric = pd.to_datetime(all_data['event_date']).map(datetime.toordinal)
                        z = np.polyfit(dates_numeric, all_data['pct_off_gms'], 1)
                        p = np.poly1d(z)
                        plt.plot(
                            all_data['event_date'], 
                            p(dates_numeric), 
                            'k--',
                            label=f"Trend: {z[0]*30:.2f}%/month"
                        )
                        
                        plt.ylabel('% Off Gold Medal Standard')
                        plt.title(f"{rower_name}'s On-Water Performance")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                else:
                    # Just plot raw times
                    for name, group in rower_water.groupby('boat_class'):
                        group = group.sort_values('event_date')
                        plt.plot(
                            group['event_date'], 
                            group['time'],
                            'o-', 
                            label=name
                        )
                    
                    plt.ylabel('Time (seconds)')
                    plt.title(f"{rower_name}'s On-Water Performance")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        # Return analysis results
        return {
            'rower': rower_name,
            'erg_data': rower_erg,
            'water_data': rower_water
        }

