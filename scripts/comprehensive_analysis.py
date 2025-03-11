import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from rowing_analysis import RowingAnalysis
from erg_data_processor import ErgDataProcessor

# Setup logging and output directory
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.makedirs('analysis_output', exist_ok=True)

class ComprehensiveAnalysis:
    def __init__(self):
        """Initialize analysis with both water and erg data processors"""
        self.water_analyzer = RowingAnalysis()
        self.erg_processor = ErgDataProcessor()
        
        # Load all data sources
        self.load_all_data()
        
    def load_all_data(self):
        """Load all data from both sources"""
        logger.info("Loading on-water performance data")
        self.water_analyzer.load_performance_data()
        self.water_analyzer.load_seat_race_data()
        
        logger.info("Loading erg data")
        self.erg_processor.load_erg_data()
        
        logger.info("Combining data sources")
        self.combined_data = self.erg_processor.combine_data_sources()
    
    def run_complete_analysis(self):
        """Run all analyses and generate visualizations"""
        logger.info("Beginning comprehensive analysis")
        
        # 1. Performance rankings with integrated metrics
        self.performance_rankings()
        
        # 2. Erg-to-water efficiency analysis
        self.erg_water_efficiency()
        
        # 3. Boat class specialization analysis
        self.boat_class_analysis()
        
        # 4. Lineup optimization and prediction
        self.lineup_optimization()
        
        # 5. Seat race analysis with erg context
        self.enhanced_seat_race_analysis()
        
        # 6. Correlation matrix of all metrics
        self.metrics_correlation()
        
        # 7. Rower development tracking
        self.development_tracking()
        
        logger.info("Analysis complete. Results saved to 'analysis_output' directory")
    
    def performance_rankings(self):
        """Create comprehensive performance rankings"""
        # Calculate GMS rankings
        gms_data = self.water_analyzer.calculate_gms_comparison()
        
        # Calculate erg-to-water efficiency
        eff_data = self.erg_processor.analyze_erg_water_efficiency_ratio(show_plot=False)
        
        # Merge the datasets
        if not gms_data.empty and not eff_data.empty:
            merged = pd.merge(
                gms_data,
                eff_data[['rower_name', 'efficiency_ratio', 'erg_time']],
                on='rower_name',
                how='outer'
            )
            
            # Calculate composite score (normalize each metric first)
            if not merged.empty and 'avg_pct_off_gms' in merged.columns:
                # Lower GMS % is better, invert for normalization
                merged['gms_norm'] = 1 - ((merged['avg_pct_off_gms'] - merged['avg_pct_off_gms'].min()) / 
                                         (merged['avg_pct_off_gms'].max() - merged['avg_pct_off_gms'].min()))
                
                # Higher efficiency is better
                if 'efficiency_ratio' in merged.columns:
                    valid_eff = merged.dropna(subset=['efficiency_ratio'])
                    if not valid_eff.empty:
                        merged['eff_norm'] = (merged['efficiency_ratio'] - valid_eff['efficiency_ratio'].min()) / \
                                            (valid_eff['efficiency_ratio'].max() - valid_eff['efficiency_ratio'].min())
                    
                        # Combined score (70% water performance, 30% erg efficiency)
                        merged['composite_score'] = merged['gms_norm']*0.7 + merged['eff_norm']*0.3
                        
                        # Sort by composite score
                        merged = merged.sort_values('composite_score', ascending=False)
            
            # Visualization
            plt.figure(figsize=(14, 10))
            
            # Create bar chart
            plt.barh(
                merged['rower_name'][-20:],  # Show top 20
                merged['composite_score'][-20:],
                color=plt.cm.viridis(merged['composite_score'][-20:])
            )
            
            plt.xlabel('Composite Performance Score')
            plt.ylabel('Rower')
            plt.title('Overall Rower Rankings (Water Performance + Erg Efficiency)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('analysis_output/comprehensive_rankings.png')
            plt.close()
            
            return merged
    
    def erg_water_efficiency(self):
        """Analyze how well erg scores translate to water performance"""
        self.erg_processor.visualize_erg_to_water_correlation(
            show_plot=False,
            save_path='analysis_output/erg_water_correlation.png'
        )
        
        
    
    def boat_class_analysis(self):
        """Analyze which rowers specialize in particular boat classes"""
        self.erg_processor.analyze_boat_class_specialists(
            show_plot=False,
            save_path='analysis_output/boat_class_specialists.png'
        )
    
    def lineup_optimization(self):
        """Predict optimal lineups based on all available data"""
        # Get all rower names from database
        connection = self.water_analyzer.engine.connect()
        rowers_df = pd.read_sql("SELECT name FROM rower ORDER BY name", connection)
        connection.close()
        
        all_rowers = rowers_df['name'].tolist()
        
        # Predict 8+ lineup
        self.erg_processor.predict_lineup_performance(  # Changed from predict_optimal_lineup
            rowers=all_rowers,
            boat_class='8+',
            show_plot=False,
            save_path='analysis_output/optimal_8plus_lineup.png'
        )
        
        # Predict 4+ lineup
        self.erg_processor.predict_lineup_performance(  # Changed from predict_optimal_lineup
            rowers=all_rowers,
            boat_class='4+',
            show_plot=False,
            save_path='analysis_output/optimal_4plus_lineup.png'
        )
    
    def enhanced_seat_race_analysis(self):
        """Analyze seat race results with erg performance context"""
        results = self.erg_processor.analyze_seat_race_with_erg_context(
            show_plot=False,
            save_path='analysis_output/erg_seat_race_analysis.png'
        )
        
        if results and 'seat_races' in results:
            seat_races = results['seat_races']
            
            # Create additional visualization of predicted vs actual margins
            if 'predicted_margin' in seat_races.columns and 'actual_margin' in seat_races.columns:
                valid_races = seat_races.dropna(subset=['predicted_margin', 'actual_margin'])
                
                if not valid_races.empty:
                    plt.figure(figsize=(12, 8))
                    
                    plt.scatter(
                        valid_races['predicted_margin'],
                        valid_races['actual_margin'],
                        alpha=0.7
                    )
                    
                    # Add identity line
                    min_val = min(valid_races['predicted_margin'].min(), valid_races['actual_margin'].min())
                    max_val = max(valid_races['predicted_margin'].max(), valid_races['actual_margin'].max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                    
                    # Add labels for each point
                    for _, row in valid_races.iterrows():
                        plt.annotate(
                            f"{row['rower1_name']} vs {row['rower2_name']}",
                            (row['predicted_margin'], row['actual_margin']),
                            xytext=(5, 0),
                            textcoords='offset points',
                            fontsize=8
                        )
                    
                    plt.xlabel('Predicted Margin from Erg (seconds)')
                    plt.ylabel('Actual Seat Race Margin (seconds)')
                    plt.title('Erg-Predicted vs Actual Seat Race Margins')
                    plt.grid(True, alpha=0.3)
                    
                    plt.savefig('analysis_output/predicted_vs_actual_margins.png')
                    plt.close()
    
    def metrics_correlation(self):
        """Create a correlation matrix of all available metrics"""
        self.erg_processor.analyze_erg_water_correlation_matrix(
            show_plot=False,
            save_path='analysis_output/metrics_correlation.png'
        )
    
    def development_tracking(self):
        """Track rower development over time, combining erg and water metrics"""
        # Select a few top rowers to analyze
        gms_data = self.water_analyzer.calculate_gms_comparison()
        
        if not gms_data.empty:
            top_rowers = gms_data.head(5)['rower_name'].tolist()
            
            # Analyze each top rower's progression
            for rower in top_rowers:
                self.erg_processor.analyze_rower_progression(
                    rower_name=rower,
                    show_plot=False,
                    save_path=f'analysis_output/{rower}_progression.png'
                )

    def analyze_weighted_performance_trend(self, day_weights=None, min_pieces=3, show_plot=True, save_path=None):
        """Analyze weighted performance with emphasis on recent results"""
        if day_weights is None:
            # Default decay function: more recent = higher weight
            day_weights = {
                30: 1.0,    # Last 30 days: 100% weight
                90: 0.75,   # 31-90 days: 75% weight
                180: 0.5,   # 91-180 days: 50% weight
                365: 0.25,  # 181-365 days: 25% weight
            }

        # Get water data with date-weighted importance
        water_data = self.water_analyzer.performance_data.copy()
        erg_data = self.erg_processor.get_recent_erg_tests(min_tests=1)

        # Calculate days since each performance
        today = pd.Timestamp.now().normalize()
        water_data['days_ago'] = (today - pd.to_datetime(water_data['event_date'])).dt.days

        # Apply weights based on recency
        water_data['weight'] = 0.1  # Minimum weight for old data
        for days, weight in sorted(day_weights.items()):
            water_data.loc[water_data['days_ago'] <= days, 'weight'] = weight

        # Calculate weighted performance metrics
        weighted_results = []

        for rower_name in water_data['rower_name'].unique():
            rower_water = water_data[water_data['rower_name'] == rower_name]

            if len(rower_water) < min_pieces:
                continue

            # Calculate weighted GMS
            weighted_gms = np.average(
                rower_water['pct_off_gms'], 
                weights=rower_water['weight']
            )

            # Get most recent erg scores
            rower_erg = None
            if erg_data is not None:
                rower_erg = erg_data[erg_data['rower_name'] == rower_name]

            result = {
                'rower_name': rower_name,
                'weighted_gms': weighted_gms,
                'raw_avg_gms': rower_water['pct_off_gms'].mean(),
                'piece_count': len(rower_water),
                'recent_weight': rower_water['weight'].mean()
            }

            # Add erg data if available
            if rower_erg is not None and not rower_erg.empty:
                result['latest_2k'] = rower_erg[rower_erg['test_type'] == '2k']['score'].min()
                result['p2w_ratio'] = rower_erg['watts_per_lb'].max() 
                result['erg_efficiency'] = (result['raw_avg_gms'] / result['latest_2k']) * 500

            weighted_results.append(result)

        # Create DataFrame and sort
        df_results = pd.DataFrame(weighted_results)
        df_results = df_results.sort_values('weighted_gms')

        # Create visualization
        if show_plot or save_path:
            plt.figure(figsize=(14, 10))

            # Create bar chart for weighted GMS
            bars = plt.barh(
                df_results['rower_name'], 
                df_results['weighted_gms'],
                color=plt.cm.viridis(
                    np.linspace(0, 1, len(df_results))
                ),
                alpha=0.8,
                label='Weighted % Off GMS (Recent Emphasis)'
            )

            # Add raw average GMS as markers
            plt.scatter(
                df_results['raw_avg_gms'],
                df_results['rower_name'],
                color='red',
                marker='o',
                label='Raw Average % Off GMS'
            )

            # Add erg efficiency if available
            if 'erg_efficiency' in df_results.columns:
                # Normalize for plotting alongside GMS
                norm_factor = df_results['weighted_gms'].mean() / df_results['erg_efficiency'].mean()
                eff_adjusted = df_results['erg_efficiency'] * norm_factor

                plt.scatter(
                    eff_adjusted,
                    df_results['rower_name'],
                    color='blue',
                    marker='s',
                    label='Erg-to-Water Efficiency (scaled)'
                )

            plt.xlabel('Percentage Off GMS (lower is better)')
            plt.ylabel('Rower')
            plt.title('Weighted Rower Performance with Recent Results Emphasized')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path)

            if show_plot:
                plt.show()
            else:
                plt.close()

        return df_results

def calculate_rower_efficiency_index(self, days_lookback=180, min_pieces=4):
    """
    Create a comprehensive efficiency index that accounts for:
    1. Erg-to-water translation efficiency
    2. Weight-adjusted performance
    3. Most recent performances (heavier weighting)
    4. Consistency across different boat classes
    """
    # Get required data
    water_data = self.water_analyzer.performance_data
    erg_data = self.erg_processor.erg_data
    
    # Filter by recency
    cutoff_date = datetime.now() - pd.Timedelta(days=days_lookback)
    water_data = water_data[water_data['event_date'] >= cutoff_date]
    erg_data = erg_data[erg_data['test_date'] >= cutoff_date]
    
    results = []
    
    for rower_name in set(water_data['rower_name'].unique()):
        rower_water = water_data[water_data['rower_name'] == rower_name]
        if len(rower_water) < min_pieces:
            continue
            
        # Get rower's erg data
        rower_erg = erg_data[erg_data['rower_name'] == rower_name]
        
        # Calculate metrics
        metrics = {
            'rower_name': rower_name,
            'water_pieces': len(rower_water),
            'avg_pct_off_gms': rower_water['pct_off_gms'].mean(),
            'consistency': rower_water['pct_off_gms'].std(),
            'recency_score': self._calculate_recency_weighted_score(rower_water),
            'boat_class_versatility': self._calculate_versatility(rower_water),
            'weight': rower_erg['weight'].mean() if not rower_erg.empty else None
        }
        
        # Add erg metrics if available
        if not rower_erg.empty:
            best_2k = self._get_best_test(rower_erg, '2k')
            best_5k = self._get_best_test(rower_erg, '5k')
            
            if best_2k is not None:
                metrics['best_2k'] = best_2k
                metrics['watts_per_lb'] = rower_erg[rower_erg['test_type'] == '2k']['watts_per_lb'].max()
            
            if best_5k is not None:
                metrics['best_5k'] = best_5k
                
            # Calculate erg-to-water efficiency
            if best_2k is not None:
                metrics['erg_water_efficiency'] = self._calculate_erg_water_efficiency(best_2k, metrics['avg_pct_off_gms'])
            
        results.append(metrics)
    
    # Convert to DataFrame and calculate composite score
    df_results = pd.DataFrame(results)
    
    # Normalize metrics for composite score
    if not df_results.empty:
        for col in ['avg_pct_off_gms', 'consistency']:
            if col in df_results.columns:
                # Lower is better, so invert
                df_results[f'{col}_norm'] = 1 - ((df_results[col] - df_results[col].min()) / 
                                         (df_results[col].max() - df_results[col].min()))
        
        for col in ['recency_score', 'boat_class_versatility', 'erg_water_efficiency']:
            if col in df_results.columns:
                # Higher is better
                df_results[f'{col}_norm'] = (df_results[col] - df_results[col].min()) / \
                                         (df_results[col].max() - df_results[col].min())
        
        # Calculate power-to-weight normalized score if available
        if 'watts_per_lb' in df_results.columns:
            df_results['p2w_norm'] = (df_results['watts_per_lb'] - df_results['watts_per_lb'].min()) / \
                                   (df_results['watts_per_lb'].max() - df_results['watts_per_lb'].min())
        
        # Create composite efficiency index (weighted formula)
        score_columns = [col for col in df_results.columns if col.endswith('_norm')]
        if score_columns:
            df_results['efficiency_index'] = df_results[score_columns].mean(axis=1)
            
    return df_results.sort_values('efficiency_index', ascending=False)

def optimize_lineup_ml(self, boat_class='8+', rowers_pool=None, n_iterations=1000, balance_sides=True):
    """
    Use machine learning to find the optimal lineup based on:
    1. Individual rower metrics
    2. Compatibility between rowers (from historical lineups)
    3. Side preference
    4. Weight distribution
    5. Seat race results
    """
    from sklearn.ensemble import RandomForestRegressor
    import itertools
    
    # First train a model to predict boat performance
    X_features, y_times = self._prepare_lineup_training_data()
    
    if len(X_features) < 5:
        logger.warning("Not enough historical lineup data to train ML model")
        return None
        
    # Train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_features, y_times)
    
    # Generate candidate lineups
    all_rowers = self._get_rower_pool(rowers_pool, boat_class)
    seats = int(boat_class[0]) if boat_class[0].isdigit() else 8
    
    # Get rower attributes for scoring
    rower_attributes = self._get_rower_attributes(all_rowers)
    
    # Track the best lineups
    best_lineups = []
    best_score = float('inf')
    
    # Generate and evaluate random lineups
    for _ in range(n_iterations):
        # Select random rowers
        if len(all_rowers) < seats:
            continue
            
        lineup = np.random.choice(all_rowers, size=seats, replace=False)
        
        if balance_sides:
            # Balance port and starboard rowers
            lineup = self._balance_lineup_sides(lineup, rower_attributes)
            
        # Generate features for this lineup
        lineup_features = self._generate_lineup_features(lineup, rower_attributes)
        
        # Predict performance
        predicted_time = model.predict([lineup_features])[0]
        
        # Add weight balance score (penalty for imbalance)
        weight_balance = self._calculate_weight_balance(lineup, rower_attributes)
        compatibility = self._calculate_compatibility(lineup)
        
        # Final score (time + penalties)
        adjusted_score = predicted_time + weight_balance + (compatibility * 3)
        
        if adjusted_score < best_score:
            best_score = adjusted_score
            best_lineups.append({
                'lineup': lineup,
                'predicted_time': predicted_time,
                'weight_balance': weight_balance,
                'compatibility': compatibility,
                'total_score': adjusted_score
            })
            
            # Keep only top 5 lineups
            best_lineups = sorted(best_lineups, key=lambda x: x['total_score'])[:5]
    
    return best_lineups

def analyze_rower_compatibility_network(self, show_plot=True, save_path=None):
    """
    Analyze how well rowers perform together using network analysis.
    The thickness of connections represents compatibility score.
    """
    try:
        import networkx as nx
        import community as community_louvain
    except ImportError:
        logger.error("This analysis requires networkx and python-louvain packages")
        return None
    
    # Calculate compatibility matrix
    compatibility_matrix = self._calculate_rower_compatibility_matrix()
    
    if compatibility_matrix is None or compatibility_matrix.empty:
        logger.warning("Not enough data to create compatibility network")
        return None
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (rowers)
    for rower in compatibility_matrix.index:
        G.add_node(rower)
    
    # Add edges (compatibility scores)
    for i in range(len(compatibility_matrix.index)):
        for j in range(i+1, len(compatibility_matrix.columns)):
            rower1 = compatibility_matrix.index[i]
            rower2 = compatibility_matrix.columns[j]
            compatibility = compatibility_matrix.iloc[i, j]
            
            if not pd.isna(compatibility) and compatibility > 0:
                G.add_edge(rower1, rower2, weight=compatibility)
    
    # Find communities
    partition = community_louvain.best_partition(G)
    
    if show_plot or save_path:
        plt.figure(figsize=(16, 16))
        
        # Node positions
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Node colors based on community
        cmap = plt.cm.rainbow
        colors = [cmap(partition[node]/max(partition.values())) for node in G.nodes()]
        
        # Node sizes based on centrality
        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
        node_sizes = [centrality[node] * 5000 + 100 for node in G.nodes()]
        
        # Edge widths based on compatibility
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('Rower Compatibility Network (Communities in Colors)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved compatibility network to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Return communities and compatibility matrix
    return {
        'communities': partition,
        'compatibility_matrix': compatibility_matrix,
        'graph': G
    }


def analyze_erg_water_translation_factors(self, show_plot=True, save_path=None):
    """
    Analyze which factors influence how well erg scores translate to water performance.
    Factors include: weight, height, experience, technique, weight-adjusted power, etc.
    """
    # Get combined data
    combined = self.erg_processor.combine_data_sources()
    
    if combined is None or combined.empty:
        logger.warning("No combined data available for analysis")
        return None
    
    # Calculate erg-to-water efficiency for each rower
    eff_data = self.erg_processor.analyze_erg_water_efficiency_ratio(
        min_water_pieces=3, show_plot=False
    )
    
    if eff_data is None or eff_data.empty:
        logger.warning("Not enough data to calculate erg-water efficiency")
        return None
    
    # Merge with additional metrics
    analysis_df = pd.merge(eff_data, combined, on='rower_name', how='inner')
    
    if analysis_df.empty:
        logger.warning("No data after merging")
        return None
    
    # Add derived metrics for analysis
    if 'weight' in analysis_df.columns and 'erg_time' in analysis_df.columns:
        # Calculate BMI if height is available
        if 'height' in analysis_df.columns:
            analysis_df['bmi'] = analysis_df['weight'] / (analysis_df['height'] ** 2)
        
        # Calculate power-to-weight specific factors
        if 'watts_per_lb' in analysis_df.columns:
            analysis_df['power_to_weight_cubed'] = analysis_df['watts_per_lb'] ** (1/3)
    
    # Analyze correlation between factors and erg-water efficiency
    correlation_cols = ['efficiency_ratio', 'weight', 'watts_per_lb', 'power_to_weight_cubed', 
                       'bmi', 'erg_time', 'weighted_water_time']
    
    correlation_cols = [col for col in correlation_cols if col in analysis_df.columns]
    correlation_matrix = analysis_df[correlation_cols].corr()
    
    # Visualize relationships
    if show_plot or save_path:
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Plot 1: Efficiency vs Power-to-Weight
        if 'watts_per_lb' in analysis_df.columns:
            ax = axes[0, 0]
            sns.scatterplot(
                x='watts_per_lb', 
                y='efficiency_ratio', 
                size='weight',
                hue='piece_count',
                data=analysis_df, 
                ax=ax,
                palette='viridis'
            )
            
            # Add trendline
            x = analysis_df['watts_per_lb']
            y = analysis_df['efficiency_ratio']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.7)
            
            # Add annotations
            for _, row in analysis_df.iterrows():
                ax.annotate(
                    row['rower_name'],
                    (row['watts_per_lb'], row['efficiency_ratio']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8
                )
                
            ax.set_title('Efficiency vs Power-to-Weight')
            ax.set_xlabel('Watts per Pound')
            ax.set_ylabel('Erg-to-Water Efficiency')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency vs Weight
        if 'weight' in analysis_df.columns:
            ax = axes[0, 1]
            sns.scatterplot(
                x='weight', 
                y='efficiency_ratio', 
                size='watts_per_lb',
                hue='piece_count',
                data=analysis_df, 
                ax=ax,
                palette='viridis'
            )
            
            # Add trendline
            x = analysis_df['weight']
            y = analysis_df['efficiency_ratio']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.7)
            
            # Add annotations
            for _, row in analysis_df.iterrows():
                ax.annotate(
                    row['rower_name'],
                    (row['weight'], row['efficiency_ratio']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8
                )
                
            ax.set_title('Efficiency vs Weight')
            ax.set_xlabel('Weight (lbs)')
            ax.set_ylabel('Erg-to-Water Efficiency')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Correlation heatmap
        ax = axes[1, 0]
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            ax=ax
        )
        ax.set_title('Correlation Matrix')
        
        # Plot 4: Optional 3D plot or additional analysis
        ax = axes[1, 1]
        if 'watts_per_lb' in analysis_df.columns and 'weight' in analysis_df.columns:
            from mpl_toolkits.mplot3d import Axes3D
            ax.remove()
            ax = fig.add_subplot(2, 2, 4, projection='3d')
            
            scatter = ax.scatter(
                analysis_df['watts_per_lb'],
                analysis_df['weight'],
                analysis_df['efficiency_ratio'],
                c=analysis_df['efficiency_ratio'],
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            
            ax.set_xlabel('Watts per Pound')
            ax.set_ylabel('Weight (lbs)')
            ax.set_zlabel('Efficiency Ratio')
            ax.set_title('3D Relationship')
            
            fig.colorbar(scatter, ax=ax, label='Efficiency Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved erg-water translation analysis to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Calculate regression model to predict efficiency from metrics
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    X_cols = ['weight', 'watts_per_lb'] if 'watts_per_lb' in analysis_df.columns else ['weight']
    X_cols = [col for col in X_cols if col in analysis_df.columns]
    
    if X_cols:
        X = analysis_df[X_cols].dropna()
        y = analysis_df.loc[X.index, 'efficiency_ratio']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Get coefficients
        coef_dict = {X_cols[i]: model.coef_[i] for i in range(len(X_cols))}
        
        return {
            'correlation_matrix': correlation_matrix,
            'regression_coefficients': coef_dict,
            'intercept': model.intercept_,
            'scaler': scaler,
            'model': model,
            'r2_score': model.score(X_scaled, y)
        }
    
    return {'correlation_matrix': correlation_matrix}

if __name__ == "__main__":
    analyzer = ComprehensiveAnalysis()
    analyzer.run_complete_analysis()