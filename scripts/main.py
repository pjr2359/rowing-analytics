import argparse
import logging
import os
from dotenv import load_dotenv

from rowing_analysis import RowingAnalysis
from erg_data_processor import ErgDataProcessor
from parse_and_load import RowingDataParser
from comprehensive_analysis import ComprehensiveAnalysis
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rowing_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the rowing analytics tool"""
    parser = argparse.ArgumentParser(description='Rowing Analytics Tool')
    parser.add_argument('--parse', action='store_true', help='Parse and load CSV files')
    parser.add_argument('--analyze', action='store_true', help='Run analysis')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--rower', type=str, help='Analyze specific rower')
    parser.add_argument('--days', type=int, default=90, help='Days to look back')
    parser.add_argument('--output', type=str, default='analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    password = os.getenv('PASSWORD')
    
    if not password:
        logger.error("Database password not found in environment variables")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create database connection
    engine = create_engine(f'postgresql://postgres:{password}@localhost/rowing-analytics')
    
    # Parse and load data if requested
    if args.parse:
        logger.info("Parsing and loading data...")
        parser = RowingDataParser(engine)
        data_dir = './data'
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and not filename.endswith('.csv:Zone.Identifier'):
                file_path = os.path.join(data_dir, filename)
                parsed_data = parser.parse_csv_file(file_path)
                if parsed_data:
                    parser.load_data_to_db(parsed_data)
    
    # Analyze data if requested
    if args.analyze:
        logger.info("Running analysis...")
        analyzer = RowingAnalysis(engine)
        analyzer.load_performance_data()
        analyzer.load_seat_race_data()
        
        # GMS Rankings
        gms_data = analyzer.calculate_gms_comparison()
        if not gms_data.empty:
            analyzer.visualize_gms_rankings(
                min_pieces=1, 
                show_plot=False, 
                save_path=os.path.join(args.output, 'gms_rankings.png')
            )
        
        # Run comprehensive analysis
        comp_analyzer = ComprehensiveAnalysis()
        comp_analyzer.run_complete_analysis()
    
    # Analyze specific rower if requested
    if args.rower:
        logger.info(f"Analyzing rower: {args.rower}")
        analyzer = RowingAnalysis(engine)
        analyzer.analyze_rower_progress(
            rower_name=args.rower, 
            days_back=args.days,
            show_plot=False, 
            save_path=os.path.join(args.output, f'{args.rower}_progress.png')
        )
        
        analyzer.analyze_seat_races(
            rower_name=args.rower,
            show_plot=False, 
            save_path=os.path.join(args.output, f'{args.rower}_seat_races.png')
        )

if __name__ == "__main__":
    main()