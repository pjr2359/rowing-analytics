import os
import argparse
import logging
from erg_data_processor import ErgDataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs('analysis_output', exist_ok=True)

def main():
    """Run various rowing analyses based on command line arguments"""
    parser = argparse.ArgumentParser(description='Run rowing analytics on erg and water data')
    parser.add_argument('--analysis', type=str, required=True, 
                       choices=['efficiency', 'lineup', 'seat_races', 'correlations',
                                'boat_specialists', 'progression', 'all'],
                       help='Type of analysis to run')
    parser.add_argument('--rower', type=str, help='Rower name for individual analysis')
    parser.add_argument('--boat-class', type=str, default='8+', help='Boat class for analysis')
    parser.add_argument('--test-type', type=str, default='2k', help='Erg test type (2k, 6k, etc)')
    parser.add_argument('--save-path', type=str, help='Path to save output')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ErgDataProcessor()
    processor.load_erg_data()
    processor.load_on_water_data()
    processor.combine_data_sources()
    
    # Run requested analysis
    if args.analysis == 'efficiency':
        save_path = args.save_path or 'analysis_output/erg_water_efficiency.png'
        result = processor.analyze_erg_water_efficiency_ratio(
            test_type=args.test_type,
            show_plot=True,
            save_path=save_path
        )
        logger.info(f"Efficiency analysis complete. Results saved to {save_path}")
        
    elif args.analysis == 'lineup':
        save_path = args.save_path or f'analysis_output/optimal_{args.boat_class}_lineup.png'
        
        # Get all rowers from database
        available_rowers = processor.combined_data['rower_name'].unique().tolist()
        
        result = processor.predict_optimal_lineup(
            available_rowers=available_rowers,
            boat_class=args.boat_class,
            show_plot=True,
            save_path=save_path
        )
        logger.info(f"Lineup optimization complete. Results saved to {save_path}")
        
    elif args.analysis == 'seat_races':
        save_path = args.save_path or 'analysis_output/seat_race_analysis.png'
        result = processor.analyze_seat_race_with_erg_context(
            show_plot=True,
            save_path=save_path
        )
        logger.info(f"Seat race analysis complete. Results saved to {save_path}")
        
    elif args.analysis == 'correlations':
        save_path = args.save_path or 'analysis_output/erg_water_correlation.png'
        result = processor.analyze_water_erg_correlation_matrix(
            test_type=args.test_type,
            show_plot=True,
            save_path=save_path
        )
        logger.info(f"Correlation analysis complete. Results saved to {save_path}")
        
    elif args.analysis == 'boat_specialists':
        save_path = args.save_path or 'analysis_output/boat_class_specialists.png'
        result = processor.analyze_boat_class_specialists(
            show_plot=True,
            save_path=save_path
        )
        logger.info(f"Boat class specialization analysis complete. Results saved to {save_path}")
        
    elif args.analysis == 'progression':
        if not args.rower:
            logger.error("Rower name required for progression analysis. Use --rower")
            return
            
        save_path = args.save_path or f'analysis_output/{args.rower}_progression.png'
        result = processor.analyze_rower_progression(
            rower_name=args.rower,
            test_type=args.test_type,
            show_plot=True,
            save_path=save_path
        )
        logger.info(f"Progression analysis complete. Results saved to {save_path}")
        
    elif args.analysis == 'all':
        # Run all analyses
        logger.info("Running all analyses...")
        
        # Efficiency
        processor.analyze_erg_water_efficiency_ratio(
            show_plot=False, 
            save_path='analysis_output/erg_water_efficiency.png'
        )
        
        # Water-erg correlation
        processor.analyze_water_erg_correlation_matrix(
            show_plot=False,
            save_path='analysis_output/erg_water_correlation_matrix.png'
        )
        
        # Boat specialists
        processor.analyze_boat_class_specialists(
            show_plot=False,
            save_path='analysis_output/boat_class_specialists.png'
        )
        
        # Seat races
        processor.analyze_seat_race_with_erg_context(
            show_plot=False,
            save_path='analysis_output/seat_race_analysis.png'
        )
        
        # Top rowers
        top_rowers = processor.combined_data['rower_name'].value_counts().head(5).index.tolist()
        for rower in top_rowers:
            processor.analyze_rower_progression(
                rower_name=rower,
                show_plot=False,
                save_path=f'analysis_output/{rower}_progression.png'
            )
            
        logger.info("All analyses complete. Results saved to analysis_output/ directory")

if __name__ == "__main__":
    main()