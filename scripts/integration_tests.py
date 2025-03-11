import unittest
from rowing_analysis import RowingAnalysis, GMS_TIMES
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class TestRowingAnalysis(unittest.TestCase):
    def setUp(self):
        # Create mock performance data
        self.mock_performance_data = pd.DataFrame({
            'rower_id': [1, 1, 2, 2, 3, 3],
            'rower_name': ['Rower1', 'Rower1', 'Rower2', 'Rower2', 'Rower3', 'Rower3'],
            'time': [360.5, 365.2, 370.2, 375.1, 355.8, 358.9],
            'boat_identifier': ['1v 8+', '1v 8+', '1v 8+', '1v 8+', '1v 8+', '1v 8+'],
            'event_date': ['2023-09-01', '2023-09-15', '2023-09-01', '2023-09-15', '2023-09-01', '2023-09-15']
        })
        
        # Create a mock analyzer with patched database connection
        with patch('rowing_analysis.create_engine') as mock_create_engine:
            self.analyzer = RowingAnalysis('mock_connection')
            self.analyzer.performance_data = self.mock_performance_data
    
    def test_calculate_gms_comparison(self):
        """Test GMS comparison calculation"""
        # Add expected GMS time to mock data
        self.analyzer.performance_data['gms_time'] = GMS_TIMES['1v 8+']
        
        result = self.analyzer.calculate_gms_comparison()
        
        # Verify the result has expected columns and length
        self.assertEqual(len(result), 3)  # 3 unique rowers
        self.assertIn('avg_pct_off_gms', result.columns)
        self.assertIn('rank', result.columns)
        
        # Verify rower3 has the lowest GMS percentage (best performance)
        best_rower = result.loc[result['rank'] == 1, 'rower_name'].iloc[0]
        self.assertEqual(best_rower, 'Rower3')
    
    def test_analyze_rower_progress(self):
        """Test rower progress analysis"""
        # Add expected GMS time to mock data
        self.analyzer.performance_data['gms_time'] = GMS_TIMES['1v 8+']
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.show'):
                result = self.analyzer.analyze_rower_progress(
                    rower_name='Rower1',
                    show_plot=False
                )
        
        # Verify result contains expected keys
        self.assertIn('rower_name', result)
        self.assertIn('avg_pct_off_gms', result)
        self.assertIn('trend_per_day', result)
        
        # Verify correct rower name
        self.assertEqual(result['rower_name'], 'Rower1')
        
        # Verify piece count
        self.assertEqual(result['num_pieces'], 2)