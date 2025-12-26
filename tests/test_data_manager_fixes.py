
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_manager import DataManager

class TestDataManagerFixes(unittest.TestCase):
    
    def setUp(self):
        # Mock environment
        self.env_patcher = patch.dict(os.environ, {
            'S3_BUCKET': 'test-bucket',
            'S3_PREFIX': 'data/raw/',
            'AWS_REGION': 'us-east-1'
        })
        self.env_patcher.start()
        
        # Mock boto3 client
        self.boto_patcher = patch('boto3.client')
        self.mock_boto = self.boto_patcher.start()
        self.mock_s3 = MagicMock()
        self.mock_boto.return_value = self.mock_s3
        
        # Initialize DataManager in cloud mode
        with patch('utils.data_manager.S3_AVAILABLE', True):
           self.dm = DataManager(mode='cloud', data_type='raw')
           # Inject mock client directly ensures usage even if init created a different one
           self.dm.s3_client = self.mock_s3
           
    def tearDown(self):
        self.env_patcher.stop()
        self.boto_patcher.stop()

    def test_pagination_in_get_latest_date(self):
        """Test that _get_latest_date_from_s3 handles pagination correctly."""
        # Setup mock to return two pages of results
        # Page 1: Old files
        # Page 2: The newest file
        
        page1 = {
            'Contents': [{'Key': 'data/raw/data_2020-01-01_from_2019-01-01.csv'}]
        }
        page2 = {
            'Contents': [{'Key': 'data/raw/data_2025-01-01_from_2020-01-01.csv'}] # This is the latest
        }
        
        # Configure paginator mock is complex, so we'll mock the internal method call structure
        # usage: s3_client.get_paginator('list_objects_v2').paginate(...)
        mock_paginator = MagicMock()
        self.mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [page1, page2]
        
        # EXECUTE
        latest_date = self.dm._get_latest_date_from_s3()
        
        # VERIFY
        # Should find the date from page 2 (2025-01-01)
        expected_date = datetime(2025, 1, 1)
        self.assertEqual(latest_date, expected_date)
        
        # Verify paginator was used
        self.mock_s3.get_paginator.assert_called_with('list_objects_v2')
        mock_paginator.paginate.assert_called_with(Bucket='test-bucket', Prefix='data/raw/')

    def test_safe_cleanup_redundant_files(self):
        """Test that saves delete superseded files."""
        # Scenario: 
        # Existing: 'data_2024-12-31_from_2024-01-01.csv' (Full year 2024)
        # New Save: 'data_2025-01-01_from_2024-01-01.csv' (2024 + 1 day)
        # Result: Old file is fully contained -> DELETE
        
        # Mock listing existing files
        existing_files = {
            'Contents': [
                {'Key': 'data/raw/data_2024-12-31_from_2024-01-01.csv'}
            ]
        }
        
        # Mock paginator for the check inside save
        mock_paginator = MagicMock()
        self.mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [existing_files]
        
        # Dataframe to save (index range 2024-01-01 to 2025-01-01)
        dates = pd.date_range(start='2024-01-01', end='2025-01-01')
        df = pd.DataFrame(index=dates, data={'Close': range(len(dates))})
        df.index.name = 'Date'
        
        # EXECUTE
        with patch.object(self.dm, '_save_to_local', return_value=True): # Avoid local FS ops
            with patch('os.remove'): # Avoid local cleanup
                self.dm.save_data(df)
        
        # VERIFY
        # Check delete_object called for the 2024 file
        self.mock_s3.delete_object.assert_called_with(
            Bucket='test-bucket', 
            Key='data/raw/data_2024-12-31_from_2024-01-01.csv'
        )

    def test_safe_cleanup_preserves_disjoint_files(self):
        """Test that saves do NOT delete disjoint daily files."""
        # Scenario:
        # Existing: 'data_2025-01-01_from_2025-01-01.csv' (Daily file)
        # New Save: 'data_2025-01-02_from_2025-01-02.csv' (Next day)
        # Result: Disjoint -> KEEP Existing
        
        existing_files = {
            'Contents': [
                {'Key': 'data/raw/data_2025-01-01_from_2025-01-01.csv'}
            ]
        }
        
        mock_paginator = MagicMock()
        self.mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [existing_files]
        
        dates = pd.date_range(start='2025-01-02', end='2025-01-02')
        df = pd.DataFrame(index=dates, data={'Close': range(len(dates))})
        df.index.name = 'Date'
        
        # EXECUTE
        with patch.object(self.dm, '_save_to_local', return_value=True):
            with patch('os.remove'):
                self.dm.save_data(df)
        
        # VERIFY
        # Should NOT delete anything
        self.mock_s3.delete_object.assert_not_called()

if __name__ == '__main__':
    unittest.main()
