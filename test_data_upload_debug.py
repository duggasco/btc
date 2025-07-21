#!/usr/bin/env python3
"""Debug data upload service"""

import sys
import os
sys.path.append('/root/btc/src/backend')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_upload_service import DataUploadService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_upload_service():
    """Test the data upload service directly"""
    
    # Create test data
    dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='D')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(95000, 96000, len(dates)),
        'high': np.random.uniform(96000, 97000, len(dates)),
        'low': np.random.uniform(94000, 95000, len(dates)),
        'close': np.random.uniform(95000, 96000, len(dates)),
        'volume': np.random.uniform(1000, 2000, len(dates))
    })
    
    # Save to temp CSV
    temp_file = '/tmp/test_ohlcv_data.csv'
    test_data.to_csv(temp_file, index=False)
    logger.info(f"Created test file with {len(test_data)} rows")
    
    # Initialize service
    service = DataUploadService()
    logger.info("Initialized DataUploadService")
    
    try:
        # Test preview
        preview_result = service.preview_file(temp_file, 'csv')
        logger.info(f"Preview result: {preview_result['columns']}")
        
        # Test upload
        result = service.process_upload(
            file_path=temp_file,
            file_type='csv',
            source='test_upload',
            data_type='price',
            symbol='BTC'
        )
        
        logger.info(f"Upload result: {result}")
        
        # Check database directly
        import sqlite3
        conn = sqlite3.connect('/root/btc/storage/data/historical_data.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM ohlcv_data WHERE source='test_upload'")
        count = cursor.fetchone()[0]
        logger.info(f"Database check: Found {count} rows with source='test_upload'")
        
        # Get sample rows
        cursor.execute("SELECT * FROM ohlcv_data WHERE source='test_upload' LIMIT 5")
        rows = cursor.fetchall()
        logger.info(f"Sample rows: {rows}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    test_upload_service()