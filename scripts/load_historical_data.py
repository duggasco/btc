#!/usr/bin/env python3
"""
Script to load historical BTC data for backtesting
Forces the system to fetch and cache more historical data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API base URL
API_BASE = "http://localhost:8090"

def check_current_data():
    """Check current data availability"""
    try:
        # Check BTC history
        response = requests.get(f"{API_BASE}/btc/history/1y")
        if response.status_code == 200:
            data = response.json()
            points = len(data.get('data', []))
            if points > 0:
                first_date = data['data'][0]['index'][:10]
                last_date = data['data'][-1]['index'][:10]
                logger.info(f"Current data: {points} points from {first_date} to {last_date}")
            else:
                logger.warning("No data points available")
        else:
            logger.error(f"Failed to fetch data: {response.status_code}")
    except Exception as e:
        logger.error(f"Error checking data: {e}")

def trigger_data_fetch():
    """Trigger fetching of historical data through various endpoints"""
    
    # 1. Try to fetch longer periods through BTC history endpoint
    periods = ['1y', '6m', '3m', '1m', '7d']
    for period in periods:
        try:
            logger.info(f"Fetching {period} of BTC history...")
            response = requests.get(f"{API_BASE}/btc/history/{period}")
            if response.status_code == 200:
                data = response.json()
                points = len(data.get('data', []))
                logger.info(f"  -> Got {points} data points for {period}")
            else:
                logger.warning(f"  -> Failed to fetch {period}: {response.status_code}")
        except Exception as e:
            logger.error(f"  -> Error fetching {period}: {e}")
    
    # 2. Try to warm the cache with historical data
    try:
        logger.info("Warming cache with historical data...")
        response = requests.post(f"{API_BASE}/cache/warm")
        if response.status_code == 200:
            logger.info("  -> Cache warming initiated")
        else:
            logger.warning(f"  -> Cache warming failed: {response.status_code}")
    except Exception as e:
        logger.error(f"  -> Error warming cache: {e}")
    
    # 3. Try to fetch comprehensive signals (which may trigger data loading)
    try:
        logger.info("Fetching comprehensive signals...")
        response = requests.get(f"{API_BASE}/signals/comprehensive")
        if response.status_code == 200:
            logger.info("  -> Comprehensive signals fetched successfully")
        else:
            logger.warning(f"  -> Failed to fetch signals: {response.status_code}")
    except Exception as e:
        logger.error(f"  -> Error fetching signals: {e}")
    
    # 4. Try to trigger model training (which requires historical data)
    try:
        logger.info("Triggering enhanced LSTM data status check...")
        response = requests.get(f"{API_BASE}/enhanced-lstm/data-status")
        if response.status_code == 200:
            status = response.json()
            logger.info(f"  -> Data status: {status}")
        else:
            logger.warning(f"  -> Failed to check data status: {response.status_code}")
    except Exception as e:
        logger.error(f"  -> Error checking data status: {e}")

def run_test_backtest():
    """Run a test backtest to verify data availability"""
    try:
        logger.info("Running test backtest...")
        
        # Try to backtest last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        backtest_params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "initial_capital": 10000,
            "position_size": 0.1,
            "stop_loss": 0.05,
            "take_profit": 0.1
        }
        
        response = requests.post(f"{API_BASE}/backtest/run", json=backtest_params)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"  -> Backtest completed: {result.get('total_trades', 0)} trades")
            logger.info(f"  -> Final balance: ${result.get('final_balance', 0):,.2f}")
            logger.info(f"  -> Total return: {result.get('total_return', 0):.2%}")
        else:
            logger.error(f"  -> Backtest failed: {response.status_code}")
            if response.text:
                logger.error(f"  -> Error: {response.text}")
    except Exception as e:
        logger.error(f"  -> Error running backtest: {e}")

def main():
    """Main function"""
    logger.info("=== Historical Data Loader ===")
    
    # Check current data
    logger.info("\n1. Checking current data availability...")
    check_current_data()
    
    # Trigger data fetching
    logger.info("\n2. Triggering data fetch from various sources...")
    trigger_data_fetch()
    
    # Check data again
    logger.info("\n3. Checking data availability after fetch...")
    check_current_data()
    
    # Run test backtest
    logger.info("\n4. Running test backtest...")
    run_test_backtest()
    
    logger.info("\n=== Data loading complete ===")
    logger.info("Note: The system currently fetches limited historical data (3 months on startup).")
    logger.info("For extensive backtesting, you may need to:")
    logger.info("  1. Modify the startup code to fetch more data (e.g., 1 year)")
    logger.info("  2. Use external data sources to populate the database")
    logger.info("  3. Implement a historical data import feature")

if __name__ == "__main__":
    main()