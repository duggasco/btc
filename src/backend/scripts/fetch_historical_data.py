#!/usr/bin/env python3
"""
Utility script to fetch and store historical data
Can be run manually or scheduled via cron
"""
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.backend.services.data_fetcher import get_fetcher
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_historical_data(symbols, granularities, force_refresh=False):
    """
    Fetch historical data for specified symbols and granularities
    
    Args:
        symbols: List of symbols to fetch
        granularities: List of granularities to fetch
        force_refresh: Whether to force refresh existing data
    """
    fetcher = get_fetcher()
    
    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")
        
        for granularity in granularities:
            try:
                logger.info(f"  Fetching {granularity} data...")
                
                # Fetch extended historical data
                df = fetcher.fetch_extended_historical_data(
                    symbol, 
                    granularity=granularity,
                    force_refresh=force_refresh
                )
                
                if not df.empty:
                    logger.info(f"  ✓ Fetched {len(df)} {granularity} records")
                    logger.info(f"    Date range: {df.index.min()} to {df.index.max()}")
                else:
                    logger.warning(f"  ✗ No data fetched for {granularity}")
                
                # Check for gaps and try to fill them
                availability = fetcher.get_data_availability(symbol)
                gran_info = availability.get(granularity, {})
                missing_periods = gran_info.get('missing_periods', [])
                
                if missing_periods:
                    logger.info(f"  Found {len(missing_periods)} gaps, attempting to fill...")
                    filled = fetcher.fill_data_gaps(symbol, granularity)
                    logger.info(f"  ✓ Filled {filled} missing records")
                
            except Exception as e:
                logger.error(f"  ✗ Error fetching {granularity} data: {e}")

def check_data_status(symbols):
    """Check and report data availability status"""
    fetcher = get_fetcher()
    
    print("\n" + "=" * 80)
    print("DATA AVAILABILITY REPORT")
    print("=" * 80)
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        print("-" * 40)
        
        try:
            availability = fetcher.get_data_availability(symbol)
            
            if not availability:
                print("  No data available")
                continue
            
            for granularity, info in sorted(availability.items()):
                records = info.get('total_records', 0)
                earliest = info.get('earliest', 'N/A')
                latest = info.get('latest', 'N/A')
                missing = info.get('missing_periods', [])
                
                print(f"\n  {granularity}:")
                print(f"    Records: {records:,}")
                print(f"    Range: {earliest} to {latest}")
                
                if missing:
                    print(f"    Gaps: {len(missing)}")
                    # Show first few gaps
                    for i, gap in enumerate(missing[:3]):
                        print(f"      - {gap['start']} to {gap['end']} ({gap['count']} missing)")
                    if len(missing) > 3:
                        print(f"      ... and {len(missing) - 3} more gaps")
                else:
                    print("    Gaps: None")
                    
        except Exception as e:
            print(f"  Error checking availability: {e}")
    
    # Overall statistics
    try:
        stats = fetcher.get_historical_stats()
        overall = stats.get('overall', {})
        
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        print(f"Total Symbols: {overall.get('symbols', 0)}")
        print(f"Total Sources: {overall.get('sources', 0)}")
        print(f"Total Records: {overall.get('total_records', 0):,}")
        print(f"Date Range: {overall.get('earliest_data', 'N/A')} to {overall.get('latest_data', 'N/A')}")
        
        print("\nBy Source:")
        for source, source_stats in stats.get('by_source', {}).items():
            print(f"  {source}:")
            print(f"    Collections: {source_stats.get('collections', 0)}")
            print(f"    Records: {source_stats.get('total_records', 0):,}")
            print(f"    Success Rate: {source_stats.get('success_rate', 0):.1f}%")
            print(f"    Avg Time: {source_stats.get('avg_time', 0):.1f}s")
            
    except Exception as e:
        print(f"\nError getting statistics: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Fetch and manage historical market data'
    )
    
    parser.add_argument(
        'command',
        choices=['fetch', 'status', 'fill-gaps'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTC'],
        help='Symbols to process (default: BTC)'
    )
    
    parser.add_argument(
        '--granularities',
        nargs='+',
        default=['1d'],
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Data granularities to fetch (default: 1d)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if data exists'
    )
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        fetch_historical_data(args.symbols, args.granularities, args.force)
    elif args.command == 'status':
        check_data_status(args.symbols)
    elif args.command == 'fill-gaps':
        fetcher = get_fetcher()
        for symbol in args.symbols:
            for granularity in args.granularities:
                logger.info(f"Filling gaps for {symbol} {granularity}...")
                filled = fetcher.fill_data_gaps(symbol, granularity)
                logger.info(f"Filled {filled} records")

if __name__ == "__main__":
    main()