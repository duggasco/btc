"""
Historical Data Manager
Provides extended historical data collection and persistent storage beyond cache
Manages aggregation from multiple sources and data continuity
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)

class ConflictResolution(Enum):
    """Strategies for handling conflicts when inserting data"""
    REPLACE = "replace"          # Delete existing and insert new
    MERGE = "merge"              # Keep both, prefer newer on conflict
    FILL_GAPS = "fill_gaps"      # Only insert where data doesn't exist
    AVERAGE = "average"          # Average values on timestamp conflicts

class HistoricalDataManager:
    """Manages historical data collection, storage, and retrieval"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv('HISTORICAL_DB_PATH', '/app/storage/data/historical_data.db')
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
        
        # Source configurations with their maximum historical capabilities
        self.source_configs = {
            'coingecko': {
                'max_days': 365,  # Free tier limitation
                'granularity': {
                    '1d': 365,
                    '1h': 90,
                    '5m': 1
                }
            },
            'binance': {
                'max_days': 1000,
                'granularity': {
                    '1d': 1000,
                    '1h': 720,  # 30 days
                    '5m': 72    # 3 days
                }
            },
            'cryptocompare': {
                'max_days': 2000,
                'granularity': {
                    '1d': 2000,
                    '1h': 168,  # 7 days
                    '1m': 1440  # 1 day
                }
            },
            'alphavantage': {
                'max_days': 1000,
                'granularity': {
                    '1d': 1000,
                    '1h': 60,   # 2.5 days with rate limits
                    '5m': 5     # Very limited
                }
            }
        }
        
    def init_database(self):
        """Initialize database tables for historical data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                granularity TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, source, timestamp, granularity)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp 
            ON ohlcv_data(symbol, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ohlcv_source_timestamp 
            ON ohlcv_data(source, timestamp)
        ''')
        
        # Data collection metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                start_date DATETIME NOT NULL,
                end_date DATETIME NOT NULL,
                granularity TEXT NOT NULL,
                records_collected INTEGER,
                collection_time REAL,
                status TEXT,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data continuity tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_continuity (
                symbol TEXT NOT NULL,
                granularity TEXT NOT NULL,
                earliest_date DATETIME NOT NULL,
                latest_date DATETIME NOT NULL,
                total_records INTEGER,
                missing_periods TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, granularity)
            )
        ''')
        
        # Upload tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS upload_tracking (
                upload_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                data_type TEXT NOT NULL,
                symbol TEXT,
                granularity TEXT,
                start_date DATETIME,
                end_date DATETIME,
                total_rows INTEGER,
                inserted_rows INTEGER,
                skipped_rows INTEGER,
                error_rows INTEGER,
                conflict_resolution TEXT,
                error_message TEXT,
                processing_time REAL
            )
        ''')
        
        # On-chain data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS onchain_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metric TEXT NOT NULL,
                value REAL,
                source TEXT,
                upload_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, metric),
                FOREIGN KEY(upload_id) REFERENCES upload_tracking(upload_id)
            )
        ''')
        
        # Sentiment data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                sentiment_score REAL,
                sentiment_label TEXT,
                volume INTEGER,
                upload_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, source),
                FOREIGN KEY(upload_id) REFERENCES upload_tracking(upload_id)
            )
        ''')
        
        # Create indexes for new tables
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_onchain_symbol_timestamp 
            ON onchain_data(symbol, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp 
            ON sentiment_data(symbol, timestamp)
        ''')
        
        # Add upload_id to existing ohlcv_data table if not exists
        cursor.execute('''
            PRAGMA table_info(ohlcv_data)
        ''')
        columns = [row[1] for row in cursor.fetchall()]
        if 'upload_id' not in columns:
            cursor.execute('''
                ALTER TABLE ohlcv_data ADD COLUMN upload_id TEXT
            ''')
        
        conn.commit()
        conn.close()
        
    def fetch_maximum_historical_data(self, symbol: str, sources: List[Any], 
                                    granularity: str = '1d') -> pd.DataFrame:
        """
        Fetch maximum available historical data from multiple sources
        
        Args:
            symbol: Symbol to fetch (e.g., 'BTC', 'bitcoin')
            sources: List of data source instances
            granularity: Data granularity ('1m', '5m', '1h', '1d')
            
        Returns:
            Combined DataFrame with all available historical data
        """
        all_data = []
        collection_results = []
        
        # Use thread pool for parallel fetching
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {}
            
            for source in sources:
                source_name = getattr(source, 'name', source.__class__.__name__.lower())
                
                # Skip if source doesn't support the granularity
                if source_name not in self.source_configs:
                    continue
                    
                max_days = self._get_max_days_for_source(source_name, granularity)
                if max_days == 0:
                    continue
                
                # Submit fetch task
                future = executor.submit(
                    self._fetch_from_source_with_pagination,
                    source, symbol, max_days, granularity
                )
                future_to_source[future] = source_name
            
            # Collect results
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                start_time = time.time()
                
                try:
                    df, metadata = future.result()
                    if df is not None and not df.empty:
                        # Add source column
                        df['source'] = source_name
                        all_data.append(df)
                        
                        # Log collection
                        self._log_collection(
                            symbol, source_name, metadata['start_date'],
                            metadata['end_date'], granularity,
                            len(df), time.time() - start_time, 'success'
                        )
                        
                        logger.info(f"Collected {len(df)} records from {source_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch from {source_name}: {e}")
                    self._log_collection(
                        symbol, source_name, datetime.now(), datetime.now(),
                        granularity, 0, time.time() - start_time, 'failed', str(e)
                    )
        
        # Combine and deduplicate data
        if all_data:
            combined_df = self._combine_and_deduplicate(all_data)
            
            # Store in database
            self._store_historical_data(combined_df, symbol, granularity)
            
            # Update continuity tracking
            self._update_continuity_tracking(symbol, granularity, combined_df)
            
            return combined_df
        
        return pd.DataFrame()
    
    def _fetch_from_source_with_pagination(self, source: Any, symbol: str, 
                                         max_days: int, granularity: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch data from a source with pagination for maximum history
        
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        all_data = []
        end_date = datetime.now()
        
        # Map granularity to period format expected by sources
        period_map = {
            '1m': '1d',    # Most sources don't support minute data beyond 1 day
            '5m': '7d',
            '15m': '7d',
            '1h': '30d',
            '4h': '90d',
            '1d': 'max'
        }
        
        # For daily data, fetch in chunks to avoid timeouts
        if granularity == '1d':
            chunk_size = 365  # Fetch 1 year at a time
            current_end = end_date
            
            while max_days > 0:
                days_to_fetch = min(chunk_size, max_days)
                current_start = current_end - timedelta(days=days_to_fetch)
                
                try:
                    # Use the source's fetch method with appropriate period
                    period = f"{days_to_fetch}d"
                    df = source.fetch(symbol, period)
                    
                    if df is not None and not df.empty:
                        # Filter to our date range
                        mask = (df.index >= current_start) & (df.index <= current_end)
                        df_filtered = df[mask]
                        
                        if not df_filtered.empty:
                            all_data.append(df_filtered)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch chunk: {e}")
                
                # Move to next chunk
                current_end = current_start
                max_days -= days_to_fetch
                
                # Add small delay to avoid rate limits
                time.sleep(0.5)
        else:
            # For intraday data, fetch what's available
            period = period_map.get(granularity, '7d')
            try:
                df = source.fetch(symbol, period)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch intraday data: {e}")
        
        # Combine all chunks
        if all_data:
            combined_df = pd.concat(all_data, axis=0)
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.sort_index(inplace=True)
            
            metadata = {
                'start_date': combined_df.index.min(),
                'end_date': combined_df.index.max(),
                'records': len(combined_df)
            }
            
            return combined_df, metadata
        
        return pd.DataFrame(), {'start_date': None, 'end_date': None, 'records': 0}
    
    def _get_max_days_for_source(self, source_name: str, granularity: str) -> int:
        """Get maximum days of data available for a source and granularity"""
        config = self.source_configs.get(source_name, {})
        granularity_map = config.get('granularity', {})
        
        # Map common granularities
        granularity_key = granularity
        if granularity in ['1m', '5m', '15m', '30m']:
            granularity_key = '5m'
        elif granularity in ['1h', '2h', '4h', '6h', '12h']:
            granularity_key = '1h'
        
        return granularity_map.get(granularity_key, 0)
    
    def _combine_and_deduplicate(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple dataframes and handle conflicts
        
        Priority order: binance > cryptocompare > coingecko > others
        """
        if not dataframes:
            return pd.DataFrame()
        
        # Concatenate all dataframes
        combined = pd.concat(dataframes, axis=0)
        
        # Sort by timestamp and source priority
        source_priority = {
            'binance': 1,
            'cryptocompare': 2,
            'coingecko': 3,
            'alphavantage': 4
        }
        
        combined['priority'] = combined['source'].map(
            lambda x: source_priority.get(x, 99)
        )
        
        # Sort by timestamp and priority
        combined.sort_values(['priority'], inplace=True)
        combined.sort_index(inplace=True)
        
        # Keep the highest priority source for each timestamp
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Drop helper columns
        combined.drop(['priority', 'source'], axis=1, errors='ignore', inplace=True)
        
        return combined
    
    def _store_historical_data(self, df: pd.DataFrame, symbol: str, granularity: str):
        """Store historical data in the database"""
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare data for insertion
        records = []
        for timestamp, row in df.iterrows():
            source = row.get('source', 'combined')
            record = (
                symbol,
                source,
                timestamp,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                float(row.get('Volume', 0)),
                granularity
            )
            records.append(record)
        
        # Insert or replace records
        cursor.executemany('''
            INSERT OR REPLACE INTO ohlcv_data 
            (symbol, source, timestamp, open, high, low, close, volume, granularity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {len(records)} historical records for {symbol}")
    
    def _log_collection(self, symbol: str, source: str, start_date: datetime,
                       end_date: datetime, granularity: str, records: int,
                       collection_time: float, status: str, error_msg: str = None):
        """Log data collection activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_collection_log
            (symbol, source, start_date, end_date, granularity, 
             records_collected, collection_time, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, source, start_date, end_date, granularity,
              records, collection_time, status, error_msg))
        
        conn.commit()
        conn.close()
    
    def _update_continuity_tracking(self, symbol: str, granularity: str, df: pd.DataFrame):
        """Update data continuity tracking"""
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate missing periods
        missing_periods = self._find_missing_periods(df, granularity)
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_continuity
            (symbol, granularity, earliest_date, latest_date, 
             total_records, missing_periods, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, granularity, df.index.min(), df.index.max(),
              len(df), json.dumps(missing_periods), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def _find_missing_periods(self, df: pd.DataFrame, granularity: str) -> List[Dict]:
        """Find gaps in the data"""
        if df.empty or len(df) < 2:
            return []
        
        # Expected frequency based on granularity
        freq_map = {
            '1m': 'T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': 'H',
            '4h': '4H',
            '1d': 'D'
        }
        
        freq = freq_map.get(granularity, 'D')
        
        # Create expected date range
        expected_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Find missing dates
        missing_dates = expected_range.difference(df.index)
        
        if len(missing_dates) == 0:
            return []
        
        # Group consecutive missing dates
        missing_periods = []
        current_start = None
        current_end = None
        
        for i, date in enumerate(missing_dates):
            if current_start is None:
                current_start = date
                current_end = date
            elif (date - current_end).total_seconds() <= pd.Timedelta(freq).total_seconds() * 1.5:
                current_end = date
            else:
                missing_periods.append({
                    'start': current_start.isoformat(),
                    'end': current_end.isoformat(),
                    'count': len(pd.date_range(current_start, current_end, freq=freq))
                })
                current_start = date
                current_end = date
        
        # Add last period
        if current_start is not None:
            missing_periods.append({
                'start': current_start.isoformat(),
                'end': current_end.isoformat(),
                'count': len(pd.date_range(current_start, current_end, freq=freq))
            })
        
        return missing_periods
    
    def load_historical_data(self, symbol: str, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None, 
                           granularity: str = '1d',
                           combine_with_cache: bool = True,
                           source: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data from database, optionally combining with cache
        
        Args:
            symbol: Symbol to load
            start_date: Start date (None for all available)
            end_date: End date (None for latest)
            granularity: Data granularity
            combine_with_cache: Whether to combine with cached recent data
            source: Optional specific data source to filter by
            
        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND granularity = ?
        '''
        params = [symbol, granularity]
        
        if source:
            query += ' AND source = ?'
            params.append(source)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp'
        
        # Load data
        df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'], index_col='timestamp')
        conn.close()
        
        # Rename columns to match expected format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if combine_with_cache and not df.empty:
            # Get recent data from cache to fill any gaps
            try:
                from .cache_service import get_cache_service
                cache = get_cache_service()
                
                # Check for recent cached data
                latest_db_date = df.index.max()
                if (datetime.now() - latest_db_date).days < 7:
                    # Try to get recent data from cache
                    cache_key = f"crypto_{symbol}_{granularity}_recent"
                    cached_data = cache.get(cache_key)
                    
                    if cached_data is not None and isinstance(cached_data, pd.DataFrame):
                        # Combine with historical data
                        cached_data = cached_data[cached_data.index > latest_db_date]
                        if not cached_data.empty:
                            df = pd.concat([df, cached_data], axis=0)
                            df = df[~df.index.duplicated(keep='last')]
                            df.sort_index(inplace=True)
            
            except Exception as e:
                logger.warning(f"Failed to combine with cache: {e}")
        
        return df
    
    def get_data_availability(self, symbol: str) -> Dict[str, Any]:
        """Get information about available historical data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get summary by granularity
        cursor.execute('''
            SELECT granularity, 
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest,
                   COUNT(*) as records,
                   COUNT(DISTINCT DATE(timestamp)) as days
            FROM ohlcv_data
            WHERE symbol = ?
            GROUP BY granularity
        ''', (symbol,))
        
        availability = {}
        for row in cursor.fetchall():
            availability[row[0]] = {
                'earliest': row[1],
                'latest': row[2],
                'total_records': row[3],
                'total_days': row[4]
            }
        
        # Get continuity info
        cursor.execute('''
            SELECT granularity, missing_periods
            FROM data_continuity
            WHERE symbol = ?
        ''', (symbol,))
        
        for row in cursor.fetchall():
            if row[0] in availability:
                availability[row[0]]['missing_periods'] = json.loads(row[1])
        
        conn.close()
        
        return availability
    
    def fill_missing_data(self, symbol: str, sources: List[Any], 
                         granularity: str = '1d') -> int:
        """
        Attempt to fill missing data periods from available sources
        
        Returns:
            Number of records added
        """
        # Get current data continuity
        availability = self.get_data_availability(symbol)
        gran_info = availability.get(granularity, {})
        missing_periods = gran_info.get('missing_periods', [])
        
        if not missing_periods:
            logger.info(f"No missing periods for {symbol} {granularity}")
            return 0
        
        total_added = 0
        
        for period in missing_periods:
            start = datetime.fromisoformat(period['start'])
            end = datetime.fromisoformat(period['end'])
            
            logger.info(f"Attempting to fill gap: {start} to {end}")
            
            # Try to fetch data for this specific period
            for source in sources:
                try:
                    # Calculate period in days
                    days = (end - start).days + 1
                    
                    # Fetch data
                    df = source.fetch(symbol, f"{days}d")
                    
                    if df is not None and not df.empty:
                        # Filter to our period
                        mask = (df.index >= start) & (df.index <= end)
                        df_filtered = df[mask]
                        
                        if not df_filtered.empty:
                            # Store the data
                            df_filtered['source'] = getattr(source, 'name', 'unknown')
                            self._store_historical_data(df_filtered, symbol, granularity)
                            total_added += len(df_filtered)
                            
                            logger.info(f"Filled {len(df_filtered)} records from {source.name}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to fill from {source}: {e}")
                    continue
        
        # Update continuity tracking
        if total_added > 0:
            df = self.load_historical_data(symbol, granularity=granularity, 
                                         combine_with_cache=False)
            self._update_continuity_tracking(symbol, granularity, df)
        
        return total_added
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about data collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('''
            SELECT COUNT(DISTINCT symbol) as symbols,
                   COUNT(DISTINCT source) as sources,
                   COUNT(*) as total_records,
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest
            FROM ohlcv_data
        ''')
        
        overall = cursor.fetchone()
        
        # Collection log stats
        cursor.execute('''
            SELECT source,
                   COUNT(*) as collections,
                   SUM(records_collected) as total_records,
                   AVG(collection_time) as avg_time,
                   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures
            FROM data_collection_log
            GROUP BY source
        ''')
        
        by_source = {}
        for row in cursor.fetchall():
            by_source[row[0]] = {
                'collections': row[1],
                'total_records': row[2] or 0,
                'avg_time': round(row[3] or 0, 2),
                'success_rate': round(row[4] / row[1] * 100, 2) if row[1] > 0 else 0
            }
        
        conn.close()
        
        return {
            'overall': {
                'symbols': overall[0],
                'sources': overall[1],
                'total_records': overall[2],
                'earliest_data': overall[3],
                'latest_data': overall[4]
            },
            'by_source': by_source
        }
    
    def get_data_gaps(self, symbol: str, granularity: str) -> List[Tuple[datetime, datetime]]:
        """
        Get gaps in historical data for a symbol and granularity
        
        Args:
            symbol: Symbol to check (e.g., 'BTC')
            granularity: Data granularity ('daily', '1d', 'hourly', '1h', etc.)
            
        Returns:
            List of tuples containing (gap_start, gap_end) datetimes
        """
        try:
            # Normalize granularity
            granularity_map = {
                'daily': '1d',
                'hourly': '1h',
                'minute': '1m',
                '5min': '5m',
                '15min': '15m',
                '30min': '30m',
                '4hour': '4h',
                '1day': '1d',
                '1hour': '1h',
                '1minute': '1m'
            }
            normalized_gran = granularity_map.get(granularity, granularity)
            
            # Check if any data exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First check if we have any data at all
            cursor.execute('''
                SELECT COUNT(*) FROM ohlcv_data
                WHERE symbol = ? AND granularity = ?
            ''', (symbol, normalized_gran))
            
            count = cursor.fetchone()[0]
            if count == 0:
                conn.close()
                logger.info(f"No data exists for {symbol} with granularity {normalized_gran}")
                return []
            
            # Get the actual data as a DataFrame to use _find_missing_periods
            conn.close()
            
            # Load the data
            df = self.load_historical_data(symbol, granularity=normalized_gran, combine_with_cache=False)
            
            if df.empty or len(df) < 2:
                return []
            
            # Use the existing _find_missing_periods method
            missing_periods = self._find_missing_periods(df, normalized_gran)
            
            # Convert the missing periods to list of tuples
            gaps = []
            for period in missing_periods:
                start = pd.to_datetime(period['start'])
                end = pd.to_datetime(period['end'])
                gaps.append((start, end))
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding data gaps: {e}")
            return []
    
    def get_data_quality_by_source(self, symbol: str, source: str) -> Dict[str, Any]:
        """
        Get data quality metrics for a specific source and symbol
        
        Args:
            symbol: Symbol to check (e.g., 'BTC')
            source: Data source name
            
        Returns:
            Dictionary with quality metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic metrics
        cursor.execute('''
            SELECT COUNT(*) as total_records,
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest,
                   COUNT(DISTINCT DATE(timestamp)) as unique_days,
                   COUNT(DISTINCT granularity) as granularities
            FROM ohlcv_data
            WHERE symbol = ? AND source = ?
        ''', (symbol, source))
        
        result = cursor.fetchone()
        
        if result is None or result[0] == 0:
            conn.close()
            return {
                'source': source,
                'symbol': symbol,
                'total_records': 0,
                'date_range': {'start': None, 'end': None},
                'unique_days': 0,
                'granularities': [],
                'completeness': 0.0,
                'quality_score': 0.0
            }
        
        total_records = result[0]
        earliest = result[1]
        latest = result[2]
        unique_days = result[3]
        
        # Get granularities
        cursor.execute('''
            SELECT DISTINCT granularity, COUNT(*) as count
            FROM ohlcv_data
            WHERE symbol = ? AND source = ?
            GROUP BY granularity
        ''', (symbol, source))
        
        granularities = {}
        for row in cursor.fetchall():
            granularities[row[0]] = row[1]
        
        # Check for null values (data quality issues)
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as null_open,
                SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) as null_high,
                SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) as null_low,
                SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
                SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
            FROM ohlcv_data
            WHERE symbol = ? AND source = ?
        ''', (symbol, source))
        
        null_counts = cursor.fetchone()
        total_nulls = sum(null_counts[i] for i in range(4))  # Exclude volume as it can be null
        
        # Check for data anomalies (negative prices, extreme values)
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 THEN 1 ELSE 0 END) as negative_prices,
                SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as high_low_errors,
                SUM(CASE WHEN (high - low) / low > 0.5 THEN 1 ELSE 0 END) as extreme_ranges
            FROM ohlcv_data
            WHERE symbol = ? AND source = ?
        ''', (symbol, source))
        
        anomalies = cursor.fetchone()
        total_anomalies = sum(anomalies)
        
        # Get collection success rate
        cursor.execute('''
            SELECT 
                COUNT(*) as attempts,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes
            FROM data_collection_log
            WHERE symbol = ? AND source = ?
        ''', (symbol, source))
        
        collection_stats = cursor.fetchone()
        success_rate = (collection_stats[1] / collection_stats[0] * 100) if collection_stats[0] > 0 else 0
        
        conn.close()
        
        # Calculate completeness
        if earliest and latest:
            earliest_dt = datetime.fromisoformat(earliest.replace('Z', '+00:00')) if isinstance(earliest, str) else earliest
            latest_dt = datetime.fromisoformat(latest.replace('Z', '+00:00')) if isinstance(latest, str) else latest
            expected_days = (latest_dt - earliest_dt).days + 1
            completeness = (unique_days / expected_days * 100) if expected_days > 0 else 0
        else:
            completeness = 0
        
        # Calculate quality score (0-100)
        quality_score = 100.0
        quality_score -= (total_nulls / (total_records * 4) * 20) if total_records > 0 else 20  # Up to -20 for nulls
        quality_score -= (total_anomalies / total_records * 30) if total_records > 0 else 30  # Up to -30 for anomalies
        quality_score -= (100 - completeness) * 0.3  # Up to -30 for incompleteness
        quality_score -= (100 - success_rate) * 0.2  # Up to -20 for collection failures
        quality_score = max(0, quality_score)
        
        return {
            'source': source,
            'symbol': symbol,
            'total_records': total_records,
            'date_range': {
                'start': earliest,
                'end': latest
            },
            'unique_days': unique_days,
            'granularities': granularities,
            'completeness': round(completeness, 2),
            'quality_issues': {
                'null_values': total_nulls,
                'negative_prices': anomalies[0],
                'high_low_errors': anomalies[1],
                'extreme_ranges': anomalies[2]
            },
            'collection_success_rate': round(success_rate, 2),
            'quality_score': round(quality_score, 2)
        }
    
    def get_historical_data(self, symbol: str, frequency: str, 
                          source: Optional[str] = None,
                          start_date: Optional[Union[datetime, str]] = None,
                          end_date: Optional[Union[datetime, str]] = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for the specified symbol and parameters
        
        Args:
            symbol: Symbol to retrieve (e.g., 'BTC', 'bitcoin')
            frequency: Data frequency/granularity ('daily', 'hourly', '1d', '1h', etc.)
            source: Optional specific data source to use
            start_date: Optional start date (datetime, date, or string)
            end_date: Optional end date (datetime, date, or string)
            
        Returns:
            DataFrame with OHLCV data or None if no data found
        """
        try:
            from datetime import date
            
            # Map frequency names to granularity codes
            frequency_map = {
                'daily': '1d',
                'hourly': '1h',
                'minute': '1m',
                '5min': '5m',
                '15min': '15m',
                '30min': '30m',
                '4hour': '4h',
                '1day': '1d',
                '1hour': '1h',
                '1minute': '1m'
            }
            
            # Normalize frequency
            granularity = frequency_map.get(frequency, frequency)
            
            # Convert dates if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            elif isinstance(start_date, date) and not isinstance(start_date, datetime):
                # Convert date to datetime
                start_date = datetime.combine(start_date, datetime.min.time())
                
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            elif isinstance(end_date, date) and not isinstance(end_date, datetime):
                # Convert date to datetime (end of day)
                end_date = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
            
            # Use the load_historical_data method for consistency
            df = self.load_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                granularity=granularity,
                source=source,
                combine_with_cache=True
            )
            
            if df.empty:
                logger.warning(f"No historical data found for {symbol} with frequency {frequency}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return None
    
    def generate_upload_id(self, filename: str) -> str:
        """Generate a unique upload ID based on filename and timestamp"""
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def bulk_insert_ohlcv(self, 
                         data: Union[pd.DataFrame, List[Dict]], 
                         symbol: str,
                         granularity: str,
                         filename: str,
                         source: str = 'upload',
                         conflict_resolution: ConflictResolution = ConflictResolution.MERGE,
                         chunk_size: int = 1000,
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Bulk insert OHLCV data with conflict resolution and progress tracking
        
        Args:
            data: DataFrame or list of dicts with OHLCV data
            symbol: Symbol for the data
            granularity: Data granularity (1d, 1h, etc.)
            filename: Original filename for tracking
            source: Data source identifier
            conflict_resolution: How to handle conflicts
            chunk_size: Number of rows to insert per batch
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Dict with upload summary including upload_id, rows inserted, skipped, etc.
        """
        upload_id = self.generate_upload_id(filename)
        start_time = time.time()
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        total_rows = len(df)
        inserted_rows = 0
        skipped_rows = 0
        error_rows = 0
        
        # Track upload start
        self._track_upload_start(upload_id, filename, 'ohlcv', symbol, granularity, total_rows, conflict_resolution)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # Normalize column names
            df.columns = df.columns.str.lower()
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            
            # Validate required columns
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif df.index.name == 'timestamp' or pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index(names=['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Get date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            # Handle conflicts based on strategy
            if conflict_resolution == ConflictResolution.REPLACE:
                # Delete existing data in the date range
                conn.execute('''
                    DELETE FROM ohlcv_data 
                    WHERE symbol = ? AND granularity = ? 
                    AND timestamp >= ? AND timestamp <= ?
                ''', (symbol, granularity, start_date, end_date))
            
            elif conflict_resolution == ConflictResolution.FILL_GAPS:
                # Get existing timestamps
                existing_df = pd.read_sql_query('''
                    SELECT timestamp FROM ohlcv_data
                    WHERE symbol = ? AND granularity = ?
                    AND timestamp >= ? AND timestamp <= ?
                ''', conn, params=(symbol, granularity, start_date, end_date))
                
                if not existing_df.empty:
                    existing_timestamps = pd.to_datetime(existing_df['timestamp'])
                    df = df[~df['timestamp'].isin(existing_timestamps)]
                    skipped_rows = total_rows - len(df)
            
            # Process in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                
                if conflict_resolution == ConflictResolution.AVERAGE:
                    # Special handling for averaging
                    inserted_chunk = self._insert_with_averaging(conn, chunk, symbol, granularity, source, upload_id)
                    inserted_rows += inserted_chunk
                else:
                    # Prepare records for insertion
                    records = []
                    for _, row in chunk.iterrows():
                        try:
                            record = (
                                symbol,
                                source,
                                row['timestamp'],
                                float(row['open']),
                                float(row['high']),
                                float(row['low']),
                                float(row['close']),
                                float(row.get('volume', 0)),
                                granularity,
                                upload_id
                            )
                            records.append(record)
                        except Exception as e:
                            logger.warning(f"Error processing row: {e}")
                            error_rows += 1
                            continue
                    
                    # Bulk insert
                    if records:
                        if conflict_resolution == ConflictResolution.MERGE:
                            # Use INSERT OR REPLACE for merge
                            conn.executemany('''
                                INSERT OR REPLACE INTO ohlcv_data 
                                (symbol, source, timestamp, open, high, low, close, volume, granularity, upload_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', records)
                        else:
                            # Normal insert
                            conn.executemany('''
                                INSERT INTO ohlcv_data 
                                (symbol, source, timestamp, open, high, low, close, volume, granularity, upload_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', records)
                        
                        inserted_rows += len(records)
                
                # Progress callback
                if progress_callback:
                    progress_callback(min(i + chunk_size, len(df)), total_rows)
            
            # Commit transaction
            conn.commit()
            
            # Update data continuity
            self._update_continuity_after_upload(symbol, granularity)
            
            # Track successful upload
            processing_time = time.time() - start_time
            self._track_upload_complete(
                upload_id, 'completed', inserted_rows, skipped_rows, 
                error_rows, processing_time, start_date, end_date
            )
            
            return {
                'upload_id': upload_id,
                'status': 'success',
                'total_rows': total_rows,
                'inserted_rows': inserted_rows,
                'skipped_rows': skipped_rows,
                'error_rows': error_rows,
                'processing_time': round(processing_time, 2),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
        except Exception as e:
            # Rollback transaction
            conn.rollback()
            logger.error(f"Bulk insert failed: {e}")
            
            # Track failed upload
            processing_time = time.time() - start_time
            self._track_upload_complete(
                upload_id, 'failed', inserted_rows, skipped_rows, 
                error_rows, processing_time, error_message=str(e)
            )
            
            raise
            
        finally:
            conn.close()
    
    def bulk_insert_onchain(self,
                           data: Union[pd.DataFrame, List[Dict]],
                           symbol: str,
                           filename: str,
                           source: str = 'upload',
                           conflict_resolution: ConflictResolution = ConflictResolution.MERGE,
                           chunk_size: int = 1000,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Bulk insert on-chain data with conflict resolution
        
        Args:
            data: DataFrame or list of dicts with on-chain metrics
            symbol: Symbol for the data
            filename: Original filename for tracking
            source: Data source identifier
            conflict_resolution: How to handle conflicts
            chunk_size: Number of rows to insert per batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with upload summary
        """
        upload_id = self.generate_upload_id(filename)
        start_time = time.time()
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        total_rows = len(df)
        inserted_rows = 0
        skipped_rows = 0
        error_rows = 0
        
        # Track upload start
        self._track_upload_start(upload_id, filename, 'onchain', symbol, None, total_rows, conflict_resolution)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # Normalize column names
            df.columns = df.columns.str.lower()
            
            # Ensure timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("No timestamp/date column found")
            
            # Get date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            # Process each metric column (all non-timestamp columns)
            metric_columns = [col for col in df.columns if col not in ['timestamp', 'date']]
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                records = []
                
                for _, row in chunk.iterrows():
                    timestamp = row['timestamp']
                    
                    for metric in metric_columns:
                        value = row.get(metric)
                        if pd.notna(value):
                            try:
                                record = (
                                    symbol,
                                    timestamp,
                                    metric,
                                    float(value),
                                    source,
                                    upload_id
                                )
                                records.append(record)
                            except Exception as e:
                                logger.warning(f"Error processing metric {metric}: {e}")
                                error_rows += 1
                
                # Bulk insert
                if records:
                    if conflict_resolution == ConflictResolution.REPLACE:
                        # Delete existing and insert
                        timestamps = [r[1] for r in records]
                        metrics = list(set(r[2] for r in records))
                        
                        placeholders = ','.join(['?'] * len(metrics))
                        conn.execute(f'''
                            DELETE FROM onchain_data 
                            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                            AND metric IN ({placeholders})
                        ''', [symbol, min(timestamps), max(timestamps)] + metrics)
                        
                        conn.executemany('''
                            INSERT INTO onchain_data 
                            (symbol, timestamp, metric, value, source, upload_id)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', records)
                        
                    elif conflict_resolution == ConflictResolution.MERGE:
                        conn.executemany('''
                            INSERT OR REPLACE INTO onchain_data 
                            (symbol, timestamp, metric, value, source, upload_id)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', records)
                        
                    elif conflict_resolution == ConflictResolution.FILL_GAPS:
                        # Only insert non-existing
                        conn.executemany('''
                            INSERT OR IGNORE INTO onchain_data 
                            (symbol, timestamp, metric, value, source, upload_id)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', records)
                    
                    inserted_rows += len(records)
                
                # Progress callback
                if progress_callback:
                    progress_callback(min(i + chunk_size, len(df)), total_rows)
            
            # Commit transaction
            conn.commit()
            
            # Track successful upload
            processing_time = time.time() - start_time
            self._track_upload_complete(
                upload_id, 'completed', inserted_rows, skipped_rows, 
                error_rows, processing_time, start_date, end_date
            )
            
            return {
                'upload_id': upload_id,
                'status': 'success',
                'total_rows': total_rows * len(metric_columns),  # Total possible metric values
                'inserted_rows': inserted_rows,
                'skipped_rows': skipped_rows,
                'error_rows': error_rows,
                'processing_time': round(processing_time, 2),
                'metrics': metric_columns,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
        except Exception as e:
            # Rollback transaction
            conn.rollback()
            logger.error(f"Bulk insert onchain failed: {e}")
            
            # Track failed upload
            processing_time = time.time() - start_time
            self._track_upload_complete(
                upload_id, 'failed', inserted_rows, skipped_rows, 
                error_rows, processing_time, error_message=str(e)
            )
            
            raise
            
        finally:
            conn.close()
    
    def bulk_insert_sentiment(self,
                            data: Union[pd.DataFrame, List[Dict]],
                            symbol: str,
                            filename: str,
                            source: str = 'upload',
                            conflict_resolution: ConflictResolution = ConflictResolution.MERGE,
                            chunk_size: int = 1000,
                            progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Bulk insert sentiment data with conflict resolution
        
        Args:
            data: DataFrame or list of dicts with sentiment data
            symbol: Symbol for the data
            filename: Original filename for tracking
            source: Data source identifier
            conflict_resolution: How to handle conflicts
            chunk_size: Number of rows to insert per batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with upload summary
        """
        upload_id = self.generate_upload_id(filename)
        start_time = time.time()
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        total_rows = len(df)
        inserted_rows = 0
        skipped_rows = 0
        error_rows = 0
        
        # Track upload start
        self._track_upload_start(upload_id, filename, 'sentiment', symbol, None, total_rows, conflict_resolution)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # Normalize column names
            df.columns = df.columns.str.lower()
            
            # Ensure timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("No timestamp/date column found")
            
            # Get date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            # Handle conflicts
            if conflict_resolution == ConflictResolution.REPLACE:
                conn.execute('''
                    DELETE FROM sentiment_data 
                    WHERE symbol = ? AND source = ?
                    AND timestamp >= ? AND timestamp <= ?
                ''', (symbol, source, start_date, end_date))
            
            # Process in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                records = []
                
                for _, row in chunk.iterrows():
                    try:
                        # Extract sentiment data
                        sentiment_score = None
                        sentiment_label = None
                        volume = None
                        
                        # Try different column names
                        for score_col in ['sentiment_score', 'score', 'sentiment']:
                            if score_col in row and pd.notna(row[score_col]):
                                sentiment_score = float(row[score_col])
                                break
                        
                        for label_col in ['sentiment_label', 'label', 'sentiment_type']:
                            if label_col in row and pd.notna(row[label_col]):
                                sentiment_label = str(row[label_col])
                                break
                        
                        for vol_col in ['volume', 'tweet_volume', 'mention_volume']:
                            if vol_col in row and pd.notna(row[vol_col]):
                                volume = int(row[vol_col])
                                break
                        
                        # Need at least score or label
                        if sentiment_score is not None or sentiment_label is not None:
                            record = (
                                symbol,
                                row['timestamp'],
                                source,
                                sentiment_score,
                                sentiment_label,
                                volume,
                                upload_id
                            )
                            records.append(record)
                        else:
                            error_rows += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing sentiment row: {e}")
                        error_rows += 1
                
                # Bulk insert
                if records:
                    if conflict_resolution == ConflictResolution.MERGE:
                        conn.executemany('''
                            INSERT OR REPLACE INTO sentiment_data 
                            (symbol, timestamp, source, sentiment_score, sentiment_label, volume, upload_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', records)
                    elif conflict_resolution == ConflictResolution.FILL_GAPS:
                        conn.executemany('''
                            INSERT OR IGNORE INTO sentiment_data 
                            (symbol, timestamp, source, sentiment_score, sentiment_label, volume, upload_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', records)
                    else:
                        conn.executemany('''
                            INSERT INTO sentiment_data 
                            (symbol, timestamp, source, sentiment_score, sentiment_label, volume, upload_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', records)
                    
                    inserted_rows += len(records)
                
                # Progress callback
                if progress_callback:
                    progress_callback(min(i + chunk_size, len(df)), total_rows)
            
            # Commit transaction
            conn.commit()
            
            # Track successful upload
            processing_time = time.time() - start_time
            self._track_upload_complete(
                upload_id, 'completed', inserted_rows, skipped_rows, 
                error_rows, processing_time, start_date, end_date
            )
            
            return {
                'upload_id': upload_id,
                'status': 'success',
                'total_rows': total_rows,
                'inserted_rows': inserted_rows,
                'skipped_rows': skipped_rows,
                'error_rows': error_rows,
                'processing_time': round(processing_time, 2),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
        except Exception as e:
            # Rollback transaction
            conn.rollback()
            logger.error(f"Bulk insert sentiment failed: {e}")
            
            # Track failed upload
            processing_time = time.time() - start_time
            self._track_upload_complete(
                upload_id, 'failed', inserted_rows, skipped_rows, 
                error_rows, processing_time, error_message=str(e)
            )
            
            raise
            
        finally:
            conn.close()
    
    def _insert_with_averaging(self, conn: sqlite3.Connection, df: pd.DataFrame, 
                              symbol: str, granularity: str, source: str, 
                              upload_id: str) -> int:
        """Handle AVERAGE conflict resolution by averaging existing and new values"""
        inserted = 0
        
        for _, row in df.iterrows():
            # Check if record exists
            existing = conn.execute('''
                SELECT open, high, low, close, volume 
                FROM ohlcv_data
                WHERE symbol = ? AND timestamp = ? AND granularity = ?
            ''', (symbol, row['timestamp'], granularity)).fetchone()
            
            if existing:
                # Average the values
                avg_open = (existing[0] + float(row['open'])) / 2
                avg_high = (existing[1] + float(row['high'])) / 2
                avg_low = (existing[2] + float(row['low'])) / 2
                avg_close = (existing[3] + float(row['close'])) / 2
                avg_volume = (existing[4] + float(row.get('volume', 0))) / 2
                
                conn.execute('''
                    UPDATE ohlcv_data 
                    SET open = ?, high = ?, low = ?, close = ?, volume = ?, 
                        source = ?, upload_id = ?
                    WHERE symbol = ? AND timestamp = ? AND granularity = ?
                ''', (avg_open, avg_high, avg_low, avg_close, avg_volume,
                      source, upload_id, symbol, row['timestamp'], granularity))
            else:
                # Insert new record
                conn.execute('''
                    INSERT INTO ohlcv_data 
                    (symbol, source, timestamp, open, high, low, close, volume, granularity, upload_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, source, row['timestamp'], float(row['open']), 
                      float(row['high']), float(row['low']), float(row['close']),
                      float(row.get('volume', 0)), granularity, upload_id))
            
            inserted += 1
        
        return inserted
    
    def _track_upload_start(self, upload_id: str, filename: str, data_type: str,
                           symbol: str, granularity: Optional[str], total_rows: int,
                           conflict_resolution: ConflictResolution):
        """Track the start of an upload"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO upload_tracking 
            (upload_id, filename, status, data_type, symbol, granularity, 
             total_rows, conflict_resolution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (upload_id, filename, 'processing', data_type, symbol, granularity,
              total_rows, conflict_resolution.value))
        conn.commit()
        conn.close()
    
    def _track_upload_complete(self, upload_id: str, status: str, inserted_rows: int,
                              skipped_rows: int, error_rows: int, processing_time: float,
                              start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None,
                              error_message: Optional[str] = None):
        """Update upload tracking with completion status"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            UPDATE upload_tracking 
            SET status = ?, inserted_rows = ?, skipped_rows = ?, error_rows = ?,
                processing_time = ?, start_date = ?, end_date = ?, error_message = ?
            WHERE upload_id = ?
        ''', (status, inserted_rows, skipped_rows, error_rows, processing_time,
              start_date, end_date, error_message, upload_id))
        conn.commit()
        conn.close()
    
    def _update_continuity_after_upload(self, symbol: str, granularity: str):
        """Update data continuity tracking after an upload"""
        # Load all data for this symbol/granularity
        df = self.load_historical_data(symbol, granularity=granularity, combine_with_cache=False)
        if not df.empty:
            self._update_continuity_tracking(symbol, granularity, df)
    
    def get_upload_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent upload history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT upload_id, filename, upload_timestamp, status, data_type,
                   symbol, granularity, total_rows, inserted_rows, skipped_rows,
                   error_rows, processing_time, start_date, end_date, 
                   conflict_resolution, error_message
            FROM upload_tracking
            ORDER BY upload_timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        uploads = []
        for row in cursor.fetchall():
            uploads.append({
                'upload_id': row[0],
                'filename': row[1],
                'timestamp': row[2],
                'status': row[3],
                'data_type': row[4],
                'symbol': row[5],
                'granularity': row[6],
                'total_rows': row[7],
                'inserted_rows': row[8],
                'skipped_rows': row[9],
                'error_rows': row[10],
                'processing_time': row[11],
                'start_date': row[12],
                'end_date': row[13],
                'conflict_resolution': row[14],
                'error_message': row[15]
            })
        
        conn.close()
        return uploads
    
    def rollback_upload(self, upload_id: str) -> Dict[str, Any]:
        """Rollback a specific upload by removing all associated data"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get upload info
            cursor = conn.cursor()
            cursor.execute('''
                SELECT data_type, symbol, granularity 
                FROM upload_tracking 
                WHERE upload_id = ?
            ''', (upload_id,))
            
            upload_info = cursor.fetchone()
            if not upload_info:
                return {'status': 'error', 'message': 'Upload not found'}
            
            data_type, symbol, granularity = upload_info
            
            # Start transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Delete data based on type
            if data_type == 'ohlcv':
                cursor.execute('''
                    DELETE FROM ohlcv_data WHERE upload_id = ?
                ''', (upload_id,))
                deleted_rows = cursor.rowcount
                
            elif data_type == 'onchain':
                cursor.execute('''
                    DELETE FROM onchain_data WHERE upload_id = ?
                ''', (upload_id,))
                deleted_rows = cursor.rowcount
                
            elif data_type == 'sentiment':
                cursor.execute('''
                    DELETE FROM sentiment_data WHERE upload_id = ?
                ''', (upload_id,))
                deleted_rows = cursor.rowcount
            
            # Update upload status
            cursor.execute('''
                UPDATE upload_tracking 
                SET status = 'rolled_back' 
                WHERE upload_id = ?
            ''', (upload_id,))
            
            # Commit transaction
            conn.commit()
            
            # Update continuity tracking if needed
            if data_type == 'ohlcv' and symbol and granularity:
                self._update_continuity_after_upload(symbol, granularity)
            
            return {
                'status': 'success',
                'upload_id': upload_id,
                'deleted_rows': deleted_rows,
                'data_type': data_type
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Rollback failed: {e}")
            return {'status': 'error', 'message': str(e)}
            
        finally:
            conn.close()

# Singleton instance
_historical_manager = None

def get_historical_manager() -> HistoricalDataManager:
    """Get singleton historical data manager instance"""
    global _historical_manager
    if _historical_manager is None:
        _historical_manager = HistoricalDataManager()
    return _historical_manager