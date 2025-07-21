"""
Data upload service for processing CSV and XLSX files
"""
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import sqlite3
import numpy as np

logger = logging.getLogger(__name__)

class DataUploadService:
    """Service for handling data file uploads and processing"""
    
    # Valid data types and their expected columns
    DATA_TYPE_TEMPLATES = {
        "price": {
            "required": ["timestamp", "open", "high", "low", "close", "volume"],
            "optional": ["price"],
            "description": "OHLCV data - requires all standard OHLCV fields"
        },
        "volume": {
            "required": ["timestamp", "volume"],
            "optional": ["price", "trades_count"]
        },
        "onchain": {
            "required": ["timestamp"],
            "optional": ["active_addresses", "transaction_count", "hash_rate", "difficulty", "fees"]
        },
        "sentiment": {
            "required": ["timestamp"],
            "optional": ["sentiment_score", "fear_greed_index", "social_volume", "mentions"]
        },
        "macro": {
            "required": ["timestamp"],
            "optional": ["dxy", "gold", "sp500", "vix", "bond_yield"]
        }
    }
    
    # Suggested sources (custom sources are also allowed)
    SUGGESTED_SOURCES = ["binance", "coingecko", "cryptowatch", "glassnode", "santiment", "custom"]
    
    def __init__(self, db_path: str = "/root/btc/storage/data/historical_data.db"):
        self.db_path = db_path
        self.temp_dir = Path("/tmp/btc_uploads")
        self.temp_dir.mkdir(exist_ok=True)
        
    def get_templates(self) -> Dict[str, Dict]:
        """Get column mapping templates for all data types"""
        return self.DATA_TYPE_TEMPLATES
    
    def get_valid_sources(self) -> Dict[str, Any]:
        """Get information about data sources"""
        return {
            "suggested_sources": self.SUGGESTED_SOURCES,
            "custom_allowed": True,
            "message": "You can use any custom source name. The suggested sources are provided for reference."
        }
    
    def preview_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Preview uploaded file and suggest column mappings"""
        try:
            # Read file based on type
            if file_type == "csv":
                df = pd.read_csv(file_path, nrows=10)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(file_path, nrows=10)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Get column info
            columns = df.columns.tolist()
            sample_data = df.head(10).to_dict('records')
            
            # Suggest mappings based on column names
            suggested_mappings = self._suggest_column_mappings(columns)
            
            return {
                "columns": columns,
                "row_count": len(df),
                "sample_data": sample_data,
                "suggested_mappings": suggested_mappings,
                "data_types": {col: str(df[col].dtype) for col in columns}
            }
            
        except Exception as e:
            logger.error(f"Error previewing file: {e}")
            raise
    
    def process_upload(
        self, 
        file_path: str,
        file_type: str,
        source: str,
        data_type: str,
        symbol: str = "BTC",
        column_mappings: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Process uploaded file and store in database"""
        try:
            # Validate inputs
            # Note: Custom source names are allowed, no validation needed
            
            if data_type not in self.DATA_TYPE_TEMPLATES:
                raise ValueError(f"Invalid data type: {data_type}. Must be one of {list(self.DATA_TYPE_TEMPLATES.keys())}")
            
            # Read full file
            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Apply column mappings if provided
            if column_mappings:
                df = df.rename(columns=column_mappings)
            
            # Validate required columns
            template = self.DATA_TYPE_TEMPLATES[data_type]
            missing_columns = [col for col in template["required"] if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Process and clean data
            df = self._process_data(df, data_type)
            
            # Store in database
            rows_inserted = self._store_in_database(df, source, data_type, symbol)
            
            # Generate summary statistics
            summary = self._generate_summary(df, data_type)
            
            return {
                "success": True,
                "rows_processed": len(df),
                "rows_inserted": rows_inserted,
                "duplicate_rows": len(df) - rows_inserted,
                "data_range": {
                    "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                    "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                },
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def _suggest_column_mappings(self, columns: List[str]) -> Dict[str, str]:
        """Suggest column mappings based on common patterns"""
        mappings = {}
        
        # Common column name patterns
        timestamp_patterns = ["date", "time", "timestamp", "datetime", "ts", "created_at"]
        price_patterns = ["price", "close", "last", "value", "btc_price", "bitcoin_price"]
        volume_patterns = ["volume", "vol", "quantity", "amount", "btc_volume"]
        
        for col in columns:
            col_lower = col.lower()
            
            # Check timestamp patterns
            if any(pattern in col_lower for pattern in timestamp_patterns):
                mappings[col] = "timestamp"
            # Check price patterns
            elif any(pattern in col_lower for pattern in price_patterns):
                mappings[col] = "price"
            # Check volume patterns
            elif any(pattern in col_lower for pattern in volume_patterns):
                mappings[col] = "volume"
            # Check for OHLC data
            elif col_lower in ["open", "high", "low", "close"]:
                mappings[col] = col_lower
            # Check for on-chain metrics
            elif "address" in col_lower:
                mappings[col] = "active_addresses"
            elif "transaction" in col_lower or "tx" in col_lower:
                mappings[col] = "transaction_count"
            elif "hash" in col_lower and "rate" in col_lower:
                mappings[col] = "hash_rate"
            elif "difficulty" in col_lower:
                mappings[col] = "difficulty"
            elif "fee" in col_lower:
                mappings[col] = "fees"
            # Check for sentiment metrics
            elif "sentiment" in col_lower:
                mappings[col] = "sentiment_score"
            elif "fear" in col_lower or "greed" in col_lower:
                mappings[col] = "fear_greed_index"
            elif "social" in col_lower:
                mappings[col] = "social_volume"
            elif "mention" in col_lower:
                mappings[col] = "mentions"
            # Check for macro metrics
            elif "dxy" in col_lower or "dollar" in col_lower:
                mappings[col] = "dxy"
            elif "gold" in col_lower:
                mappings[col] = "gold"
            elif "sp500" in col_lower or "s&p" in col_lower:
                mappings[col] = "sp500"
            elif "vix" in col_lower:
                mappings[col] = "vix"
            elif "bond" in col_lower or "yield" in col_lower:
                mappings[col] = "bond_yield"
                
        return mappings
    
    def _process_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process and clean data based on type"""
        # Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates based on timestamp
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Handle missing values based on data type
        if data_type == "price":
            # Forward fill price data
            for col in ['price', 'open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].ffill()
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col != 'timestamp':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Remove rows with all NaN values (except timestamp)
        non_timestamp_cols = [col for col in df.columns if col != 'timestamp']
        df = df.dropna(subset=non_timestamp_cols, how='all')
        
        return df
    
    def _store_in_database(self, df: pd.DataFrame, source: str, data_type: str, symbol: str) -> int:
        """Store processed data in database"""
        # Fix the database path for Docker environment
        db_path = self.db_path
        if db_path.startswith('/root/btc/'):
            db_path = db_path.replace('/root/btc/', '/app/')
        
        logger.info(f"Storing {len(df)} rows to database at {db_path}")
        logger.info(f"Data type: {data_type}, Source: {source}, Symbol: {symbol}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        rows_inserted = 0
        
        # For price/OHLCV data, store in the proper ohlcv_data table
        if data_type == "price":
            # Don't create table - use existing schema
            # The table already exists with granularity column
            
            # Determine granularity from data frequency
            granularity = self._infer_granularity(df)
            
            # Insert OHLCV data
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ohlcv_data 
                        (symbol, timestamp, open, high, low, close, volume, source, granularity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume']) if pd.notna(row['volume']) else 0.0,
                        source,
                        granularity
                    ))
                    rows_inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting OHLCV row: {e}")
                    logger.error(f"Row data: timestamp={row['timestamp']}, open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}, volume={row.get('volume', 0)}")
        
        else:
            # For other data types, use the generic historical_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    timestamp TEXT,
                    symbol TEXT,
                    source TEXT,
                    data_type TEXT,
                    data JSON,
                    created_at TEXT,
                    PRIMARY KEY (timestamp, symbol, source, data_type)
                )
            """)
            
            for _, row in df.iterrows():
                # Prepare data dict
                data = row.to_dict()
                if 'timestamp' in data:
                    timestamp = data.pop('timestamp')
                    if pd.isna(timestamp):
                        continue
                    timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                else:
                    continue
                
                # Convert NaN values to None for JSON
                data = {k: (None if pd.isna(v) else v) for k, v in data.items()}
                
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO historical_data 
                        (timestamp, symbol, source, data_type, data, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp_str,
                        symbol,
                        source,
                        data_type,
                        json.dumps(data),
                        datetime.now().isoformat()
                    ))
                    rows_inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate entry, skip
                    pass
                except Exception as e:
                    logger.error(f"Error inserting row: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database operation complete. Rows inserted: {rows_inserted}")
        return rows_inserted
    
    def _infer_granularity(self, df: pd.DataFrame) -> str:
        """Infer time granularity from dataframe timestamps"""
        if 'timestamp' not in df.columns or len(df) < 2:
            return '1h'  # Default to hourly
        
        # Calculate time differences
        sorted_df = df.sort_values('timestamp')
        time_diffs = sorted_df['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return '1h'
        
        # Get median time difference in seconds
        median_diff = time_diffs.median().total_seconds()
        
        # Map to standard granularities
        if median_diff < 300:  # < 5 minutes
            return '1m'
        elif median_diff < 900:  # < 15 minutes
            return '5m'
        elif median_diff < 1800:  # < 30 minutes
            return '15m'
        elif median_diff < 3600:  # < 1 hour
            return '30m'
        elif median_diff < 7200:  # < 2 hours
            return '1h'
        elif median_diff < 14400:  # < 4 hours
            return '2h'
        elif median_diff < 28800:  # < 8 hours
            return '4h'
        elif median_diff < 86400:  # < 1 day
            return '12h'
        elif median_diff < 604800:  # < 1 week
            return '1d'
        else:
            return '1w'
    
    def _generate_summary(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Generate summary statistics for uploaded data"""
        summary = {
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "data_type": data_type
        }
        
        # Add type-specific summaries
        if data_type == "price":
            # For OHLCV data, use close price for stats
            if 'close' in df.columns:
                summary["price_stats"] = {
                    "min": float(df['close'].min()),
                    "max": float(df['close'].max()),
                    "mean": float(df['close'].mean()),
                    "std": float(df['close'].std())
                }
            # Add OHLCV-specific stats
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                summary["ohlcv_stats"] = {
                    "high_max": float(df['high'].max()),
                    "low_min": float(df['low'].min()),
                    "avg_range": float((df['high'] - df['low']).mean())
                }
        
        if 'volume' in df.columns:
            summary["volume_stats"] = {
                "total": float(df['volume'].sum()),
                "mean": float(df['volume'].mean())
            }
        
        return summary