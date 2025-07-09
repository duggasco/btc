"""
Cache Integration Module
Provides seamless integration between data fetchers and the cache service
Maintains backward compatibility while adding persistent caching
"""
import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .cache_service import get_cache_service, CacheService

logger = logging.getLogger(__name__)

# Cache TTL configurations based on data volatility
CACHE_TTL_CONFIG = {
    # Real-time data - very short TTL
    'real_time_price': 30,        # 30 seconds
    'order_book': 10,             # 10 seconds
    
    # Market data - short to medium TTL
    'ohlcv_1m': 30,               # 30 seconds for 1-minute candles
    'ohlcv_5m': 60,               # 1 minute for 5-minute candles
    'ohlcv_15m': 180,             # 3 minutes for 15-minute candles
    'ohlcv_1h': 300,              # 5 minutes for hourly candles
    'ohlcv_4h': 600,              # 10 minutes for 4-hour candles
    'ohlcv_1d': 900,              # 15 minutes for daily candles
    'ohlcv_default': 300,         # 5 minutes default
    
    # Indicators and analytics - medium TTL
    'technical_indicators': 300,   # 5 minutes
    'sentiment': 1800,            # 30 minutes
    'fear_greed': 3600,           # 1 hour
    'news': 900,                  # 15 minutes
    
    # On-chain data - medium TTL
    'onchain_metrics': 1800,      # 30 minutes
    'network_stats': 1800,        # 30 minutes
    'blockchain_data': 3600,      # 1 hour
    
    # Macro data - long TTL
    'macro_indicators': 3600,     # 1 hour
    'economic_data': 7200,        # 2 hours
    'forex_rates': 300,           # 5 minutes
    'commodity_prices': 600,      # 10 minutes
    
    # Default fallback
    'default': 300                # 5 minutes
}

def generate_cache_key(source: str, method: str, *args, **kwargs) -> str:
    """
    Generate a unique cache key based on source, method, and parameters
    
    Args:
        source: Data source name (e.g., 'binance', 'coingecko')
        method: Method name (e.g., 'fetch', 'get_current_price')
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Unique cache key string
    """
    # Build key components
    key_parts = [source.lower(), method.lower()]
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            key_parts.append('_'.join(str(x) for x in arg))
        elif isinstance(arg, dict):
            # Sort dict keys for consistency
            sorted_items = sorted(arg.items())
            key_parts.append('_'.join(f"{k}={v}" for k, v in sorted_items))
        else:
            # For complex objects, use type name
            key_parts.append(type(arg).__name__)
    
    # Add keyword arguments (sorted for consistency)
    for key, value in sorted(kwargs.items()):
        if key.startswith('_'):  # Skip private parameters
            continue
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}={value}")
        elif value is not None:
            key_parts.append(f"{key}={type(value).__name__}")
    
    # Create hash for very long keys
    key_string = ':'.join(key_parts)
    if len(key_string) > 200:
        # Use hash for long keys to avoid database issues
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{source}:{method}:{key_hash}"
    
    return key_string

def determine_ttl(data_type: str, period: Optional[str] = None) -> int:
    """
    Determine appropriate TTL based on data type and period
    
    Args:
        data_type: Type of data being cached
        period: Optional period parameter (e.g., '1m', '1h', '1d')
        
    Returns:
        TTL in seconds
    """
    # Check for period-specific TTL for OHLCV data
    if data_type == 'ohlcv' and period:
        period_key = f'ohlcv_{period.lower()}'
        if period_key in CACHE_TTL_CONFIG:
            return CACHE_TTL_CONFIG[period_key]
    
    # Check for specific data type TTL
    if data_type in CACHE_TTL_CONFIG:
        return CACHE_TTL_CONFIG[data_type]
    
    # Return default TTL
    return CACHE_TTL_CONFIG['default']

def cached_api_call(
    data_type: str = 'default',
    ttl: Optional[int] = None,
    cache_on_error: bool = True,
    skip_cache: bool = False
):
    """
    Decorator for caching API calls with automatic key generation
    
    Args:
        data_type: Type of data for TTL selection
        ttl: Override TTL in seconds (None to use default)
        cache_on_error: Whether to return cached data on API error
        skip_cache: Force skip cache (for testing or real-time needs)
    
    Usage:
        @cached_api_call(data_type='ohlcv', ttl=300)
        def fetch_data(symbol: str, period: str):
            return api_request(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if cache should be skipped
            if skip_cache or kwargs.get('skip_cache', False):
                kwargs.pop('skip_cache', None)  # Remove from kwargs
                return func(self, *args, **kwargs)
            
            # Generate cache key
            source_name = getattr(self, 'name', self.__class__.__name__)
            cache_key = generate_cache_key(source_name, func.__name__, *args, **kwargs)
            
            # Get cache service
            cache = get_cache_service()
            
            # Try to get from cache
            try:
                cached_data = cache.get(cache_key)
                if cached_data is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                # Continue to fetch fresh data
            
            # Fetch fresh data
            try:
                result = func(self, *args, **kwargs)
                
                # Determine TTL
                actual_ttl = ttl
                if actual_ttl is None:
                    # Extract period from args/kwargs if available
                    period = kwargs.get('period') or (args[1] if len(args) > 1 else None)
                    actual_ttl = determine_ttl(data_type, period)
                
                # Cache the result
                try:
                    cache.set(
                        cache_key,
                        result,
                        data_type=data_type,
                        api_source=source_name,
                        ttl=actual_ttl,
                        metadata={
                            'method': func.__name__,
                            'args': str(args)[:100],  # Truncate for storage
                            'cached_at': datetime.now().isoformat()
                        }
                    )
                    logger.debug(f"Cached result: {cache_key} (TTL: {actual_ttl}s)")
                except Exception as e:
                    logger.error(f"Cache write error: {e}")
                    # Continue without caching
                
                return result
                
            except Exception as api_error:
                logger.error(f"API call failed: {api_error}")
                
                # Try to return stale cached data if available and allowed
                if cache_on_error:
                    try:
                        # Get even expired data in case of API failure
                        conn = cache.db_path
                        import sqlite3
                        with sqlite3.connect(conn) as db:
                            cursor = db.cursor()
                            cursor.execute(
                                "SELECT response_data, response_format FROM api_cache WHERE cache_key = ?",
                                (cache_key,)
                            )
                            row = cursor.fetchone()
                            if row:
                                data = cache._deserialize(row[0], row[1])
                                logger.warning(f"Returning stale cache due to API error: {cache_key}")
                                return data
                    except Exception as e:
                        logger.error(f"Failed to retrieve stale cache: {e}")
                
                # Re-raise the original API error
                raise api_error
        
        return wrapper
    return decorator

class CachedDataFetcher:
    """
    Base class for data fetchers with built-in caching support
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service or get_cache_service()
        self._cache_enabled = True
        
    def enable_cache(self):
        """Enable caching"""
        self._cache_enabled = True
        
    def disable_cache(self):
        """Disable caching temporarily"""
        self._cache_enabled = False
        
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern"""
        source_name = getattr(self, 'name', self.__class__.__name__)
        return self.cache.invalidate(
            pattern=pattern,
            api_source=source_name,
            reason="Manual clear"
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this fetcher"""
        stats = self.cache.get_stats()
        
        # Add fetcher-specific stats
        source_name = getattr(self, 'name', self.__class__.__name__)
        entries = self.cache.get_entries(api_source=source_name, limit=1000)
        
        stats['fetcher_stats'] = {
            'source': source_name,
            'total_entries': len(entries),
            'cache_size_mb': sum(e.get('size_kb', 0) for e in entries) / 1024,
            'most_accessed': sorted(entries, key=lambda x: x.get('hit_count', 0), reverse=True)[:5]
        }
        
        return stats

def batch_cache_get(cache_keys: list) -> Dict[str, Any]:
    """
    Get multiple cache entries in a single operation
    
    Args:
        cache_keys: List of cache keys to retrieve
        
    Returns:
        Dict mapping cache keys to their values (or None if not found)
    """
    cache = get_cache_service()
    results = {}
    
    for key in cache_keys:
        try:
            results[key] = cache.get(key)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            results[key] = None
    
    return results

def batch_cache_set(items: Dict[str, Tuple[Any, str, Optional[int]]]) -> Dict[str, bool]:
    """
    Set multiple cache entries in a single operation
    
    Args:
        items: Dict mapping cache keys to (data, data_type, ttl) tuples
        
    Returns:
        Dict mapping cache keys to success status
    """
    cache = get_cache_service()
    results = {}
    
    for key, (data, data_type, ttl) in items.items():
        try:
            success = cache.set(key, data, data_type=data_type, ttl=ttl)
            results[key] = success
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            results[key] = False
    
    return results

def warm_cache(data_sources: list, symbols: list, periods: list):
    """
    Pre-populate cache with common data requests
    
    Args:
        data_sources: List of data source instances
        symbols: List of symbols to cache
        periods: List of periods to cache
    """
    logger.info(f"Warming cache for {len(data_sources)} sources, {len(symbols)} symbols, {len(periods)} periods")
    
    total_cached = 0
    errors = 0
    
    for source in data_sources:
        for symbol in symbols:
            for period in periods:
                try:
                    # Check if already cached
                    cache_key = generate_cache_key(
                        source.name,
                        'fetch',
                        symbol,
                        period
                    )
                    
                    cache = get_cache_service()
                    if cache.get(cache_key) is not None:
                        continue  # Already cached
                    
                    # Fetch and cache
                    logger.debug(f"Warming cache: {source.name} - {symbol} - {period}")
                    source.fetch(symbol, period)
                    total_cached += 1
                    
                except Exception as e:
                    logger.error(f"Cache warming error for {source.name} - {symbol} - {period}: {e}")
                    errors += 1
    
    logger.info(f"Cache warming complete: {total_cached} entries cached, {errors} errors")

def get_cache_info() -> Dict[str, Any]:
    """Get comprehensive cache information"""
    cache = get_cache_service()
    stats = cache.get_stats()
    
    # Add additional info
    import sqlite3
    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()
    
    # Get data type distribution
    cursor.execute("""
        SELECT data_type, COUNT(*) as count, 
               AVG(hit_count) as avg_hits,
               SUM(LENGTH(response_data)) / 1024.0 / 1024.0 as size_mb
        FROM api_cache
        GROUP BY data_type
        ORDER BY count DESC
    """)
    
    data_types = []
    for row in cursor.fetchall():
        data_types.append({
            'type': row[0],
            'count': row[1],
            'avg_hits': round(row[2] or 0, 2),
            'size_mb': round(row[3] or 0, 2)
        })
    
    # Get API source distribution
    cursor.execute("""
        SELECT api_source, COUNT(*) as count,
               SUM(hit_count) as total_hits
        FROM api_cache
        GROUP BY api_source
        ORDER BY total_hits DESC
    """)
    
    sources = []
    for row in cursor.fetchall():
        sources.append({
            'source': row[0],
            'entries': row[1],
            'total_hits': row[2] or 0
        })
    
    conn.close()
    
    stats['data_type_distribution'] = data_types
    stats['api_source_distribution'] = sources
    
    return stats