"""
API Response Cache Service
Provides persistent caching for external API responses in SQLite
"""
import sqlite3
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta, date
from typing import Any, Dict, Optional, List, Tuple
from functools import wraps
import pandas as pd
import pickle
import base64
import numpy as np

# Custom JSON encoder for datetime and numpy types
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

logger = logging.getLogger(__name__)

class CacheService:
    """
    Persistent cache service for API responses using SQLite
    """
    
    def __init__(self, db_path: str = "/app/data/api_cache.db"):
        self.db_path = db_path
        self.init_database()
        
        # Default TTLs for different data types (in seconds)
        self.default_ttls = {
            'price': 60,           # 1 minute for real-time prices
            'ohlcv': 300,          # 5 minutes for OHLCV data
            'sentiment': 1800,     # 30 minutes for sentiment
            'onchain': 1800,       # 30 minutes for on-chain data
            'macro': 3600,         # 1 hour for macro data
            'news': 900,           # 15 minutes for news
            'fear_greed': 3600,    # 1 hour for fear & greed index
            'network_stats': 1800, # 30 minutes for network stats
            'default': 300         # 5 minutes default
        }
        
        # Track cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'invalidations': 0
        }
    
    def init_database(self):
        """Initialize cache database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                data_type TEXT NOT NULL,
                api_source TEXT NOT NULL,
                response_data TEXT NOT NULL,
                response_format TEXT DEFAULT 'json',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Cache invalidation log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_invalidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT,
                reason TEXT,
                invalidated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entries_affected INTEGER
            )
        ''')
        
        # Cache statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_hits INTEGER DEFAULT 0,
                total_misses INTEGER DEFAULT 0,
                total_writes INTEGER DEFAULT 0,
                total_invalidations INTEGER DEFAULT 0,
                unique_keys INTEGER DEFAULT 0,
                total_size_bytes INTEGER DEFAULT 0
            )
        ''')
        
        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_key ON api_cache(cache_key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON api_cache(expires_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON api_cache(data_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_source ON api_cache(api_source)')
        
        conn.commit()
        conn.close()
        
        logger.info("Cache database initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached data by key
        Returns None if not found or expired
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clean expired entries first
            self._clean_expired(cursor)
            
            # Fetch cache entry
            cursor.execute('''
                SELECT response_data, response_format, expires_at 
                FROM api_cache 
                WHERE cache_key = ? AND expires_at > ?
            ''', (key, datetime.now()))
            
            row = cursor.fetchone()
            
            if row:
                response_data, response_format, expires_at = row
                
                # Update hit count and last accessed
                cursor.execute('''
                    UPDATE api_cache 
                    SET hit_count = hit_count + 1, 
                        last_accessed = CURRENT_TIMESTAMP 
                    WHERE cache_key = ?
                ''', (key,))
                
                conn.commit()
                
                # Deserialize data based on format
                data = self._deserialize(response_data, response_format)
                
                self.stats['hits'] += 1
                self._update_daily_stats(cursor, hits=1)
                
                logger.debug(f"Cache hit for key: {key}")
                return data
            else:
                self.stats['misses'] += 1
                self._update_daily_stats(cursor, misses=1)
                logger.debug(f"Cache miss for key: {key}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cache entry: {e}")
            return None
        finally:
            conn.close()
    
    def set(self, key: str, data: Any, data_type: str = 'default', 
            api_source: str = 'unknown', ttl: Optional[int] = None,
            metadata: Optional[Dict] = None) -> bool:
        """
        Set cache entry with automatic serialization
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Determine TTL
            if ttl is None:
                ttl = self.default_ttls.get(data_type, self.default_ttls['default'])
            
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Serialize data
            response_data, response_format = self._serialize(data)
            
            # Prepare metadata
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Insert or replace cache entry
            cursor.execute('''
                INSERT OR REPLACE INTO api_cache 
                (cache_key, data_type, api_source, response_data, response_format, 
                 expires_at, metadata, created_at, hit_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0, CURRENT_TIMESTAMP)
            ''', (key, data_type, api_source, response_data, response_format, 
                  expires_at, metadata_json))
            
            conn.commit()
            
            self.stats['writes'] += 1
            self._update_daily_stats(cursor, writes=1)
            
            logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False
        finally:
            conn.close()
    
    def invalidate(self, pattern: Optional[str] = None, data_type: Optional[str] = None,
                   api_source: Optional[str] = None, reason: str = "Manual invalidation") -> int:
        """
        Invalidate cache entries matching criteria
        Returns number of entries invalidated
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build query
            query = "DELETE FROM api_cache WHERE 1=1"
            params = []
            
            if pattern:
                query += " AND cache_key LIKE ?"
                params.append(f"%{pattern}%")
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type)
            
            if api_source:
                query += " AND api_source = ?"
                params.append(api_source)
            
            # Execute deletion
            cursor.execute(query, params)
            entries_affected = cursor.rowcount
            
            # Log invalidation
            cursor.execute('''
                INSERT INTO cache_invalidation_log 
                (pattern, reason, entries_affected)
                VALUES (?, ?, ?)
            ''', (pattern or 'all', reason, entries_affected))
            
            conn.commit()
            
            self.stats['invalidations'] += entries_affected
            self._update_daily_stats(cursor, invalidations=entries_affected)
            
            logger.info(f"Invalidated {entries_affected} cache entries")
            return entries_affected
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get overall stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT data_type) as unique_types,
                    COUNT(DISTINCT api_source) as unique_sources,
                    SUM(LENGTH(response_data)) as total_size_bytes,
                    AVG(hit_count) as avg_hits_per_entry
                FROM api_cache
            ''')
            
            row = cursor.fetchone()
            
            # Get today's stats
            cursor.execute('''
                SELECT total_hits, total_misses, total_writes, total_invalidations
                FROM cache_stats
                WHERE date = DATE('now')
            ''')
            
            today_row = cursor.fetchone()
            
            stats = {
                'total_entries': row[0] or 0,
                'unique_types': row[1] or 0,
                'unique_sources': row[2] or 0,
                'total_size_mb': (row[3] or 0) / (1024 * 1024),
                'avg_hits_per_entry': row[4] or 0,
                'session_stats': self.stats.copy(),
                'today_stats': {
                    'hits': today_row[0] if today_row else 0,
                    'misses': today_row[1] if today_row else 0,
                    'writes': today_row[2] if today_row else 0,
                    'invalidations': today_row[3] if today_row else 0
                } if today_row else None
            }
            
            # Get hit rate
            total_requests = self.stats['hits'] + self.stats['misses']
            stats['session_stats']['hit_rate'] = (
                self.stats['hits'] / total_requests if total_requests > 0 else 0
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
        finally:
            conn.close()
    
    def get_entries(self, data_type: Optional[str] = None, 
                    api_source: Optional[str] = None,
                    limit: int = 100) -> List[Dict]:
        """Get cache entries with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = '''
                SELECT cache_key, data_type, api_source, created_at, 
                       expires_at, hit_count, last_accessed, 
                       LENGTH(response_data) as size_bytes
                FROM api_cache
                WHERE 1=1
            '''
            params = []
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type)
            
            if api_source:
                query += " AND api_source = ?"
                params.append(api_source)
            
            query += " ORDER BY last_accessed DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            entries = []
            
            for row in cursor.fetchall():
                entry = dict(zip(columns, row))
                entry['size_kb'] = entry['size_bytes'] / 1024
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error getting cache entries: {e}")
            return []
        finally:
            conn.close()
    
    def clear_expired(self) -> int:
        """Clear all expired entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM api_cache 
                WHERE expires_at <= ?
            ''', (datetime.now(),))
            
            entries_removed = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleared {entries_removed} expired cache entries")
            return entries_removed
            
        except Exception as e:
            logger.error(f"Error clearing expired entries: {e}")
            return 0
        finally:
            conn.close()
    
    def batch_get(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """
        Get multiple cache entries in a single database operation
        
        Args:
            keys: List of cache keys to retrieve
            
        Returns:
            Dict mapping keys to their values (None if not found/expired)
        """
        if not keys:
            return {}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = {}
        
        try:
            # Clean expired entries first
            self._clean_expired(cursor)
            
            # Build query with placeholders
            placeholders = ','.join(['?' for _ in keys])
            cursor.execute(f'''
                SELECT cache_key, response_data, response_format, expires_at 
                FROM api_cache 
                WHERE cache_key IN ({placeholders}) AND expires_at > ?
            ''', keys + [datetime.now()])
            
            # Process results
            for row in cursor.fetchall():
                cache_key, response_data, response_format, expires_at = row
                
                # Update hit count
                cursor.execute('''
                    UPDATE api_cache 
                    SET hit_count = hit_count + 1, 
                        last_accessed = CURRENT_TIMESTAMP 
                    WHERE cache_key = ?
                ''', (cache_key,))
                
                # Deserialize data
                data = self._deserialize(response_data, response_format)
                results[cache_key] = data
                
            conn.commit()
            
            # Fill in None for missing keys
            for key in keys:
                if key not in results:
                    results[key] = None
                    self.stats['misses'] += 1
                else:
                    self.stats['hits'] += 1
                    
            self._update_daily_stats(cursor, 
                                   hits=sum(1 for v in results.values() if v is not None),
                                   misses=sum(1 for v in results.values() if v is None))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch get: {e}")
            return {key: None for key in keys}
        finally:
            conn.close()
    
    def batch_set(self, entries: Dict[str, Tuple[Any, Dict[str, Any]]]) -> Dict[str, bool]:
        """
        Set multiple cache entries in a single database operation
        
        Args:
            entries: Dict mapping cache keys to (data, metadata) tuples
                    metadata should include: data_type, api_source, ttl (optional)
                    
        Returns:
            Dict mapping keys to success status
        """
        if not entries:
            return {}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = {}
        
        try:
            # Prepare batch insert data
            batch_data = []
            for key, (data, metadata) in entries.items():
                # Get metadata
                data_type = metadata.get('data_type', 'default')
                api_source = metadata.get('api_source', 'unknown')
                ttl = metadata.get('ttl', self.default_ttls.get(data_type, self.default_ttls['default']))
                meta_json = json.dumps(metadata.get('metadata'))
                
                # Calculate expiration
                expires_at = datetime.now() + timedelta(seconds=ttl)
                
                # Serialize data
                response_data, response_format = self._serialize(data)
                
                batch_data.append((
                    key, data_type, api_source, response_data, response_format,
                    expires_at, meta_json
                ))
                
            # Batch insert/replace
            cursor.executemany('''
                INSERT OR REPLACE INTO api_cache 
                (cache_key, data_type, api_source, response_data, response_format, 
                 expires_at, metadata, created_at, hit_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0, CURRENT_TIMESTAMP)
            ''', batch_data)
            
            conn.commit()
            
            # All successful
            results = {key: True for key in entries.keys()}
            self.stats['writes'] += len(entries)
            self._update_daily_stats(cursor, writes=len(entries))
            
            logger.debug(f"Batch set {len(entries)} cache entries")
            
        except Exception as e:
            logger.error(f"Error in batch set: {e}")
            results = {key: False for key in entries.keys()}
            
        finally:
            conn.close()
            
        return results
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics including performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            base_stats = self.get_stats()
            
            # Get age distribution
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN (julianday('now') - julianday(created_at)) * 24 < 1 THEN '< 1 hour'
                        WHEN (julianday('now') - julianday(created_at)) * 24 < 24 THEN '1-24 hours'
                        WHEN (julianday('now') - julianday(created_at)) < 7 THEN '1-7 days'
                        ELSE '> 7 days'
                    END as age_group,
                    COUNT(*) as count
                FROM api_cache
                GROUP BY age_group
            ''')
            
            age_distribution = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get hit rate by data type
            cursor.execute('''
                SELECT data_type, 
                       COUNT(*) as entries,
                       SUM(hit_count) as total_hits,
                       AVG(hit_count) as avg_hits,
                       MAX(hit_count) as max_hits
                FROM api_cache
                GROUP BY data_type
                ORDER BY total_hits DESC
            ''')
            
            hit_rates = []
            for row in cursor.fetchall():
                hit_rates.append({
                    'data_type': row[0],
                    'entries': row[1],
                    'total_hits': row[2] or 0,
                    'avg_hits': round(row[3] or 0, 2),
                    'max_hits': row[4] or 0
                })
            
            # Get API source performance
            cursor.execute('''
                SELECT api_source,
                       COUNT(*) as cached_calls,
                       SUM(hit_count) as cache_hits,
                       AVG(LENGTH(response_data)) as avg_response_size,
                       MIN(created_at) as oldest_entry,
                       MAX(last_accessed) as latest_access
                FROM api_cache
                GROUP BY api_source
                ORDER BY cache_hits DESC
            ''')
            
            api_performance = []
            for row in cursor.fetchall():
                api_performance.append({
                    'api_source': row[0],
                    'cached_calls': row[1],
                    'cache_hits': row[2] or 0,
                    'avg_response_size_kb': round((row[3] or 0) / 1024, 2),
                    'oldest_entry': row[4],
                    'latest_access': row[5]
                })
            
            # Get expiration timeline
            cursor.execute('''
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', expires_at) as hour,
                    COUNT(*) as expiring_count
                FROM api_cache
                WHERE expires_at > datetime('now')
                GROUP BY hour
                ORDER BY hour
                LIMIT 24
            ''')
            
            expiration_timeline = [
                {'hour': row[0], 'count': row[1]} 
                for row in cursor.fetchall()
            ]
            
            # Compile detailed stats
            detailed_stats = {
                **base_stats,
                'age_distribution': age_distribution,
                'hit_rates_by_type': hit_rates,
                'api_performance': api_performance,
                'expiration_timeline': expiration_timeline,
                'cache_efficiency': {
                    'total_api_calls_saved': sum(h['total_hits'] for h in hit_rates),
                    'estimated_time_saved_seconds': sum(h['total_hits'] for h in hit_rates) * 0.5,  # Assume 0.5s per API call
                    'cache_freshness': len([e for e in expiration_timeline if e['count'] > 0]) / 24 * 100 if expiration_timeline else 0
                }
            }
            
            return detailed_stats
            
        except Exception as e:
            logger.error(f"Error getting detailed stats: {e}")
            return base_stats
        finally:
            conn.close()
    
    def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimize cache by removing low-value entries and analyzing usage patterns
        
        Returns:
            Optimization report
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            report = {
                'started_at': datetime.now().isoformat(),
                'actions': []
            }
            
            # 1. Remove entries with zero hits that are older than 1 hour
            cursor.execute('''
                DELETE FROM api_cache
                WHERE hit_count = 0 
                AND created_at < datetime('now', '-1 hour')
            ''')
            
            zero_hit_removed = cursor.rowcount
            report['actions'].append({
                'action': 'remove_zero_hit_entries',
                'removed': zero_hit_removed
            })
            
            # 2. Remove very large entries with low hit rates
            cursor.execute('''
                DELETE FROM api_cache
                WHERE LENGTH(response_data) > 1048576  -- > 1MB
                AND hit_count < 5
                AND created_at < datetime('now', '-1 hour')
            ''')
            
            large_low_hit_removed = cursor.rowcount
            report['actions'].append({
                'action': 'remove_large_low_hit_entries',
                'removed': large_low_hit_removed
            })
            
            # 3. Analyze and suggest TTL adjustments
            cursor.execute('''
                SELECT data_type,
                       AVG(CASE 
                           WHEN hit_count = 0 THEN 0
                           ELSE (julianday(last_accessed) - julianday(created_at)) * 24 * 60
                       END) as avg_useful_life_minutes,
                       COUNT(*) as entries
                FROM api_cache
                GROUP BY data_type
            ''')
            
            ttl_suggestions = []
            for row in cursor.fetchall():
                data_type, avg_life, entries = row
                if avg_life and entries > 10:  # Only suggest for types with enough data
                    current_ttl = self.default_ttls.get(data_type, self.default_ttls['default'])
                    suggested_ttl = int(avg_life * 60 * 1.2)  # 20% buffer
                    
                    if abs(suggested_ttl - current_ttl) > 60:  # Only suggest if difference > 1 minute
                        ttl_suggestions.append({
                            'data_type': data_type,
                            'current_ttl': current_ttl,
                            'suggested_ttl': suggested_ttl,
                            'based_on_entries': entries
                        })
            
            report['ttl_suggestions'] = ttl_suggestions
            
            # 4. Vacuum database to reclaim space
            conn.execute('VACUUM')
            
            # 5. Update statistics
            cursor.execute('ANALYZE')
            
            conn.commit()
            
            # Get final stats
            cursor.execute('SELECT COUNT(*), SUM(LENGTH(response_data)) FROM api_cache')
            final_count, final_size = cursor.fetchone()
            
            report['completed_at'] = datetime.now().isoformat()
            report['final_stats'] = {
                'total_entries': final_count or 0,
                'total_size_mb': (final_size or 0) / (1024 * 1024),
                'entries_removed': zero_hit_removed + large_low_hit_removed
            }
            
            logger.info(f"Cache optimization completed: {report['final_stats']['entries_removed']} entries removed")
            
            return report
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    def export_metrics(self, format: str = 'json') -> Optional[str]:
        """
        Export cache metrics for monitoring systems
        
        Args:
            format: Export format ('json', 'prometheus')
            
        Returns:
            Formatted metrics string
        """
        try:
            stats = self.get_detailed_stats()
            
            if format == 'json':
                return json.dumps(stats, indent=2, default=str)
            
            elif format == 'prometheus':
                # Prometheus text format
                lines = []
                
                # Basic metrics
                lines.append(f'# HELP btc_cache_total_entries Total number of cache entries')
                lines.append(f'# TYPE btc_cache_total_entries gauge')
                lines.append(f'btc_cache_total_entries {stats.get("total_entries", 0)}')
                
                lines.append(f'# HELP btc_cache_size_mb Total cache size in MB')
                lines.append(f'# TYPE btc_cache_size_mb gauge')
                lines.append(f'btc_cache_size_mb {stats.get("total_size_mb", 0)}')
                
                # Session metrics
                session = stats.get('session_stats', {})
                lines.append(f'# HELP btc_cache_session_hits Cache hits in current session')
                lines.append(f'# TYPE btc_cache_session_hits counter')
                lines.append(f'btc_cache_session_hits {session.get("hits", 0)}')
                
                lines.append(f'# HELP btc_cache_session_misses Cache misses in current session')
                lines.append(f'# TYPE btc_cache_session_misses counter')
                lines.append(f'btc_cache_session_misses {session.get("misses", 0)}')
                
                lines.append(f'# HELP btc_cache_hit_rate Cache hit rate')
                lines.append(f'# TYPE btc_cache_hit_rate gauge')
                lines.append(f'btc_cache_hit_rate {session.get("hit_rate", 0)}')
                
                return '\n'.join(lines)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None
    
    def _serialize(self, data: Any) -> Tuple[str, str]:
        """Serialize data for storage"""
        try:
            # Check if it's a pandas DataFrame
            if pd is not None and isinstance(data, pd.DataFrame):
                # Serialize DataFrame as pickle
                pickled = pickle.dumps(data)
                encoded = base64.b64encode(pickled).decode('utf-8')
                return encoded, 'dataframe'
            elif isinstance(data, (dict, list)):
                # Serialize as JSON
                return json.dumps(data, cls=DateTimeEncoder), 'json'
            else:
                # Serialize other types as pickle
                pickled = pickle.dumps(data)
                encoded = base64.b64encode(pickled).decode('utf-8')
                return encoded, 'pickle'
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            # Fallback to pickle for any serialization errors
            pickled = pickle.dumps(data)
            encoded = base64.b64encode(pickled).decode('utf-8')
            return encoded, 'pickle'
    
    def _deserialize(self, data: str, format: str) -> Any:
        """Deserialize data from storage"""
        if format == 'json':
            return json.loads(data)
        elif format == 'dataframe':
            decoded = base64.b64decode(data.encode('utf-8'))
            return pickle.loads(decoded)
        elif format == 'pickle':
            decoded = base64.b64decode(data.encode('utf-8'))
            return pickle.loads(decoded)
        else:
            # Try JSON first, then pickle
            try:
                return json.loads(data)
            except:
                decoded = base64.b64decode(data.encode('utf-8'))
                return pickle.loads(decoded)
    
    def _clean_expired(self, cursor):
        """Clean expired entries (called during get operations)"""
        cursor.execute('''
            DELETE FROM api_cache 
            WHERE expires_at <= ?
        ''', (datetime.now(),))
    
    def _update_daily_stats(self, cursor, hits: int = 0, misses: int = 0, 
                           writes: int = 0, invalidations: int = 0):
        """Update daily statistics"""
        today = datetime.now().date()
        
        cursor.execute('''
            INSERT OR IGNORE INTO cache_stats 
            (date, total_hits, total_misses, total_writes, total_invalidations)
            VALUES (?, 0, 0, 0, 0)
        ''', (today,))
        
        cursor.execute('''
            UPDATE cache_stats
            SET total_hits = total_hits + ?,
                total_misses = total_misses + ?,
                total_writes = total_writes + ?,
                total_invalidations = total_invalidations + ?
            WHERE date = ?
        ''', (hits, misses, writes, invalidations, today))

def cache_api_call(data_type: str = 'default', ttl: Optional[int] = None):
    """
    Decorator for caching API calls
    
    Usage:
        @cache_api_call(data_type='price', ttl=60)
        def fetch_btc_price():
            return requests.get(...).json()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Get cache service instance
            cache = kwargs.pop('_cache', None) or CacheService()
            
            # Try to get from cache
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Call original function
            result = func(*args, **kwargs)
            
            # Cache the result
            api_source = func.__module__.split('.')[-1]
            cache.set(cache_key, result, data_type=data_type, 
                     api_source=api_source, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator

# Singleton cache instance
_cache_instance = None

def get_cache_service() -> CacheService:
    """Get singleton cache service instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance