"""
Cache Warming and Maintenance Jobs
Automated tasks to optimize cache performance and ensure data availability
"""
import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import threading

from .cache_service import get_cache_service
from .cache_integration import warm_cache, get_cache_info
from .data_fetcher import get_fetcher
from .enhanced_data_fetcher import EnhancedDataFetcher

logger = logging.getLogger(__name__)


class CacheMaintenanceManager:
    """Manages cache warming and maintenance tasks"""
    
    def __init__(self):
        self.cache = get_cache_service()
        self.fetcher = get_fetcher()
        self.enhanced_fetcher = EnhancedDataFetcher()
        self.is_running = False
        self._thread = None
        
        # Configuration
        self.config = {
            'warm_on_startup': True,
            'warm_interval_minutes': 30,
            'optimize_interval_hours': 6,
            'clear_expired_interval_hours': 1,
            'monitor_interval_minutes': 5,
            'common_symbols': ['BTC', 'BTCUSDT', 'bitcoin'],
            'common_periods': ['5m', '1h', '1d', '7d', '30d'],
            'priority_sources': ['binance', 'coingecko'],
            'hit_rate_threshold': 0.7,  # Alert if hit rate drops below 70%
            'cache_size_limit_mb': 1000  # Alert if cache exceeds 1GB
        }
    
    def start(self):
        """Start maintenance tasks in background thread"""
        if self.is_running:
            logger.warning("Cache maintenance already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        
        logger.info("Cache maintenance started")
        
        # Warm cache on startup if configured
        if self.config['warm_on_startup']:
            threading.Thread(target=self._initial_warm_cache, daemon=True).start()
    
    def stop(self):
        """Stop maintenance tasks"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Cache maintenance stopped")
    
    def _run_scheduler(self):
        """Run scheduled maintenance tasks"""
        # Schedule tasks
        schedule.every(self.config['warm_interval_minutes']).minutes.do(self._warm_cache_job)
        schedule.every(self.config['optimize_interval_hours']).hours.do(self._optimize_cache_job)
        schedule.every(self.config['clear_expired_interval_hours']).hours.do(self._clear_expired_job)
        schedule.every(self.config['monitor_interval_minutes']).minutes.do(self._monitor_cache_job)
        
        # Run initial tasks
        self._clear_expired_job()
        self._monitor_cache_job()
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _initial_warm_cache(self):
        """Initial cache warming on startup"""
        logger.info("Starting initial cache warming...")
        
        try:
            # Warm essential data
            essential_tasks = [
                # Current prices
                ('current_prices', self._warm_current_prices),
                # Recent OHLCV data
                ('recent_ohlcv', self._warm_recent_ohlcv),
                # Sentiment data
                ('sentiment', self._warm_sentiment_data),
                # On-chain metrics
                ('onchain', self._warm_onchain_data)
            ]
            
            for task_name, task_func in essential_tasks:
                try:
                    logger.info(f"Warming {task_name}...")
                    task_func()
                except Exception as e:
                    logger.error(f"Failed to warm {task_name}: {e}")
            
            logger.info("Initial cache warming completed")
            
        except Exception as e:
            logger.error(f"Initial cache warming failed: {e}")
    
    def _warm_cache_job(self):
        """Periodic cache warming job"""
        try:
            logger.info("Running periodic cache warming...")
            
            # Get cache statistics
            stats = self.cache.get_stats()
            hit_rate = stats['session_stats'].get('hit_rate', 0)
            
            # Determine what to warm based on hit rate
            if hit_rate < self.config['hit_rate_threshold']:
                # Low hit rate - warm more aggressively
                self._warm_comprehensive_data()
            else:
                # Good hit rate - just refresh common data
                self._warm_common_data()
                
        except Exception as e:
            logger.error(f"Cache warming job failed: {e}")
    
    def _warm_current_prices(self):
        """Warm current price data"""
        for symbol in self.config['common_symbols']:
            try:
                self.fetcher.fetch_price(symbol)
            except Exception as e:
                logger.warning(f"Failed to warm price for {symbol}: {e}")
    
    def _warm_recent_ohlcv(self):
        """Warm recent OHLCV data"""
        periods = ['5m', '1h', '1d']  # Focus on short-term data
        
        for source in self.fetcher.crypto_sources[:2]:  # Top 2 sources
            if source.name not in self.config['priority_sources']:
                continue
                
            for symbol in ['BTC', 'BTCUSDT'][:1]:  # Just BTC
                for period in periods:
                    try:
                        source.fetch(symbol, period)
                        time.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to warm {source.name} {symbol} {period}: {e}")
    
    def _warm_sentiment_data(self):
        """Warm sentiment data"""
        try:
            self.fetcher.sentiment_source.fetch_fear_greed_index()
            time.sleep(0.5)
            self.fetcher.sentiment_source.fetch_reddit_sentiment()
        except Exception as e:
            logger.warning(f"Failed to warm sentiment data: {e}")
    
    def _warm_onchain_data(self):
        """Warm on-chain data"""
        try:
            self.fetcher.onchain_source.fetch_blockchain_info_metrics()
        except Exception as e:
            logger.warning(f"Failed to warm on-chain data: {e}")
    
    def _warm_common_data(self):
        """Warm commonly requested data"""
        # Use the cache_integration warm_cache function
        data_sources = []
        
        # Get priority sources
        for source in self.fetcher.crypto_sources:
            if source.name in self.config['priority_sources']:
                data_sources.append(source)
        
        if data_sources:
            warm_cache(
                data_sources,
                self.config['common_symbols'][:2],  # Top 2 symbols
                self.config['common_periods'][:3]    # Top 3 periods
            )
    
    def _warm_comprehensive_data(self):
        """Warm comprehensive data when hit rate is low"""
        logger.info("Performing comprehensive cache warming due to low hit rate")
        
        # Warm all configured data
        data_sources = self.fetcher.crypto_sources[:3]  # Top 3 sources
        
        warm_cache(
            data_sources,
            self.config['common_symbols'],
            self.config['common_periods']
        )
    
    def _optimize_cache_job(self):
        """Periodic cache optimization"""
        try:
            logger.info("Running cache optimization...")
            
            report = self.cache.optimize_cache()
            
            # Log optimization results
            if report and 'final_stats' in report:
                entries_removed = report['final_stats'].get('entries_removed', 0)
                logger.info(f"Cache optimization completed: {entries_removed} entries removed")
                
                # Check TTL suggestions
                if 'ttl_suggestions' in report and report['ttl_suggestions']:
                    logger.info(f"TTL adjustment suggestions: {report['ttl_suggestions']}")
                    
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    def _clear_expired_job(self):
        """Clear expired cache entries"""
        try:
            entries_removed = self.cache.clear_expired()
            if entries_removed > 0:
                logger.info(f"Cleared {entries_removed} expired cache entries")
        except Exception as e:
            logger.error(f"Failed to clear expired entries: {e}")
    
    def _monitor_cache_job(self):
        """Monitor cache health and alert on issues"""
        try:
            stats = self.cache.get_detailed_stats()
            
            # Check hit rate
            hit_rate = stats['session_stats'].get('hit_rate', 0)
            if hit_rate < self.config['hit_rate_threshold']:
                logger.warning(f"Cache hit rate low: {hit_rate:.2%}")
            
            # Check cache size
            cache_size_mb = stats.get('total_size_mb', 0)
            if cache_size_mb > self.config['cache_size_limit_mb']:
                logger.warning(f"Cache size exceeds limit: {cache_size_mb:.2f}MB")
                # Trigger optimization
                self._optimize_cache_job()
            
            # Check error rates
            total_requests = stats['session_stats'].get('hits', 0) + stats['session_stats'].get('misses', 0)
            if total_requests > 100:  # Only check after sufficient requests
                # Log cache performance
                logger.info(
                    f"Cache performance - Hit rate: {hit_rate:.2%}, "
                    f"Size: {cache_size_mb:.2f}MB, "
                    f"Entries: {stats.get('total_entries', 0)}"
                )
                
        except Exception as e:
            logger.error(f"Cache monitoring failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current maintenance status"""
        return {
            'is_running': self.is_running,
            'config': self.config,
            'cache_stats': self.cache.get_stats(),
            'scheduled_jobs': [
                {
                    'name': 'warm_cache',
                    'interval_minutes': self.config['warm_interval_minutes'],
                    'next_run': self._get_next_run_time('warm_cache')
                },
                {
                    'name': 'optimize_cache',
                    'interval_hours': self.config['optimize_interval_hours'],
                    'next_run': self._get_next_run_time('optimize_cache')
                },
                {
                    'name': 'clear_expired',
                    'interval_hours': self.config['clear_expired_interval_hours'],
                    'next_run': self._get_next_run_time('clear_expired')
                }
            ]
        }
    
    def _get_next_run_time(self, job_name: str) -> str:
        """Get next scheduled run time for a job"""
        # This is a simplified implementation
        # In production, you'd track actual schedule times
        now = datetime.now()
        
        if job_name == 'warm_cache':
            next_run = now + timedelta(minutes=self.config['warm_interval_minutes'])
        elif job_name == 'optimize_cache':
            next_run = now + timedelta(hours=self.config['optimize_interval_hours'])
        elif job_name == 'clear_expired':
            next_run = now + timedelta(hours=self.config['clear_expired_interval_hours'])
        else:
            next_run = now
            
        return next_run.isoformat()
    
    def trigger_warm_cache(self, aggressive: bool = False):
        """Manually trigger cache warming"""
        if aggressive:
            self._warm_comprehensive_data()
        else:
            self._warm_common_data()
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update maintenance configuration"""
        self.config.update(config_updates)
        logger.info(f"Cache maintenance config updated: {config_updates}")


# Singleton instance
_maintenance_manager = None

def get_maintenance_manager() -> CacheMaintenanceManager:
    """Get singleton maintenance manager instance"""
    global _maintenance_manager
    if _maintenance_manager is None:
        _maintenance_manager = CacheMaintenanceManager()
    return _maintenance_manager

# Convenience functions
def start_cache_maintenance():
    """Start cache maintenance tasks"""
    manager = get_maintenance_manager()
    manager.start()

def stop_cache_maintenance():
    """Stop cache maintenance tasks"""
    manager = get_maintenance_manager()
    manager.stop()

def get_maintenance_status():
    """Get maintenance status"""
    manager = get_maintenance_manager()
    return manager.get_status()