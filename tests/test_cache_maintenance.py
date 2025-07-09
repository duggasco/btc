"""
Test cache maintenance functionality
"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))

from services.cache_maintenance import CacheMaintenanceManager, get_maintenance_manager


class TestCacheMaintenanceManager:
    """Test cache maintenance manager"""
    
    @pytest.fixture
    def mock_cache(self):
        """Create mock cache service"""
        cache = Mock()
        cache.get_stats.return_value = {
            'session_stats': {
                'hits': 100,
                'misses': 20,
                'hit_rate': 0.83
            },
            'total_entries': 150,
            'total_size_mb': 2.5
        }
        cache.get_detailed_stats.return_value = {
            'session_stats': {
                'hits': 100,
                'misses': 20,
                'hit_rate': 0.83
            },
            'total_entries': 150,
            'total_size_mb': 2.5
        }
        cache.optimize_cache.return_value = {
            'final_stats': {
                'entries_removed': 10
            },
            'ttl_suggestions': {}
        }
        cache.clear_expired.return_value = 5
        return cache
    
    @pytest.fixture
    def mock_fetcher(self):
        """Create mock data fetcher"""
        fetcher = Mock()
        fetcher.fetch_price.return_value = 45000.0
        
        # Mock crypto sources
        source1 = Mock()
        source1.name = 'binance'
        source1.fetch.return_value = Mock()
        
        source2 = Mock()
        source2.name = 'coingecko'
        source2.fetch.return_value = Mock()
        
        fetcher.crypto_sources = [source1, source2]
        
        # Mock sentiment source
        sentiment_source = Mock()
        sentiment_source.fetch_fear_greed_index.return_value = 45
        sentiment_source.fetch_reddit_sentiment.return_value = 0.6
        fetcher.sentiment_source = sentiment_source
        
        # Mock onchain source
        onchain_source = Mock()
        onchain_source.fetch_blockchain_info_metrics.return_value = {}
        fetcher.onchain_source = onchain_source
        
        return fetcher
    
    @pytest.fixture
    def maintenance_manager(self, mock_cache, mock_fetcher):
        """Create maintenance manager with mocks"""
        with patch('services.cache_maintenance.get_cache_service', return_value=mock_cache), \
             patch('services.cache_maintenance.get_fetcher', return_value=mock_fetcher), \
             patch('services.cache_maintenance.EnhancedDataFetcher'):
            manager = CacheMaintenanceManager()
            return manager
    
    def test_initialization(self, maintenance_manager):
        """Test manager initialization"""
        assert maintenance_manager is not None
        assert not maintenance_manager.is_running
        assert maintenance_manager.config['warm_interval_minutes'] == 30
        assert maintenance_manager.config['optimize_interval_hours'] == 6
    
    def test_start_stop(self, maintenance_manager):
        """Test starting and stopping maintenance"""
        # Start
        maintenance_manager.start()
        assert maintenance_manager.is_running
        
        # Give it a moment to start the thread
        time.sleep(0.1)
        
        # Stop
        maintenance_manager.stop()
        assert not maintenance_manager.is_running
    
    def test_get_status(self, maintenance_manager):
        """Test getting maintenance status"""
        status = maintenance_manager.get_status()
        
        assert 'is_running' in status
        assert 'config' in status
        assert 'cache_stats' in status
        assert 'scheduled_jobs' in status
        
        # Check scheduled jobs
        assert len(status['scheduled_jobs']) >= 3
        job_names = [job['name'] for job in status['scheduled_jobs']]
        assert 'warm_cache' in job_names
        assert 'optimize_cache' in job_names
        assert 'clear_expired' in job_names
    
    def test_warm_current_prices(self, maintenance_manager):
        """Test warming current prices"""
        maintenance_manager._warm_current_prices()
        
        # Check fetcher was called for common symbols
        assert maintenance_manager.fetcher.fetch_price.called
        call_args = [call[0][0] for call in maintenance_manager.fetcher.fetch_price.call_args_list]
        assert 'BTC' in call_args or 'BTCUSDT' in call_args or 'bitcoin' in call_args
    
    def test_warm_recent_ohlcv(self, maintenance_manager):
        """Test warming recent OHLCV data"""
        maintenance_manager._warm_recent_ohlcv()
        
        # Check sources were called
        for source in maintenance_manager.fetcher.crypto_sources:
            if source.name in ['binance', 'coingecko']:
                assert source.fetch.called
    
    def test_warm_sentiment_data(self, maintenance_manager):
        """Test warming sentiment data"""
        maintenance_manager._warm_sentiment_data()
        
        # Check sentiment source was called
        assert maintenance_manager.fetcher.sentiment_source.fetch_fear_greed_index.called
        assert maintenance_manager.fetcher.sentiment_source.fetch_reddit_sentiment.called
    
    def test_warm_onchain_data(self, maintenance_manager):
        """Test warming on-chain data"""
        maintenance_manager._warm_onchain_data()
        
        # Check onchain source was called
        assert maintenance_manager.fetcher.onchain_source.fetch_blockchain_info_metrics.called
    
    def test_optimize_cache_job(self, maintenance_manager):
        """Test cache optimization job"""
        maintenance_manager._optimize_cache_job()
        
        # Check cache optimize was called
        assert maintenance_manager.cache.optimize_cache.called
    
    def test_clear_expired_job(self, maintenance_manager):
        """Test clearing expired entries"""
        maintenance_manager._clear_expired_job()
        
        # Check cache clear_expired was called
        assert maintenance_manager.cache.clear_expired.called
    
    def test_monitor_cache_job(self, maintenance_manager):
        """Test cache monitoring"""
        # Set up low hit rate to trigger warning
        maintenance_manager.cache.get_detailed_stats.return_value = {
            'session_stats': {
                'hits': 60,
                'misses': 40,
                'hit_rate': 0.6  # Below threshold
            },
            'total_entries': 150,
            'total_size_mb': 2.5
        }
        
        with patch('services.cache_maintenance.logger') as mock_logger:
            maintenance_manager._monitor_cache_job()
            
            # Check warning was logged for low hit rate
            mock_logger.warning.assert_called()
    
    def test_trigger_warm_cache(self, maintenance_manager):
        """Test manual cache warming trigger"""
        with patch.object(maintenance_manager, '_warm_common_data') as mock_common, \
             patch.object(maintenance_manager, '_warm_comprehensive_data') as mock_comprehensive:
            
            # Normal warm
            maintenance_manager.trigger_warm_cache(aggressive=False)
            assert mock_common.called
            assert not mock_comprehensive.called
            
            # Reset mocks
            mock_common.reset_mock()
            mock_comprehensive.reset_mock()
            
            # Aggressive warm
            maintenance_manager.trigger_warm_cache(aggressive=True)
            assert not mock_common.called
            assert mock_comprehensive.called
    
    def test_update_config(self, maintenance_manager):
        """Test updating configuration"""
        new_config = {
            'warm_interval_minutes': 60,
            'hit_rate_threshold': 0.8
        }
        
        maintenance_manager.update_config(new_config)
        
        assert maintenance_manager.config['warm_interval_minutes'] == 60
        assert maintenance_manager.config['hit_rate_threshold'] == 0.8
        assert maintenance_manager.config['optimize_interval_hours'] == 6  # Unchanged
    
    def test_singleton_manager(self):
        """Test singleton pattern for get_maintenance_manager"""
        manager1 = get_maintenance_manager()
        manager2 = get_maintenance_manager()
        
        assert manager1 is manager2


@pytest.mark.asyncio
class TestCacheMaintenanceIntegration:
    """Integration tests for cache maintenance"""
    
    async def test_maintenance_lifecycle(self):
        """Test full maintenance lifecycle"""
        with patch('services.cache_maintenance.get_cache_service'), \
             patch('services.cache_maintenance.get_fetcher'), \
             patch('services.cache_maintenance.EnhancedDataFetcher'):
            
            manager = get_maintenance_manager()
            
            # Start maintenance
            manager.start()
            assert manager.is_running
            
            # Wait a bit
            await asyncio.sleep(0.2)
            
            # Get status
            status = manager.get_status()
            assert status['is_running'] is True
            
            # Stop maintenance
            manager.stop()
            assert not manager.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])