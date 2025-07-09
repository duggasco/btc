"""
Comprehensive tests for the SQLite-based cache service
"""
import pytest
import pandas as pd
import numpy as np
import time
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the modules to test
import sys
sys.path.insert(0, '/root/btc/src/backend')

from services.cache_service import CacheService, get_cache_service
from services.cache_integration import (
    cached_api_call, CachedDataFetcher, generate_cache_key,
    determine_ttl, batch_cache_get, batch_cache_set,
    warm_cache, get_cache_info
)


class TestCacheService:
    """Test the core CacheService functionality"""
    
    @pytest.fixture
    def cache_service(self):
        """Create a test cache service with temporary database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            cache = CacheService(db_path=tmp.name)
            yield cache
            # Cleanup
            os.unlink(tmp.name)
    
    def test_cache_initialization(self, cache_service):
        """Test cache database initialization"""
        assert cache_service is not None
        assert os.path.exists(cache_service.db_path)
        
        # Check tables were created
        import sqlite3
        conn = sqlite3.connect(cache_service.db_path)
        cursor = conn.cursor()
        
        # Check api_cache table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_cache'")
        assert cursor.fetchone() is not None
        
        # Check indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cache_key'")
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_set_and_get(self, cache_service):
        """Test basic set and get operations"""
        # Test simple data
        key = "test_key"
        data = {"price": 100.5, "volume": 1000}
        
        # Set cache
        success = cache_service.set(key, data, data_type='price', api_source='test')
        assert success is True
        
        # Get cache
        cached_data = cache_service.get(key)
        assert cached_data == data
        
        # Check hit stats
        assert cache_service.stats['hits'] == 1
        assert cache_service.stats['misses'] == 0
    
    def test_ttl_expiration(self, cache_service):
        """Test TTL expiration"""
        key = "ttl_test"
        data = "test_data"
        
        # Set with 1 second TTL
        cache_service.set(key, data, ttl=1)
        
        # Should get data immediately
        assert cache_service.get(key) == data
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should return None after expiration
        assert cache_service.get(key) is None
        assert cache_service.stats['misses'] == 1
    
    def test_dataframe_serialization(self, cache_service):
        """Test DataFrame serialization and deserialization"""
        key = "df_test"
        df = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'timestamp': pd.date_range('2024-01-01', periods=3)
        })
        
        # Set DataFrame
        success = cache_service.set(key, df, data_type='dataframe')
        assert success is True
        
        # Get DataFrame
        cached_df = cache_service.get(key)
        assert isinstance(cached_df, pd.DataFrame)
        pd.testing.assert_frame_equal(cached_df, df)
    
    def test_batch_operations(self, cache_service):
        """Test batch get and set operations"""
        # Prepare batch data
        entries = {
            'key1': ({'data': 1}, {'data_type': 'test', 'api_source': 'batch'}),
            'key2': ({'data': 2}, {'data_type': 'test', 'api_source': 'batch'}),
            'key3': ({'data': 3}, {'data_type': 'test', 'api_source': 'batch'})
        }
        
        # Batch set
        results = cache_service.batch_set(entries)
        assert all(results.values())
        
        # Batch get
        keys = list(entries.keys())
        batch_data = cache_service.batch_get(keys)
        
        assert batch_data['key1'] == {'data': 1}
        assert batch_data['key2'] == {'data': 2}
        assert batch_data['key3'] == {'data': 3}
    
    def test_invalidation(self, cache_service):
        """Test cache invalidation"""
        # Add multiple entries
        cache_service.set('test_price_1', 100, data_type='price')
        cache_service.set('test_price_2', 200, data_type='price')
        cache_service.set('test_volume_1', 1000, data_type='volume')
        
        # Invalidate by pattern
        removed = cache_service.invalidate(pattern='price')
        assert removed == 2
        
        # Verify entries removed
        assert cache_service.get('test_price_1') is None
        assert cache_service.get('test_price_2') is None
        assert cache_service.get('test_volume_1') == 1000
    
    def test_cache_stats(self, cache_service):
        """Test cache statistics"""
        # Generate some activity
        cache_service.set('key1', 'data1')
        cache_service.set('key2', 'data2')
        cache_service.get('key1')  # Hit
        cache_service.get('key1')  # Hit
        cache_service.get('missing')  # Miss
        
        stats = cache_service.get_stats()
        
        assert stats['total_entries'] == 2
        assert stats['session_stats']['hits'] == 2
        assert stats['session_stats']['misses'] == 1
        assert stats['session_stats']['hit_rate'] == 2/3
    
    def test_detailed_stats(self, cache_service):
        """Test detailed statistics"""
        # Add various types of data
        cache_service.set('btc_price', 100000, data_type='real_time_price', api_source='binance')
        cache_service.set('eth_price', 3000, data_type='real_time_price', api_source='binance')
        cache_service.set('btc_ohlcv', {'o': 100, 'h': 101, 'l': 99, 'c': 100.5}, 
                         data_type='ohlcv', api_source='coingecko')
        
        detailed_stats = cache_service.get_detailed_stats()
        
        assert 'age_distribution' in detailed_stats
        assert 'hit_rates_by_type' in detailed_stats
        assert 'api_performance' in detailed_stats
        assert 'cache_efficiency' in detailed_stats
    
    def test_optimization(self, cache_service):
        """Test cache optimization"""
        # Add entries with various characteristics
        for i in range(10):
            cache_service.set(f'zero_hit_{i}', f'data_{i}')
        
        # Add large entry with low hits
        large_data = 'x' * (2 * 1024 * 1024)  # 2MB
        cache_service.set('large_entry', large_data)
        
        # Wait to make entries old
        time.sleep(1)
        
        # Run optimization
        report = cache_service.optimize_cache()
        
        assert 'actions' in report
        assert 'final_stats' in report
        assert report['final_stats']['entries_removed'] > 0
    
    def test_metrics_export(self, cache_service):
        """Test metrics export functionality"""
        # Add some data
        cache_service.set('test_key', 'test_data')
        
        # Export JSON
        json_metrics = cache_service.export_metrics(format='json')
        assert json_metrics is not None
        parsed = json.loads(json_metrics)
        assert 'total_entries' in parsed
        
        # Export Prometheus
        prom_metrics = cache_service.export_metrics(format='prometheus')
        assert prom_metrics is not None
        assert 'btc_cache_total_entries' in prom_metrics


class TestCacheIntegration:
    """Test the cache integration module"""
    
    def test_generate_cache_key(self):
        """Test cache key generation"""
        # Simple case
        key = generate_cache_key('binance', 'fetch', 'BTC', '1d')
        assert key == 'binance:fetch:BTC:1d'
        
        # With kwargs
        key = generate_cache_key('coingecko', 'get_price', 'bitcoin', interval='1h', limit=100)
        assert 'coingecko:get_price:bitcoin' in key
        assert 'interval=1h' in key
        assert 'limit=100' in key
        
        # Long key should be hashed
        long_args = ['x' * 50 for _ in range(10)]
        key = generate_cache_key('source', 'method', *long_args)
        assert len(key) < 100  # Should be shortened
    
    def test_determine_ttl(self):
        """Test TTL determination logic"""
        # Real-time price
        assert determine_ttl('real_time_price') == 30
        
        # OHLCV with period
        assert determine_ttl('ohlcv', '1m') == 30
        assert determine_ttl('ohlcv', '1h') == 300
        assert determine_ttl('ohlcv', '1d') == 900
        
        # Other types
        assert determine_ttl('sentiment') == 1800
        assert determine_ttl('macro_indicators') == 3600
        
        # Default
        assert determine_ttl('unknown_type') == 300
    
    def test_cached_api_call_decorator(self):
        """Test the cached_api_call decorator"""
        # Create a mock data source
        class MockDataSource:
            def __init__(self):
                self.name = 'mock_source'
                self.call_count = 0
            
            @cached_api_call(data_type='test', ttl=60)
            def fetch_data(self, symbol):
                self.call_count += 1
                return {'symbol': symbol, 'price': 100}
        
        # Create temporary cache
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Override get_cache_service to use our test cache
            test_cache = CacheService(db_path=tmp.name)
            with patch('services.cache_integration.get_cache_service', return_value=test_cache):
                source = MockDataSource()
                
                # First call should hit the API
                result1 = source.fetch_data('BTC')
                assert result1 == {'symbol': 'BTC', 'price': 100}
                assert source.call_count == 1
                
                # Second call should hit the cache
                result2 = source.fetch_data('BTC')
                assert result2 == result1
                assert source.call_count == 1  # No additional API call
                
                # Different symbol should hit API
                result3 = source.fetch_data('ETH')
                assert result3 == {'symbol': 'ETH', 'price': 100}
                assert source.call_count == 2
            
            os.unlink(tmp.name)
    
    def test_cached_data_fetcher_base_class(self):
        """Test the CachedDataFetcher base class"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_cache = CacheService(db_path=tmp.name)
            
            class TestFetcher(CachedDataFetcher):
                def __init__(self):
                    super().__init__(cache_service=test_cache)
                    self.name = 'test_fetcher'
            
            fetcher = TestFetcher()
            
            # Test cache operations
            assert fetcher._cache_enabled is True
            
            # Disable cache
            fetcher.disable_cache()
            assert fetcher._cache_enabled is False
            
            # Enable cache
            fetcher.enable_cache()
            assert fetcher._cache_enabled is True
            
            # Get stats
            stats = fetcher.get_cache_stats()
            assert 'total_entries' in stats
            
            os.unlink(tmp.name)
    
    def test_batch_operations_integration(self):
        """Test batch cache operations"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_cache = CacheService(db_path=tmp.name)
            
            with patch('services.cache_integration.get_cache_service', return_value=test_cache):
                # Test batch set
                items = {
                    'key1': ({'data': 1}, 'test', 60),
                    'key2': ({'data': 2}, 'test', 60),
                    'key3': ({'data': 3}, 'test', 60)
                }
                
                results = batch_cache_set(items)
                assert all(results.values())
                
                # Test batch get
                keys = list(items.keys())
                data = batch_cache_get(keys)
                
                assert data['key1'] == {'data': 1}
                assert data['key2'] == {'data': 2}
                assert data['key3'] == {'data': 3}
            
            os.unlink(tmp.name)
    
    def test_cache_info(self):
        """Test cache info function"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_cache = CacheService(db_path=tmp.name)
            
            # Add some test data
            test_cache.set('price_1', 100, data_type='price', api_source='binance')
            test_cache.set('ohlcv_1', {'o': 100}, data_type='ohlcv', api_source='coingecko')
            
            with patch('services.cache_integration.get_cache_service', return_value=test_cache):
                info = get_cache_info()
                
                assert 'data_type_distribution' in info
                assert 'api_source_distribution' in info
                assert len(info['data_type_distribution']) > 0
                assert len(info['api_source_distribution']) > 0
            
            os.unlink(tmp.name)


class TestCacheAPIEndpoints:
    """Test cache management API endpoints"""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app with cache endpoints"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Create minimal app with cache endpoints
        app = FastAPI()
        
        # Import cache endpoint functions
        from api.main import (
            get_cache_stats, get_cache_entries, invalidate_cache,
            clear_expired_cache, optimize_cache, export_cache_metrics,
            get_cache_information
        )
        
        # Register endpoints
        app.get("/cache/stats")(get_cache_stats)
        app.get("/cache/entries")(get_cache_entries)
        app.post("/cache/invalidate")(invalidate_cache)
        app.post("/cache/clear-expired")(clear_expired_cache)
        app.post("/cache/optimize")(optimize_cache)
        app.get("/cache/metrics/{format}")(export_cache_metrics)
        app.get("/cache/info")(get_cache_information)
        
        return TestClient(app)
    
    def test_cache_stats_endpoint(self, test_app):
        """Test /cache/stats endpoint"""
        response = test_app.get("/cache/stats")
        
        # Should work even with empty cache
        assert response.status_code == 200
        data = response.json()
        
        assert 'total_entries' in data
        assert 'session_stats' in data
        assert response.headers.get('X-Cache-Status') == 'active'
    
    def test_cache_entries_endpoint(self, test_app):
        """Test /cache/entries endpoint"""
        # Test without filters
        response = test_app.get("/cache/entries")
        assert response.status_code == 200
        data = response.json()
        
        assert 'entries' in data
        assert 'count' in data
        assert 'filters' in data
        
        # Test with filters
        response = test_app.get("/cache/entries?data_type=price&limit=10")
        assert response.status_code == 200
        data = response.json()
        
        assert data['filters']['data_type'] == 'price'
        assert data['filters']['limit'] == 10
    
    def test_cache_invalidate_endpoint(self, test_app):
        """Test /cache/invalidate endpoint"""
        response = test_app.post("/cache/invalidate", params={
            'pattern': 'test',
            'data_type': 'price',
            'reason': 'Testing'
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'entries_removed' in data
        assert data['pattern'] == 'test'
        assert data['data_type'] == 'price'
        assert data['reason'] == 'Testing'
    
    def test_cache_clear_expired_endpoint(self, test_app):
        """Test /cache/clear-expired endpoint"""
        response = test_app.post("/cache/clear-expired")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'entries_removed' in data
        assert 'timestamp' in data
    
    def test_cache_optimize_endpoint(self, test_app):
        """Test /cache/optimize endpoint"""
        response = test_app.post("/cache/optimize")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return optimization report
        assert 'started_at' in data or 'error' in data
        if 'started_at' in data:
            assert 'actions' in data
            assert 'final_stats' in data
    
    def test_cache_metrics_endpoint(self, test_app):
        """Test /cache/metrics endpoint"""
        # Test JSON format
        response = test_app.get("/cache/metrics/json")
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'
        
        # Test Prometheus format
        response = test_app.get("/cache/metrics/prometheus")
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/plain; charset=utf-8'
        assert 'btc_cache_total_entries' in response.text
        
        # Test invalid format
        response = test_app.get("/cache/metrics/invalid")
        assert response.status_code == 400
    
    def test_cache_info_endpoint(self, test_app):
        """Test /cache/info endpoint"""
        response = test_app.get("/cache/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have comprehensive info
        assert 'total_entries' in data
        assert 'data_type_distribution' in data
        assert 'api_source_distribution' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])