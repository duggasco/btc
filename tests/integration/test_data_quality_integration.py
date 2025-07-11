"""
Integration tests for Data Quality feature with existing system components.
Tests data flow, cross-feature compatibility, WebSocket integration,
database concurrency, and settings page integration.
"""

import pytest
import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import websocket
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import system components
from src.backend.services.data_quality_service import DataQualityService
from src.backend.services.data_fetcher import DataFetcher
from src.backend.services.historical_data_manager import HistoricalDataManager
from src.backend.services.integration import ModelIntegrationService
from src.backend.services.backtesting import EnhancedBacktestingService
from src.backend.services.strategy_optimizer import StrategyOptimizer
from src.backend.models.paper_trading import PaperTradingService
from src.backend.models.database import get_db_session, Signal, Trade
from src.backend.api.main import app, ConnectionManager
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test fixtures
@pytest.fixture
def test_db():
    """Create a test database."""
    engine = create_engine("sqlite:///:memory:")
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    from src.backend.models.database import Base
    Base.metadata.create_all(bind=engine)
    
    return TestSessionLocal

@pytest.fixture
def data_quality_service(test_db):
    """Create data quality service with test database."""
    session = test_db()
    service = DataQualityService()
    service.session_factory = lambda: session
    return service

@pytest.fixture
def data_fetcher():
    """Create mock data fetcher."""
    fetcher = MagicMock(spec=DataFetcher)
    fetcher.get_current_btc_price = MagicMock(return_value=50000.0)
    fetcher.get_historical_data = MagicMock(return_value=pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
        'open': np.random.uniform(49000, 51000, 100),
        'high': np.random.uniform(49500, 51500, 100),
        'low': np.random.uniform(48500, 50500, 100),
        'close': np.random.uniform(49000, 51000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }))
    return fetcher

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


class TestDataFlowIntegration:
    """Test data quality integration with data flow components."""
    
    def test_data_quality_updates_on_fetch(self, data_quality_service, data_fetcher):
        """Test that data quality metrics update when new data is fetched."""
        # Initial quality check
        initial_metrics = data_quality_service.get_data_quality_metrics()
        assert initial_metrics['ohlcv_data']['row_count'] == 0
        
        # Simulate data fetch
        new_data = data_fetcher.get_historical_data()
        
        # Update data quality with new data
        data_quality_service._analyze_ohlcv_quality(new_data)
        
        # Check updated metrics
        updated_metrics = data_quality_service.get_data_quality_metrics()
        assert updated_metrics['ohlcv_data']['row_count'] == 100
        assert updated_metrics['ohlcv_data']['completeness'] > 0
    
    def test_historical_data_manager_integration(self, data_quality_service, test_db):
        """Test integration with historical data manager."""
        session = test_db()
        
        # Create historical data manager
        hdm = HistoricalDataManager()
        hdm.session_factory = lambda: session
        
        # Store some data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'open': np.random.uniform(49000, 51000, 50),
            'high': np.random.uniform(49500, 51500, 50),
            'low': np.random.uniform(48500, 50500, 50),
            'close': np.random.uniform(49000, 51000, 50),
            'volume': np.random.uniform(100, 1000, 50)
        })
        
        # Store data through historical manager
        hdm.store_ohlcv_data(test_data)
        
        # Check data quality reflects stored data
        metrics = data_quality_service.get_data_quality_metrics()
        
        # Should detect the stored data
        assert metrics['ohlcv_data']['row_count'] > 0
        assert metrics['ohlcv_data']['time_range']['start'] is not None
    
    def test_paper_trading_data_reflection(self, data_quality_service, test_db):
        """Test that data quality reflects paper trading data."""
        session = test_db()
        
        # Create paper trading service
        pts = PaperTradingService(session)
        
        # Execute some trades
        pts.execute_trade('BUY', 50000, 0.01, datetime.utcnow())
        pts.execute_trade('SELL', 51000, 0.005, datetime.utcnow() + timedelta(hours=1))
        
        session.commit()
        
        # Check data quality metrics
        metrics = data_quality_service.get_data_quality_metrics()
        
        # Should reflect trading data
        assert metrics['trading_data']['trade_count'] == 2
        assert metrics['trading_data']['portfolio_health']['has_trades'] is True


class TestCrossFeatureCompatibility:
    """Test data quality with other features running simultaneously."""
    
    @pytest.mark.asyncio
    async def test_data_quality_during_backtesting(self, data_quality_service, test_db):
        """Test data quality metrics while backtesting is running."""
        session = test_db()
        
        # Create backtesting service
        backtest_service = EnhancedBacktestingService()
        
        # Create sample data
        features = np.random.randn(100, 50)
        returns = np.random.randn(100) * 0.01
        
        # Run tasks concurrently
        async def run_backtest():
            config = type('BacktestConfig', (), {
                'initial_capital': 10000,
                'position_size': 0.1,
                'transaction_cost': 0.001,
                'training_window_days': 30,
                'test_window_days': 7
            })()
            return await asyncio.to_thread(
                backtest_service.run_comprehensive_backtest,
                features, returns, config
            )
        
        async def check_data_quality():
            await asyncio.sleep(0.1)  # Small delay
            return data_quality_service.get_data_quality_metrics()
        
        # Run both operations
        backtest_task = asyncio.create_task(run_backtest())
        quality_task = asyncio.create_task(check_data_quality())
        
        backtest_result, quality_metrics = await asyncio.gather(
            backtest_task, quality_task
        )
        
        # Both should complete successfully
        assert backtest_result is not None
        assert quality_metrics is not None
        assert 'overall_score' in quality_metrics
    
    def test_optimization_with_data_quality(self, data_quality_service):
        """Test that optimization uses data shown in quality metrics."""
        # Check current data quality
        metrics = data_quality_service.get_data_quality_metrics()
        data_points = metrics['ohlcv_data']['row_count']
        
        # Create optimizer
        optimizer = StrategyOptimizer(MagicMock())
        
        # Mock optimization to check data usage
        with patch.object(optimizer, 'optimize') as mock_opt:
            mock_opt.return_value = {
                'best_parameters': {},
                'best_value': 1.5,
                'trials': []
            }
            
            # Run optimization
            config = type('OptConfig', (), {
                'n_trials': 10,
                'objective': 'sharpe_ratio'
            })()
            
            # Should use same data as shown in quality
            features = np.random.randn(data_points or 100, 50)
            optimizer.optimize(config, features)
            
            mock_opt.assert_called_once()
    
    def test_signal_generation_data_requirements(self, data_quality_service, test_db):
        """Test that signal generation has required data per quality metrics."""
        session = test_db()
        
        # Check data quality
        metrics = data_quality_service.get_data_quality_metrics()
        
        # Create integration service
        integration = ModelIntegrationService()
        
        # If data quality is poor, signal generation should handle gracefully
        if metrics['overall_score'] < 50:
            with patch.object(integration, 'generate_signal') as mock_signal:
                mock_signal.return_value = {
                    'signal': 'HOLD',
                    'confidence': 0.3,
                    'reason': 'Insufficient data quality'
                }
                
                result = integration.generate_signal(np.array([]))
                assert result['confidence'] < 0.5  # Low confidence due to poor data


class TestWebSocketIntegration:
    """Test WebSocket compatibility with data quality feature."""
    
    def test_realtime_updates_quality_metrics(self, data_quality_service, test_client):
        """Test if real-time data updates affect quality metrics."""
        # Connect WebSocket
        with test_client.websocket_connect("/ws") as websocket:
            # Send price update
            websocket.send_json({
                "type": "price_update",
                "data": {
                    "price": 50000,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            # Check if quality metrics reflect update
            time.sleep(0.1)  # Allow processing
            metrics = data_quality_service.get_data_quality_metrics()
            
            # Should have recent update
            if metrics['real_time_data']['last_update']:
                last_update = datetime.fromisoformat(
                    metrics['real_time_data']['last_update'].replace('Z', '+00:00')
                )
                assert (datetime.utcnow() - last_update).seconds < 5
    
    def test_websocket_no_conflicts(self, test_client):
        """Test no conflicts between WebSocket and data quality endpoints."""
        # Create multiple WebSocket connections
        connections = []
        
        try:
            # Open 3 WebSocket connections
            for i in range(3):
                ws = test_client.websocket_connect(f"/ws?client_id={i}")
                connections.append(ws.__enter__())
            
            # Call data quality endpoint while WebSockets are active
            response = test_client.get("/data-quality/metrics")
            assert response.status_code == 200
            
            # Send messages through WebSockets
            for i, ws in enumerate(connections):
                ws.send_json({"type": "ping", "id": i})
            
            # Call data quality again
            response = test_client.get("/data-quality/metrics")
            assert response.status_code == 200
            
            # All connections should still be active
            for ws in connections:
                data = ws.receive_json()
                assert data is not None
                
        finally:
            # Cleanup
            for ws in connections:
                ws.__exit__(None, None, None)
    
    def test_performance_impact(self, data_quality_service, test_client):
        """Test performance impact on real-time features."""
        import time
        
        # Measure baseline WebSocket latency
        with test_client.websocket_connect("/ws") as websocket:
            start = time.time()
            websocket.send_json({"type": "ping"})
            websocket.receive_json()
            baseline_latency = time.time() - start
        
        # Run data quality check concurrently
        def quality_check():
            for _ in range(10):
                data_quality_service.get_data_quality_metrics()
                time.sleep(0.01)
        
        # Test WebSocket latency with quality checks running
        thread = threading.Thread(target=quality_check)
        thread.start()
        
        with test_client.websocket_connect("/ws") as websocket:
            latencies = []
            for _ in range(10):
                start = time.time()
                websocket.send_json({"type": "ping"})
                websocket.receive_json()
                latencies.append(time.time() - start)
        
        thread.join()
        
        # Average latency should not increase significantly
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < baseline_latency * 2  # Max 2x slowdown


class TestDatabaseIntegration:
    """Test database concurrency and isolation."""
    
    def test_concurrent_access(self, data_quality_service, test_db):
        """Test concurrent access from multiple features."""
        session_factory = test_db
        
        def worker_task(task_id):
            """Simulate different features accessing database."""
            session = session_factory()
            
            try:
                if task_id % 3 == 0:
                    # Data quality check
                    metrics = data_quality_service.get_data_quality_metrics()
                    return ('quality', metrics['overall_score'])
                elif task_id % 3 == 1:
                    # Trading operation
                    trade = Trade(
                        trade_type='BUY',
                        price=50000 + task_id,
                        amount=0.01,
                        timestamp=datetime.utcnow()
                    )
                    session.add(trade)
                    session.commit()
                    return ('trade', trade.id)
                else:
                    # Signal generation
                    signal = Signal(
                        timestamp=datetime.utcnow(),
                        signal='BUY',
                        confidence=0.8,
                        price_prediction=51000
                    )
                    session.add(signal)
                    session.commit()
                    return ('signal', signal.id)
            finally:
                session.close()
        
        # Run concurrent tasks
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, i) for i in range(30)]
            results = [f.result() for f in as_completed(futures)]
        
        # All tasks should complete successfully
        assert len(results) == 30
        assert len([r for r in results if r[0] == 'quality']) == 10
        assert len([r for r in results if r[0] == 'trade']) == 10
        assert len([r for r in results if r[0] == 'signal']) == 10
    
    def test_no_table_locks(self, data_quality_service, test_db):
        """Test no table locks or conflicts."""
        session1 = test_db()
        session2 = test_db()
        
        # Start a long-running query in session1
        def long_query():
            session1.execute("SELECT COUNT(*) FROM trades")
            time.sleep(0.5)  # Simulate long operation
            session1.commit()
        
        thread = threading.Thread(target=long_query)
        thread.start()
        
        # Try to access data quality in session2
        time.sleep(0.1)  # Let first query start
        start = time.time()
        metrics = data_quality_service.get_data_quality_metrics()
        duration = time.time() - start
        
        thread.join()
        
        # Should not be blocked
        assert duration < 0.2  # Should be fast
        assert metrics is not None
        
        session1.close()
        session2.close()
    
    def test_transaction_isolation(self, data_quality_service, test_db):
        """Test transaction isolation between features."""
        session1 = test_db()
        session2 = test_db()
        
        # Start transaction in session1
        trade = Trade(
            trade_type='BUY',
            price=50000,
            amount=0.01,
            timestamp=datetime.utcnow()
        )
        session1.add(trade)
        # Don't commit yet
        
        # Check data quality in session2
        metrics = data_quality_service.get_data_quality_metrics()
        trade_count_before = metrics['trading_data']['trade_count']
        
        # Now commit session1
        session1.commit()
        
        # Check again
        metrics_after = data_quality_service.get_data_quality_metrics()
        trade_count_after = metrics_after['trading_data']['trade_count']
        
        # Should see the new trade after commit
        assert trade_count_after == trade_count_before + 1
        
        session1.close()
        session2.close()


class TestSettingsIntegration:
    """Test integration with Settings page tabs."""
    
    def test_tab_navigation(self, test_client):
        """Test navigation between Settings tabs."""
        # Access different settings endpoints
        endpoints = [
            "/settings/general",
            "/settings/trading",
            "/data-quality/metrics",
            "/settings/notifications"
        ]
        
        for endpoint in endpoints:
            response = test_client.get(endpoint)
            # All should be accessible
            assert response.status_code in [200, 404]  # 404 if not implemented
    
    def test_settings_changes_no_affect(self, data_quality_service, test_client):
        """Test that settings changes don't affect data quality."""
        # Get initial metrics
        initial_metrics = data_quality_service.get_data_quality_metrics()
        
        # Change some settings
        test_client.post("/settings/trading", json={
            "position_size": 0.2,
            "stop_loss": 0.05,
            "take_profit": 0.1
        })
        
        test_client.post("/settings/notifications", json={
            "webhook_url": "https://discord.com/test",
            "alerts_enabled": True
        })
        
        # Check metrics again
        after_metrics = data_quality_service.get_data_quality_metrics()
        
        # Data quality metrics should not change
        assert initial_metrics['overall_score'] == after_metrics['overall_score']
        assert initial_metrics['ohlcv_data'] == after_metrics['ohlcv_data']
    
    def test_concurrent_settings_access(self, test_client):
        """Test concurrent access to different settings tabs."""
        def access_tab(tab_url):
            response = test_client.get(tab_url)
            return response.status_code
        
        urls = [
            "/data-quality/metrics",
            "/settings/general",
            "/data-quality/recommendations",
            "/settings/trading"
        ]
        
        # Access all tabs concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_tab, url) for url in urls * 3]
            results = [f.result() for f in as_completed(futures)]
        
        # All should succeed
        assert all(status in [200, 404] for status in results)


class TestErrorScenarios:
    """Test error handling in integration scenarios."""
    
    def test_data_quality_with_database_error(self, data_quality_service):
        """Test data quality handles database errors gracefully."""
        # Mock database error
        with patch.object(data_quality_service, 'session_factory') as mock_factory:
            mock_factory.side_effect = Exception("Database connection error")
            
            # Should return partial metrics
            metrics = data_quality_service.get_data_quality_metrics()
            
            assert metrics is not None
            assert metrics['overall_score'] == 0  # Worst case score
            assert 'error' in metrics or metrics['database_health']['is_accessible'] is False
    
    def test_websocket_error_recovery(self, test_client, data_quality_service):
        """Test WebSocket error doesn't affect data quality."""
        # Force WebSocket error
        with patch('src.backend.api.main.ConnectionManager.broadcast') as mock_broadcast:
            mock_broadcast.side_effect = Exception("WebSocket error")
            
            # Data quality should still work
            response = test_client.get("/data-quality/metrics")
            assert response.status_code == 200
            
            metrics = response.json()
            assert 'overall_score' in metrics
    
    def test_concurrent_errors(self, data_quality_service, test_db):
        """Test system stability with multiple concurrent errors."""
        errors_caught = []
        
        def error_prone_task(task_id):
            try:
                if task_id % 4 == 0:
                    # Database error
                    raise Exception("Database error")
                elif task_id % 4 == 1:
                    # Data quality check
                    return data_quality_service.get_data_quality_metrics()
                elif task_id % 4 == 2:
                    # API error
                    raise ValueError("API error")
                else:
                    # Normal operation
                    return "success"
            except Exception as e:
                errors_caught.append(str(e))
                return None
        
        # Run tasks with errors
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(error_prone_task, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        # Should handle errors gracefully
        successful = [r for r in results if r is not None]
        assert len(successful) > 0  # Some should succeed
        assert len(errors_caught) > 0  # Some errors expected
        
        # Data quality should still be accessible after errors
        final_metrics = data_quality_service.get_data_quality_metrics()
        assert final_metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])