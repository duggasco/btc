"""API Client for frontend-backend communication"""
import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

class APIClient:
    """Client for interacting with the backend API"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('API_BASE_URL', 'http://backend:8000')
        self.session = requests.Session()
        self.timeout = 30
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            data = response.json()
            # Ensure we always return a dict for consistency
            if isinstance(data, list):
                return {"data": data, "success": True}
            return data
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "success": False}
    
    def get_current_price(self) -> Dict:
        """Get current BTC price"""
        return self._make_request("GET", "/price/current")
    
    def get_latest_signal(self) -> Dict:
        """Get latest trading signal"""
        return self._make_request("GET", "/signals/enhanced/latest")
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent trading signals"""
        response = self._make_request("GET", f"/signals/history?limit={limit}")
        if isinstance(response, dict):
            return response.get("signals", response.get("data", []))
        return []
    
    def get_portfolio(self) -> Dict:
        """Get portfolio information"""
        return self._make_request("GET", "/portfolio/metrics")
    
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        response = self._make_request("GET", "/portfolio/positions")
        if isinstance(response, dict):
            return response.get("positions", response.get("data", []))
        return []
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades"""
        response = self._make_request("GET", f"/trades/recent?limit={limit}")
        if isinstance(response, dict):
            return response.get("trades", response.get("data", []))
        return []
    
    def get_historical_data(self, period: str = "24h") -> List[Dict]:
        """Get historical price data"""
        response = self._make_request("GET", f"/market/btc-data?period={period}")
        # Handle both direct data response and wrapped response
        if isinstance(response, dict) and "data" in response:
            return response.get("data", [])
        elif isinstance(response, list):
            return response
        return []
    
    def place_order(self, side: str, amount: float, order_type: str = "market", 
                   price: Optional[float] = None) -> Dict:
        """Place a trading order"""
        payload = {
            "side": side,
            "amount": amount,
            "order_type": order_type
        }
        
        if price and order_type == "limit":
            payload["price"] = price
        
        return self._make_request("POST", "/trades/execute", json=payload)
    
    def get_backtest_results(self, period: str = "30d") -> Dict:
        """Get backtest results"""
        return self._make_request("GET", "/backtest/results/latest")
    
    def run_backtest(self, config: Dict) -> Dict:
        """Run a new backtest"""
        return self._make_request("POST", "/backtest/enhanced/run", json=config)
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return self._make_request("GET", "/system/status")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return self._make_request("GET", "/model/info")
    
    def update_settings(self, settings: Dict) -> Dict:
        """Update system settings"""
        return self._make_request("POST", "/config/update", json=settings)
    
    def get_settings(self) -> Dict:
        """Get current settings"""
        return self._make_request("GET", "/config/current")

# Singleton instance
_api_client = None

def get_api_client() -> APIClient:
    """Get or create API client instance"""
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client