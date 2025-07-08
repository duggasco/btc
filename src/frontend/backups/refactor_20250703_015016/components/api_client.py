import requests
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class APIClient:
    """Enhanced API client with caching and retry logic"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self._cache = {}
        self._cache_ttl = 60  # Cache TTL in seconds
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        # Check cache for GET requests
        if method == "GET":
            cache_key = f"{url}:{kwargs.get('params', {})}"
            cached_data = self._get_cached(cache_key)
            if cached_data is not None:
                return cached_data
        
        retries = 3
        for attempt in range(retries):
            try:
                response = self.session.request(
                    method, 
                    url, 
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Cache successful GET responses
                if method == "GET":
                    self._set_cached(cache_key, data)
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
            except ValueError as e:
                logger.error(f"Invalid JSON response: {e}")
                return None
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, data: Dict[str, Any]):
        """Set cached data with timestamp"""
        self._cache[key] = (data, time.time())
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Make GET request"""
        return self._make_request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Make POST request"""
        return self._make_request("POST", endpoint, json=data)
    
    def put(self, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Make PUT request"""
        return self._make_request("PUT", endpoint, json=data)
    
    def delete(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make DELETE request"""
        return self._make_request("DELETE", endpoint)
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
