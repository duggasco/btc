
import requests
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache
import time
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class APIClient:
    """Enhanced API client with caching, retry logic, and batch requests"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._cache = {}
        self._cache_ttl = 60  # Cache TTL in seconds
        self._request_history = []
        self._rate_limit_remaining = 100
        self._rate_limit_reset = datetime.now()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "BTC-Trading-System/2.0"
        })
        
    def _check_rate_limit(self) -> bool:
        """Check if we are within rate limits"""
        if datetime.now() > self._rate_limit_reset:
            self._rate_limit_remaining = 100
            self._rate_limit_reset = datetime.now() + timedelta(minutes=1)
            
        return self._rate_limit_remaining > 0
        
    def _update_rate_limit(self, headers: Dict):
        """Update rate limit info from response headers"""
        if "X-RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in headers:
            self._rate_limit_reset = datetime.fromtimestamp(int(headers["X-RateLimit-Reset"]))
            
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic and caching"""
        url = f"{self.base_url}{endpoint}"
        
        # Check rate limit
        if not self._check_rate_limit():
            wait_time = (self._rate_limit_reset - datetime.now()).total_seconds()
            logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Check cache for GET requests
        if method == "GET":
            cache_key = f"{url}:{json.dumps(kwargs.get("params", {}), sort_keys=True)}"
            cached_data = self._get_cached(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Record request
        request_start = time.time()
        
        retries = 3
        for attempt in range(retries):
            try:
                response = self.session.request(
                    method, 
                    url, 
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Update rate limit info
                self._update_rate_limit(response.headers)
                
                response.raise_for_status()
                
                data = response.json()
                
                # Record successful request
                self._request_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "duration": time.time() - request_start
                })
                
                # Cache successful GET responses
                if method == "GET" and response.status_code == 200:
                    self._set_cached(cache_key, data)
                
                return data
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1}/{retries})")
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error (attempt {attempt + 1}/{retries})")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                elif e.response.status_code >= 500:  # Server error
                    logger.error(f"Server error {e.response.status_code} (attempt {attempt + 1}/{retries})")
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                    return None
            except ValueError as e:
                logger.error(f"Invalid JSON response: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
                
            # Exponential backoff
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        return None
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for {key}")
                return data
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, data: Dict[str, Any]):
        """Set cached data with timestamp"""
        self._cache[key] = (data, time.time())
        
        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
            for old_key, _ in sorted_cache[:100]:
                del self._cache[old_key]
    
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
    
    def batch_get(self, endpoints: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Make multiple GET requests efficiently"""
        results = {}
        for endpoint in endpoints:
            results[endpoint] = self.get(endpoint)
        return results
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        logger.info("API cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API client statistics"""
        recent_requests = self._request_history[-100:]  # Last 100 requests
        
        if recent_requests:
            avg_duration = sum(r["duration"] for r in recent_requests) / len(recent_requests)
            success_rate = sum(1 for r in recent_requests if r["status_code"] < 400) / len(recent_requests)
        else:
            avg_duration = 0
            success_rate = 0
            
        return {
            "cache_size": len(self._cache),
            "total_requests": len(self._request_history),
            "recent_requests": len(recent_requests),
            "avg_response_time": avg_duration,
            "success_rate": success_rate,
            "rate_limit_remaining": self._rate_limit_remaining,
            "rate_limit_reset": self._rate_limit_reset.isoformat()
        }
    
    def health_check(self) -> bool:
        """Check if API is accessible"""
        try:
            response = self.get("/health")
            return response is not None and response.get("status") == "healthy"
        except Exception:
            return False

