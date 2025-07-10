"""
WebSocket connection manager with page-specific cleanup
Handles singleton WebSocket connections and proper cleanup on page transitions
"""

import streamlit as st
import logging
from typing import Optional
from .websocket_client import EnhancedWebSocketClient
from config import WS_BASE_URL

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections with page tracking and cleanup
    
    Features:
    - Singleton WebSocket connection
    - Page transition detection
    - Automatic cleanup on page exit
    - Connection state persistence
    """
    
    @staticmethod
    def get_instance() -> 'WebSocketManager':
        """Get or create singleton WebSocketManager instance"""
        if 'websocket_manager' not in st.session_state:
            st.session_state.websocket_manager = WebSocketManager()
        return st.session_state.websocket_manager
    
    def __init__(self):
        """Initialize WebSocket manager"""
        self.client: Optional[EnhancedWebSocketClient] = None
        self.current_page: Optional[str] = None
        self.connection_count = 0
        
    def get_client(self, page_name: str) -> EnhancedWebSocketClient:
        """
        Get WebSocket client for a specific page
        
        Args:
            page_name: Name of the page requesting the client
            
        Returns:
            WebSocket client instance
        """
        # Detect page change
        if self.current_page and self.current_page != page_name:
            logger.info(f"Page transition detected: {self.current_page} -> {page_name}")
            self.cleanup_for_page(self.current_page)
        
        self.current_page = page_name
        
        # Create client if needed
        if self.client is None:
            logger.info(f"Creating new WebSocket client for {page_name}")
            self.client = EnhancedWebSocketClient(WS_BASE_URL)
            self.client.connect()
            self.connection_count += 1
            
            # Wait a moment for connection to establish
            import time
            max_wait = 5  # seconds
            start_time = time.time()
            while not self.client.is_connected() and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
            
            if self.client.is_connected():
                logger.info(f"WebSocket connected successfully for {page_name}")
            else:
                logger.error(f"WebSocket failed to connect for {page_name}")
        else:
            logger.debug(f"Reusing existing WebSocket client for {page_name}")
            # Check if connection is still alive
            if not self.client.is_connected():
                logger.warning(f"WebSocket disconnected, attempting to reconnect for {page_name}")
                self.client.connect()
        
        # Store page tracking in session state
        st.session_state['websocket_active_page'] = page_name
        
        return self.client
    
    def cleanup_for_page(self, page_name: str):
        """
        Clean up WebSocket resources for a specific page
        
        Args:
            page_name: Name of the page to clean up
        """
        logger.info(f"Cleaning up WebSocket for page: {page_name}")
        
        # Clear active page if it matches
        if st.session_state.get('websocket_active_page') == page_name:
            st.session_state['websocket_active_page'] = None
        
        # Only close client if no other page is using it
        if self.current_page == page_name:
            self.current_page = None
            
            # Check if any other page needs the connection
            active_page = st.session_state.get('websocket_active_page')
            if not active_page and self.client:
                logger.info("No active pages, closing WebSocket connection")
                self.close()
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.client is not None and self.client.is_connected()
    
    def get_stats(self) -> dict:
        """Get WebSocket connection statistics"""
        stats = {
            'current_page': self.current_page,
            'connection_count': self.connection_count,
            'is_connected': self.is_connected()
        }
        
        if self.client:
            stats.update(self.client.get_connection_stats())
            
        return stats
    
    def close(self):
        """Close WebSocket connection"""
        if self.client:
            logger.info("Closing WebSocket client")
            self.client.close()
            self.client = None


# Convenience functions for backward compatibility
def get_websocket_client(page_name: str = "default") -> EnhancedWebSocketClient:
    """
    Get WebSocket client for a specific page
    
    Args:
        page_name: Name of the page (used for tracking)
        
    Returns:
        WebSocket client instance
    """
    manager = WebSocketManager.get_instance()
    return manager.get_client(page_name)


def register_page(page_name: str):
    """
    Register a page as using WebSocket
    
    Args:
        page_name: Name of the page
    """
    logger.info(f"Registering WebSocket page: {page_name}")
    get_websocket_client(page_name)


def is_websocket_connected() -> bool:
    """Check if WebSocket is connected"""
    manager = WebSocketManager.get_instance()
    return manager.is_connected()


def cleanup_websocket(page_name: str):
    """
    Clean up WebSocket for a specific page
    
    Args:
        page_name: Name of the page
    """
    manager = WebSocketManager.get_instance()
    manager.cleanup_for_page(page_name)


def get_websocket_stats() -> dict:
    """Get WebSocket connection statistics"""
    manager = WebSocketManager.get_instance()
    return manager.get_stats()