"""
Simple WebSocket client for Streamlit that avoids threading issues
"""
import websocket
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

class SimpleWebSocketClient:
    """Simple WebSocket client without background threads for Streamlit compatibility"""
    
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.messages = []
        self.last_ping = datetime.now()
        
    def connect(self):
        """Connect to WebSocket server"""
        try:
            if self.ws:
                self.close()
            
            self.ws = websocket.create_connection(self.url, timeout=5)
            logger.info(f"WebSocket connected to {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        if not self.ws:
            return False
        
        try:
            # Send ping if needed
            if (datetime.now() - self.last_ping).total_seconds() > 30:
                self.ping()
            return True
        except:
            self.ws = None
            return False
    
    def ping(self):
        """Send ping message"""
        try:
            if self.ws:
                self.ws.send(json.dumps({"action": "ping"}))
                self.last_ping = datetime.now()
                # Try to receive pong
                self.ws.settimeout(0.1)
                try:
                    response = self.ws.recv()
                    data = json.loads(response)
                    if data.get("type") == "pong":
                        logger.debug("Received pong")
                except:
                    pass
                self.ws.settimeout(5)
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            self.ws = None
    
    def subscribe(self, channel: str):
        """Subscribe to a channel"""
        try:
            if self.ws:
                msg = {
                    "action": f"subscribe_{channel}",
                    "channel": channel,
                    "timestamp": datetime.now().isoformat()
                }
                self.ws.send(json.dumps(msg))
                logger.info(f"Subscribed to channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            self.ws = None
    
    def get_messages(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get available messages without blocking"""
        messages = []
        
        if not self.ws:
            return messages
        
        try:
            # Set non-blocking timeout
            self.ws.settimeout(0.01)
            
            for _ in range(max_messages):
                try:
                    msg = self.ws.recv()
                    data = json.loads(msg)
                    messages.append(data)
                except websocket._exceptions.WebSocketTimeoutException:
                    break
                except json.JSONDecodeError:
                    logger.warning(f"Invalid message: {msg}")
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    self.ws = None
                    break
            
            # Reset timeout
            if self.ws:
                self.ws.settimeout(5)
                
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            self.ws = None
            
        return messages
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.ws = None
            logger.info("WebSocket closed")


def get_simple_websocket_client():
    """Get or create simple WebSocket client in session state"""
    try:
        if 'simple_websocket_client' not in st.session_state:
            from config import WS_BASE_URL
            client = SimpleWebSocketClient(WS_BASE_URL)
            if client.connect():
                st.session_state.simple_websocket_client = client
            else:
                return None
        
        client = st.session_state.simple_websocket_client
        
        # Check connection and reconnect if needed
        if not client.is_connected():
            logger.info("WebSocket disconnected, reconnecting...")
            if not client.connect():
                return None
        
        return client
    except Exception as e:
        logger.error(f"Error getting WebSocket client: {e}")
        # Fallback to creating a new client without session state
        from config import WS_BASE_URL
        client = SimpleWebSocketClient(WS_BASE_URL)
        if client.connect():
            return client
        return None