
import websocket
import threading
import queue
import json
import logging
import time
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from config import WS_RECONNECT_INTERVAL

logger = logging.getLogger(__name__)

class EnhancedWebSocketClient:
    """Enhanced WebSocket client with auto-reconnection and event handling"""
    
    def __init__(self, url: str, reconnect_interval: int = WS_RECONNECT_INTERVAL):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.ws = None
        self.message_queue = queue.Queue()
        self.running = False
        self.connected = False
        self.callbacks = {}
        self.subscriptions = set()
        self.connection_start = None
        self.messages_received = 0
        
        # Message type handlers
        self.on_price_update = None
        self.on_signal_update = None
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.messages_received += 1
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
            
            self.message_queue.put(data)
            
            # Handle callbacks for specific message types
            msg_type = data.get("type", "unknown")
            if msg_type in self.callbacks:
                try:
                    self.callbacks[msg_type](data)
                except Exception as e:
                    logger.error(f"Callback error for {msg_type}: {e}")
            
            # Handle message type specific handlers
            if msg_type == "price_update" and self.on_price_update:
                try:
                    self.on_price_update(data.get("data", {}))
                except Exception as e:
                    logger.error(f"Price update handler error: {e}")
            elif msg_type == "signal_update" and self.on_signal_update:
                try:
                    self.on_signal_update(data.get("data", {}))
                except Exception as e:
                    logger.error(f"Signal update handler error: {e}")
                    
        except json.JSONDecodeError:
            logger.warning(f"Invalid WebSocket message: {message}")
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        logger.error(f"WebSocket error type: {type(error)}")
        logger.error(f"WebSocket state - running: {self.running}, connected: {self.connected}")
        self.connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Auto-reconnect if still running
        if self.running:
            logger.info(f"Auto-reconnecting in {self.reconnect_interval} seconds...")
            time.sleep(self.reconnect_interval)
            if self.running:  # Check again after sleep
                self.connect()
            
    def on_open(self, ws):
        logger.info("WebSocket connected")
        self.connected = True
        self.connection_start = datetime.now()
        
        # Re-subscribe to all channels
        for channel in self.subscriptions:
            self._send_subscription(channel)
        
        # Send initial ping
        self.ping()
    
    def connect(self):
        """Connect to WebSocket server with auto-reconnection"""
        try:
            # Don't connect if already running
            if self.running and self.connected:
                logger.warning("WebSocket already connected, skipping connection attempt")
                return
            
            # Clear any existing WebSocket
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None
            
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.running = True
            wst = threading.Thread(target=self._run_forever, daemon=True, name=f"WebSocket-{id(self)}")
            wst.start()
            
            logger.info(f"WebSocket client started for {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            logger.error(f"Exception type: {type(e)}")
            self.connected = False
            self.running = False
    
    def _run_forever(self):
        """Run WebSocket connection in a loop with auto-reconnection"""
        while self.running:
            try:
                if self.ws:
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                else:
                    logger.error("WebSocket object is None in _run_forever")
                    break
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
                logger.error(f"WebSocket run error type: {type(e)}")
                if self.running:
                    logger.info(f"Waiting {self.reconnect_interval} seconds before reconnecting...")
                    time.sleep(self.reconnect_interval)
    
    def _send_subscription(self, channel: str):
        """Send subscription message for a channel"""
        if self.connected and self.ws:
            msg = {
                "action": f"subscribe_{channel}",
                "channel": channel,
                "timestamp": datetime.now().isoformat()
            }
            self.ws.send(json.dumps(msg))
            logger.info(f"Subscribed to channel: {channel}")
    
    def subscribe(self, channel: str):
        """Subscribe to a specific channel"""
        self.subscriptions.add(channel)
        if self.connected:
            self._send_subscription(channel)
    
    def unsubscribe(self, channel: str):
        """Unsubscribe from a specific channel"""
        self.subscriptions.discard(channel)
        if self.connected and self.ws:
            msg = {
                "action": f"unsubscribe_{channel}",
                "channel": channel,
                "timestamp": datetime.now().isoformat()
            }
            self.ws.send(json.dumps(msg))
    
    def send_message(self, message: Dict[str, Any]):
        """Send a message to the WebSocket server"""
        if self.connected and self.ws:
            try:
                if "timestamp" not in message:
                    message["timestamp"] = datetime.now().isoformat()
                self.ws.send(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                self.connected = False
                return False
        logger.warning("Cannot send message - WebSocket not connected")
        return False
    
    def ping(self):
        """Send ping message"""
        return self.send_message({"action": "ping"})
    
    def register_callback(self, msg_type: str, callback: Callable):
        """Register a callback for specific message types"""
        self.callbacks[msg_type] = callback
        logger.info(f"Registered callback for message type: {msg_type}")
    
    def get_messages(self, max_messages: int = 100) -> List[Dict[str, Any]]:
        """Get all pending messages from the queue"""
        messages = []
        for _ in range(max_messages):
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def get_latest_message(self, msg_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the latest message of a specific type"""
        messages = self.get_messages()
        if msg_type:
            typed_messages = [m for m in messages if m.get("type") == msg_type]
            return typed_messages[-1] if typed_messages else None
        return messages[-1] if messages else None
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connected and self.ws is not None
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = None
        if self.connected and self.connection_start:
            uptime = (datetime.now() - self.connection_start).total_seconds()
            
        return {
            "connected": self.connected,
            "url": self.url,
            "uptime_seconds": uptime,
            "messages_received": self.messages_received,
            "subscriptions": list(self.subscriptions)
        }
    
    def close(self):
        """Close WebSocket connection"""
        logger.info(f"Closing WebSocket client - running: {self.running}, connected: {self.connected}")
        self.running = False
        self.connected = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        self.ws = None
        logger.info("WebSocket client closed")

