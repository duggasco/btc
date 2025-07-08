import websocket
import threading
import queue
import json
import logging
import time
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

class EnhancedWebSocketClient:
    """Enhanced WebSocket client with auto-reconnection and event handling"""
    
    def __init__(self, url: str, reconnect_interval: int = 5):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.ws = None
        self.message_queue = queue.Queue()
        self.running = False
        self.connected = False
        self.callbacks = {}
        self.subscriptions = set()
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.message_queue.put(data)
            
            # Handle callbacks for specific message types
            msg_type = data.get('type', 'unknown')
            if msg_type in self.callbacks:
                self.callbacks[msg_type](data)
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid WebSocket message: {message}")
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        self.connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Auto-reconnect if still running
        if self.running:
            time.sleep(self.reconnect_interval)
            self.connect()
            
    def on_open(self, ws):
        logger.info("WebSocket connected")
        self.connected = True
        
        # Re-subscribe to all channels
        for channel in self.subscriptions:
            self.subscribe(channel)
    
    def connect(self):
        """Connect to WebSocket server with auto-reconnection"""
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.running = True
            wst = threading.Thread(target=self._run_forever)
            wst.daemon = True
            wst.start()
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            self.connected = False
    
    def _run_forever(self):
        """Run WebSocket connection in a loop with auto-reconnection"""
        while self.running:
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
                if self.running:
                    time.sleep(self.reconnect_interval)
    
    def subscribe(self, channel: str):
        """Subscribe to a specific channel"""
        self.subscriptions.add(channel)
        if self.connected and self.ws:
            self.ws.send(json.dumps({
                "action": "subscribe",
                "channel": channel
            }))
    
    def unsubscribe(self, channel: str):
        """Unsubscribe from a specific channel"""
        self.subscriptions.discard(channel)
        if self.connected and self.ws:
            self.ws.send(json.dumps({
                "action": "unsubscribe",
                "channel": channel
            }))
    
    def send_message(self, message: Dict[str, Any]):
        """Send a message to the WebSocket server"""
        if self.connected and self.ws:
            self.ws.send(json.dumps(message))
    
    def register_callback(self, msg_type: str, callback: Callable):
        """Register a callback for specific message types"""
        self.callbacks[msg_type] = callback
    
    def get_messages(self, max_messages: int = 100):
        """Get all pending messages from the queue"""
        messages = []
        for _ in range(max_messages):
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connected
    
    def close(self):
        """Close WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
