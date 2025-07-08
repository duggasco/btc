#!/usr/bin/env python3
"""Test WebSocket functionality"""

import asyncio
import websockets
import json
import time
from datetime import datetime

async def test_websocket():
    uri = "ws://localhost:8090/ws"
    print(f"Connecting to WebSocket: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket successfully")
            
            # Listen for messages for 10 seconds
            timeout = 10
            start_time = time.time()
            message_count = 0
            
            print(f"Listening for messages for {timeout} seconds...")
            
            while time.time() - start_time < timeout:
                try:
                    # Wait for message with a short timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    # Display message info
                    msg_type = data.get('type', 'unknown')
                    timestamp = data.get('timestamp', 'N/A')
                    
                    if msg_type == 'price_update':
                        price = data.get('data', {}).get('price', 'N/A')
                        print(f"📈 Price Update: ${price:,} at {timestamp}")
                    elif msg_type == 'signal_update':
                        signal = data.get('data', {}).get('signal', 'N/A')
                        confidence = data.get('data', {}).get('confidence', 0)
                        print(f"🎯 Signal Update: {signal} (Confidence: {confidence:.1%}) at {timestamp}")
                    else:
                        print(f"📦 Message: {msg_type} at {timestamp}")
                        
                except asyncio.TimeoutError:
                    # No message received in 1 second, continue waiting
                    continue
                except json.JSONDecodeError:
                    print("⚠️  Received non-JSON message")
                    continue
            
            print(f"\n✅ WebSocket test completed")
            print(f"📊 Total messages received: {message_count}")
            
            if message_count > 0:
                print("✅ WebSocket is functioning correctly")
                return True
            else:
                print("⚠️  No messages received - WebSocket may not be broadcasting")
                return False
                
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_websocket())
    exit(0 if result else 1)