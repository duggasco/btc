#!/usr/bin/env python3
"""Test script for enhanced LSTM system endpoints"""

import requests
import time
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8090"  # Using port 8090 as per deployment

def test_enhanced_lstm_status():
    """Test enhanced LSTM status endpoint"""
    print("\n1. Testing Enhanced LSTM Status...")
    
    try:
        response = requests.get(f"{BASE_URL}/enhanced-lstm/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status: {data.get('status')}")
            print(f"  Model Trained: {data.get('model_trained')}")
            print(f"  Needs Retraining: {data.get('needs_retraining')}")
            if data.get('last_training_date'):
                print(f"  Last Training: {data.get('last_training_date')}")
            return True
        else:
            print(f"✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_enhanced_data_status():
    """Test data availability endpoint"""
    print("\n2. Testing Enhanced Data Status...")
    
    try:
        response = requests.get(f"{BASE_URL}/enhanced-lstm/data-status")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Data Status: {data.get('status')}")
            if data.get('status') == 'available':
                print(f"  Days Available: {data.get('days_fetched')}")
                print(f"  Total Features: {data.get('total_features')}")
                categories = data.get('feature_categories', {})
                print(f"  Technical Indicators: {categories.get('technical_indicators', 0)}")
                print(f"  On-chain Metrics: {categories.get('on_chain', 0)}")
                print(f"  Sentiment Features: {categories.get('sentiment', 0)}")
            return True
        else:
            print(f"✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_enhanced_prediction():
    """Test enhanced prediction endpoint"""
    print("\n3. Testing Enhanced Prediction...")
    
    try:
        response = requests.get(f"{BASE_URL}/enhanced-lstm/predict")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Signal: {data.get('signal')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            print(f"  Predicted Price: ${data.get('predicted_price', 0):,.2f}")
            if 'prediction_range' in data:
                range_data = data['prediction_range']
                print(f"  Prediction Range: ${range_data.get('lower', 0):,.2f} - ${range_data.get('upper', 0):,.2f}")
            return True
        elif response.status_code == 400:
            print("✓ Model not trained yet (expected)")
            return True
        else:
            print(f"✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_enhanced_signal_endpoint():
    """Test the main enhanced signal endpoint"""
    print("\n4. Testing Enhanced Signal Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/signals/enhanced/latest")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Signal: {data.get('signal')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            
            analysis = data.get('analysis', {})
            if 'model_type' in analysis:
                print(f"  Model Type: {analysis.get('model_type')}")
            if 'components' in analysis:
                components = analysis['components']
                print(f"  Components: {len(components)} signal types")
            return True
        else:
            print(f"✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_train_model():
    """Test model training endpoint (optional - takes time)"""
    print("\n5. Testing Model Training (Optional)...")
    print("⚠️  This will take 5-10 minutes. Skip with Ctrl+C")
    
    try:
        time.sleep(3)  # Give user time to cancel
        
        print("Starting training...")
        response = requests.post(f"{BASE_URL}/enhanced-lstm/train")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Training Status: {data.get('status')}")
            print(f"  Message: {data.get('message')}")
            
            metrics = data.get('training_metrics', {})
            if metrics:
                print(f"  Test RMSE: {metrics.get('avg_rmse', 0):.4f}")
                print(f"  Directional Accuracy: {metrics.get('avg_directional_accuracy', 0):.2%}")
            
            features = data.get('selected_features', [])
            if features:
                print(f"  Top Features: {', '.join(features[:5])}...")
            
            return True
        else:
            print(f"✗ Failed with status {response.status_code}")
            error_detail = response.json().get('detail', 'Unknown error')
            print(f"  Error: {error_detail}")
            return False
    except KeyboardInterrupt:
        print("\n⚠️  Training skipped by user")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Enhanced LSTM System Test Suite")
    print("=" * 60)
    print(f"API Base URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("\n❌ API is not responding. Please ensure the system is deployed.")
            return
    except:
        print("\n❌ Cannot connect to API. Please ensure the system is deployed.")
        print(f"   Try: docker-compose up -d")
        return
    
    # Run tests
    tests = [
        test_enhanced_lstm_status,
        test_enhanced_data_status,
        test_enhanced_prediction,
        test_enhanced_signal_endpoint,
        # test_train_model  # Commented out by default due to time
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    print("\nNote: To train the enhanced model, uncomment test_train_model in the tests list")
    print("      or run: curl -X POST http://localhost:8090/enhanced-lstm/train")

if __name__ == "__main__":
    main()