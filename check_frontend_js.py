#!/usr/bin/env python3
"""Check if frontend JavaScript is executing"""
import requests
import time

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

def test_with_selenium():
    """Use Selenium to check JavaScript execution"""
    print("Testing frontend with Selenium...")
    
    # Configure Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        # Create driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load the dashboard
        driver.get("http://localhost:8502/")
        
        # Wait for JavaScript to execute (up to 10 seconds)
        wait = WebDriverWait(driver, 10)
        
        # Wait for price to update from $0.00
        price_element = wait.until(
            EC.presence_of_element_located((By.ID, "btc-price"))
        )
        
        # Give it a moment for JS to update
        time.sleep(2)
        
        # Get the actual displayed price
        displayed_price = driver.find_element(By.ID, "btc-price").text
        print(f"Displayed BTC Price in header: {displayed_price}")
        
        # Check if main price display exists
        try:
            main_price = driver.find_element(By.ID, "current-price-value").text
            print(f"Main price display: {main_price}")
        except:
            print("Main price display element not found")
        
        # Check console logs for errors
        logs = driver.get_log('browser')
        if logs:
            print("\nBrowser console logs:")
            for log in logs:
                print(f"  {log['level']}: {log['message']}")
        
        driver.quit()
        
    except Exception as e:
        print(f"Selenium test failed: {e}")
        print("Note: This requires Chrome/Chromium and chromedriver to be installed")

def test_simple():
    """Simple test without Selenium"""
    print("Simple frontend test...")
    
    # Get dashboard HTML
    response = requests.get("http://localhost:8502/")
    
    # Check if JavaScript files are loaded
    js_files = ['api-client.js', 'dashboard.js', 'realtime-updates.js']
    for js_file in js_files:
        if js_file in response.text:
            print(f"✓ {js_file} is included")
        else:
            print(f"✗ {js_file} is NOT included")
    
    # Check API response
    api_response = requests.get("http://localhost:8502/api/dashboard-data")
    data = api_response.json()
    print(f"\nAPI returns price: ${data['price']['current']:,.2f}")
    
    # Look for any hardcoded prices
    if "95000" in response.text or "95,000" in response.text:
        print("\nWARNING: Found '95000' in HTML!")
    elif "$0.00" in response.text:
        print("\nHTML contains default $0.00 - JavaScript should update this")
    
    print("\nConclusion: If you see $95,000 in your browser but API returns ~$117,000,")
    print("it could be:")
    print("1. Browser cache - try hard refresh (Ctrl+Shift+R)")
    print("2. JavaScript error - check browser console (F12)")
    print("3. Different data source - check if you have multiple tabs/windows open")

if __name__ == "__main__":
    test_simple()
    print("\n" + "="*50 + "\n")
    
    # Try Selenium test if available
    if SELENIUM_AVAILABLE:
        try:
            test_with_selenium()
        except Exception as e:
            print(f"Selenium test failed: {e}")
            print("For full JavaScript testing, you need Chrome/Chromium + chromedriver")
    else:
        print("Selenium not available - install with: pip install selenium")
        print("For full JavaScript testing, you need Selenium + Chrome/Chromium")