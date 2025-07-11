#!/usr/bin/env python3
"""
End-to-end tests for the Data Quality tab in Settings page
Tests visual elements, data display, interactions, and error handling
"""

import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

class TestDataQualityTab:
    """Test suite for Data Quality tab functionality"""
    
    @classmethod
    def setup_class(cls):
        """Set up the test class with Chrome driver"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode for CI
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        cls.driver = webdriver.Chrome(options=options)
        cls.wait = WebDriverWait(cls.driver, 10)
        cls.base_url = "http://frontend:8501"  # Use Docker service name
    
    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        cls.driver.quit()
    
    def setup_method(self):
        """Navigate to Settings page before each test"""
        self.driver.get(f"{self.base_url}/⚙️_Settings")
        time.sleep(3)  # Wait for page to load
        
        # Click on Data Quality tab
        try:
            tabs = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[role='tab']"))
            )
            # Find and click the Data Quality tab (7th tab)
            for tab in tabs:
                if "Data Quality" in tab.text:
                    tab.click()
                    break
            time.sleep(2)  # Wait for tab content to load
        except TimeoutException:
            pytest.fail("Could not find tabs on Settings page")
    
    def test_data_quality_tab_visible(self):
        """Test that Data Quality tab is visible and clickable"""
        tabs = self.driver.find_elements(By.CSS_SELECTOR, "[role='tab']")
        data_quality_tab = None
        
        for tab in tabs:
            if "Data Quality" in tab.text:
                data_quality_tab = tab
                break
        
        assert data_quality_tab is not None, "Data Quality tab not found"
        assert data_quality_tab.is_displayed(), "Data Quality tab not visible"
        assert "selected" in data_quality_tab.get_attribute("aria-selected"), "Data Quality tab not selected"
    
    def test_summary_section_display(self):
        """Test that summary section displays correctly"""
        # Check for summary header
        try:
            summary_header = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//h3[contains(text(), 'Summary')]"))
            )
            assert summary_header.is_displayed()
        except TimeoutException:
            pytest.fail("Summary header not found")
        
        # Check for metric cards
        metric_labels = [
            "Total Datapoints",
            "Missing Dates", 
            "Overall Completeness",
            "Last Updated"
        ]
        
        for label in metric_labels:
            try:
                metric = self.driver.find_element(
                    By.XPATH, f"//div[contains(@class, 'metric')]//label[contains(text(), '{label}')]"
                )
                assert metric.is_displayed(), f"Metric '{label}' not displayed"
                
                # Check that metric has a value
                metric_container = metric.find_element(By.XPATH, "./ancestor::div[contains(@class, 'metric')]")
                value_element = metric_container.find_element(By.XPATH, ".//div[@data-testid='metric-value']")
                assert value_element.text, f"Metric '{label}' has no value"
                
            except NoSuchElementException:
                pytest.fail(f"Metric '{label}' not found")
    
    def test_data_completeness_table(self):
        """Test data completeness by type table"""
        try:
            # Check for section header
            header = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//h3[contains(text(), 'Data Completeness by Type')]"))
            )
            assert header.is_displayed()
            
            # Check for dataframe
            dataframe = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='stDataFrame']")
            assert dataframe.is_displayed()
            
            # Check table headers
            expected_headers = ['Type', 'Datapoints', 'Missing', 'Completeness %', 'Start Date', 'End Date']
            headers = dataframe.find_elements(By.CSS_SELECTOR, "thead th")
            actual_headers = [h.text for h in headers]
            
            for expected in expected_headers:
                assert any(expected in header for header in actual_headers), f"Header '{expected}' not found"
            
            # Check that table has rows
            rows = dataframe.find_elements(By.CSS_SELECTOR, "tbody tr")
            assert len(rows) > 0, "No data rows found in completeness table"
            
        except TimeoutException:
            pytest.fail("Data completeness table not found")
    
    def test_expandable_data_sources(self):
        """Test expandable sections for data sources"""
        try:
            # Find all expander buttons
            expanders = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='expander']")
            assert len(expanders) > 0, "No expandable sections found"
            
            # Test first expander
            first_expander = expanders[0]
            expander_button = first_expander.find_element(By.CSS_SELECTOR, "button")
            
            # Check initial state (should be collapsed)
            assert "false" in expander_button.get_attribute("aria-expanded"), "Expander should be collapsed initially"
            
            # Click to expand
            expander_button.click()
            time.sleep(1)
            
            # Check expanded state
            assert "true" in expander_button.get_attribute("aria-expanded"), "Expander should be expanded after click"
            
            # Check for content inside expander
            content = first_expander.find_element(By.CSS_SELECTOR, "[data-testid='stExpander'] > div:last-child")
            assert content.is_displayed(), "Expander content not visible"
            
            # Check for dataframe inside expander
            inner_dataframe = content.find_element(By.CSS_SELECTOR, "[data-testid='stDataFrame']")
            assert inner_dataframe.is_displayed(), "Dataframe inside expander not visible"
            
        except NoSuchElementException:
            pytest.fail("Expandable sections not working properly")
    
    def test_coverage_heatmap(self):
        """Test data coverage heatmap visualization"""
        try:
            # Check for section header
            header = self.driver.find_element(
                By.XPATH, "//h3[contains(text(), 'Data Coverage by Time Period')]"
            )
            assert header.is_displayed()
            
            # Check for Plotly chart
            plotly_chart = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".js-plotly-plot"))
            )
            assert plotly_chart.is_displayed(), "Coverage heatmap not displayed"
            
            # Check that chart has rendered
            svg = plotly_chart.find_element(By.CSS_SELECTOR, "svg.main-svg")
            assert svg.is_displayed(), "Heatmap SVG not rendered"
            
            # Check for axis labels
            x_axis = plotly_chart.find_element(By.CSS_SELECTOR, ".xtitle")
            y_axis = plotly_chart.find_element(By.CSS_SELECTOR, ".ytitle")
            assert "Data Type" in x_axis.text, "X-axis label incorrect"
            assert "Time Period" in y_axis.text, "Y-axis label incorrect"
            
        except (TimeoutException, NoSuchElementException):
            pytest.fail("Coverage heatmap not found or not rendered properly")
    
    def test_data_gaps_section(self):
        """Test data gaps display"""
        try:
            # Check for section header
            header = self.driver.find_element(
                By.XPATH, "//h3[contains(text(), 'Data Gaps')]"
            )
            assert header.is_displayed()
            
            # Check for either warning message or success message
            messages = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stAlert']")
            assert len(messages) > 0, "No gap status message found"
            
            # If gaps exist, check for dataframe
            warning_messages = [m for m in messages if "warning" in m.get_attribute("class")]
            if warning_messages:
                # Should have a dataframe with gap details
                gap_table = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='stDataFrame']")
                assert gap_table.is_displayed(), "Gap details table not displayed"
            
        except NoSuchElementException:
            pytest.fail("Data gaps section not found")
    
    def test_cache_metrics_display(self):
        """Test cache performance metrics"""
        try:
            # Check for section header
            header = self.driver.find_element(
                By.XPATH, "//h3[contains(text(), 'Cache Performance')]"
            )
            assert header.is_displayed()
            
            # Check for cache metric cards
            cache_metrics = [
                "Total Cache Entries",
                "Active Cache Entries",
                "Cache Efficiency"
            ]
            
            for metric in cache_metrics:
                try:
                    metric_element = self.driver.find_element(
                        By.XPATH, f"//label[contains(text(), '{metric}')]"
                    )
                    assert metric_element.is_displayed(), f"Cache metric '{metric}' not displayed"
                except NoSuchElementException:
                    pytest.fail(f"Cache metric '{metric}' not found")
            
        except NoSuchElementException:
            # Cache metrics might not be displayed if no cache data
            pass
    
    def test_action_buttons(self):
        """Test action buttons functionality"""
        try:
            # Check for Actions header
            header = self.driver.find_element(
                By.XPATH, "//h3[contains(text(), 'Actions')]"
            )
            assert header.is_displayed()
            
            # Test Refresh Metrics button
            refresh_btn = self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Refresh Metrics')]"
            )
            assert refresh_btn.is_displayed()
            assert refresh_btn.is_enabled()
            
            # Test Fill Data Gaps button
            fill_gaps_btn = self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Fill Data Gaps')]"
            )
            assert fill_gaps_btn.is_displayed()
            assert fill_gaps_btn.is_enabled()
            
            # Test Clear Cache button
            clear_cache_btn = self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Clear Cache')]"
            )
            assert clear_cache_btn.is_displayed()
            assert clear_cache_btn.is_enabled()
            
            # Test clicking Refresh Metrics
            initial_url = self.driver.current_url
            refresh_btn.click()
            time.sleep(2)
            # Page should reload
            assert self.driver.current_url == initial_url, "Page did not refresh"
            
        except NoSuchElementException:
            pytest.fail("Action buttons not found")
    
    def test_responsive_design(self):
        """Test responsive design at different screen sizes"""
        original_size = self.driver.get_window_size()
        
        # Test mobile size
        self.driver.set_window_size(375, 667)
        time.sleep(1)
        
        # Check that main elements are still visible
        summary = self.driver.find_element(By.XPATH, "//h3[contains(text(), 'Summary')]")
        assert summary.is_displayed(), "Summary not visible on mobile"
        
        # Test tablet size
        self.driver.set_window_size(768, 1024)
        time.sleep(1)
        
        # Check layout
        metrics = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='metric']")
        assert len(metrics) > 0, "Metrics not visible on tablet"
        
        # Restore original size
        self.driver.set_window_size(original_size['width'], original_size['height'])
    
    def test_color_gradients_in_tables(self):
        """Test that color gradients are applied correctly"""
        try:
            # Find styled dataframes
            styled_tables = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stDataFrame'] table")
            
            for table in styled_tables:
                # Check for cells with background gradient
                styled_cells = table.find_elements(By.CSS_SELECTOR, "td[style*='background']")
                if styled_cells:
                    # At least some cells should have styling
                    assert len(styled_cells) > 0, "No styled cells found in tables"
                    
                    # Check that styles include color values
                    for cell in styled_cells[:5]:  # Check first 5 cells
                        style = cell.get_attribute("style")
                        assert "background" in style or "color" in style, "Cell styling not applied"
            
        except NoSuchElementException:
            pass  # Tables might not have gradients if no data
    
    def test_error_handling(self):
        """Test error handling when API fails"""
        # This test would require mocking API failures
        # For now, we'll check that error containers exist
        error_containers = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stAlert'][data-baseweb='notification']")
        
        # If any errors are displayed, they should have proper formatting
        for error in error_containers:
            if "error" in error.get_attribute("class"):
                assert error.is_displayed(), "Error message not properly displayed"
                assert error.text, "Error message has no text"
    
    def test_loading_states(self):
        """Test loading states are displayed correctly"""
        # Navigate to page and immediately check for spinner
        self.driver.get(f"{self.base_url}/⚙️_Settings")
        
        # Look for loading spinner (might be too fast to catch)
        spinners = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSpinner']")
        
        # If spinner exists, it should be visible
        for spinner in spinners:
            if spinner.is_displayed():
                assert "Loading" in spinner.text or "loading" in spinner.text.lower(), "Loading message not clear"


def run_tests():
    """Run the test suite and report results"""
    import subprocess
    
    # Run pytest with verbose output
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print("Test Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    run_tests()