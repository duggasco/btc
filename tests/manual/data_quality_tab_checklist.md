# Data Quality Tab - Manual Test Checklist

## Test Environment Setup
- [ ] Docker environment is running (`docker compose up -d`)
- [ ] Frontend is accessible at http://localhost:8501
- [ ] Backend API is running at http://localhost:8000
- [ ] Test data is available in the database

## Visual Testing

### 1. Tab Visibility and Navigation
- [ ] Navigate to Settings page (Settings)
- [ ] Verify "Data Quality" tab is visible as the 7th tab
- [ ] Click on Data Quality tab
- [ ] Verify tab content loads without errors
- [ ] Verify tab remains selected when clicked

### 2. Summary Section
- [ ] Verify "Summary" header is displayed
- [ ] Check all 4 metric cards are visible:
  - [ ] Total Datapoints (shows comma-separated number)
  - [ ] Missing Dates (shows count)
  - [ ] Overall Completeness (shows percentage with one decimal)
  - [ ] Last Updated (shows date and time)
- [ ] Hover over each metric - verify help tooltips appear
- [ ] Verify metric values are realistic (not null or undefined)

### 3. Data Completeness by Type Table
- [ ] Verify section header "Data Completeness by Type" is visible
- [ ] Check table displays with proper columns:
  - [ ] Type (capitalized: Price, Volume, etc.)
  - [ ] Datapoints (comma-separated numbers)
  - [ ] Missing (comma-separated numbers)
  - [ ] Completeness % (with color gradient)
  - [ ] Start Date (YYYY-MM-DD format)
  - [ ] End Date (YYYY-MM-DD format)
- [ ] Verify color gradient on Completeness % column:
  - [ ] Red/orange for low completeness (0-50%)
  - [ ] Yellow for medium completeness (50-80%)
  - [ ] Green for high completeness (80-100%)
- [ ] Check all data types are listed (price, volume, onchain, sentiment, macro)

### 4. Data Availability by Source (Expandable Sections)
- [ ] Verify "Data Availability by Source" section exists
- [ ] Check expandable sections for each data type:
  - [ ] Price Data Sources
  - [ ] Volume Data Sources
  - [ ] Onchain Data Sources
  - [ ] Sentiment Data Sources
  - [ ] Macro Data Sources
- [ ] Click each expander:
  - [ ] Verify smooth expand/collapse animation
  - [ ] Check table displays inside with columns:
    - Source, Datapoints, Missing, Completeness %, Last Update
  - [ ] Verify color gradients work in nested tables
  - [ ] Check multiple sources are listed per type

### 5. Data Coverage Heatmap
- [ ] Verify "Data Coverage by Time Period" section displays
- [ ] Check interactive Plotly heatmap loads:
  - [ ] X-axis shows data types (Price, Volume, Onchain, Sentiment, Macro)
  - [ ] Y-axis shows time periods (Last 24 Hours, Last 7 Days, etc.)
  - [ ] Cells show coverage percentages
  - [ ] Color scale from red (0%) to green (100%)
- [ ] Test heatmap interactivity:
  - [ ] Hover over cells - verify tooltip shows exact percentage
  - [ ] Try zooming in/out
  - [ ] Test pan functionality
  - [ ] Check Plotly toolbar options work

### 6. Data Gaps Section
- [ ] Verify "Data Gaps" section is displayed
- [ ] If gaps exist:
  - [ ] Warning message shows gap count
  - [ ] Table displays with columns:
    - Start, End, Days Missing, Symbol, Granularity
  - [ ] Dates are properly formatted
- [ ] If no gaps:
  - [ ] Success message "No significant data gaps found!" displays

### 7. Cache Performance Section
- [ ] Verify "Cache Performance" section (may not always appear)
- [ ] If displayed, check 3 metrics:
  - [ ] Total Cache Entries (comma-separated)
  - [ ] Active Cache Entries (comma-separated)
  - [ ] Cache Efficiency (percentage)
- [ ] Verify efficiency calculation makes sense (active/total * 100)

### 8. Actions Section
- [ ] Verify "Actions" section is displayed
- [ ] Check 3 action buttons in columns:
  - [ ] Refresh Metrics
  - [ ] Fill Data Gaps
  - [ ] Clear Cache
- [ ] Test "Refresh Metrics" button:
  - [ ] Click button
  - [ ] Verify page reloads
  - [ ] Check metrics update (timestamp should change)
- [ ] Test "Fill Data Gaps" button:
  - [ ] Click button
  - [ ] Verify spinner appears
  - [ ] Check info message displays
- [ ] Test "Clear Cache" button:
  - [ ] Click button
  - [ ] Verify checkbox confirmation appears
  - [ ] Check checkbox and verify warning message

## Responsive Design Testing

### 9. Mobile View (375x667)
- [ ] Resize browser to mobile dimensions
- [ ] Verify all sections stack vertically
- [ ] Check metric cards display in single column
- [ ] Verify tables are scrollable horizontally
- [ ] Test expandable sections work on mobile
- [ ] Check buttons are full width and tappable

### 10. Tablet View (768x1024)
- [ ] Resize browser to tablet dimensions
- [ ] Verify 2-column layout for metrics
- [ ] Check tables display properly
- [ ] Verify heatmap resizes appropriately
- [ ] Test all interactive elements

### 11. Desktop View (1920x1080)
- [ ] Verify 4-column layout for summary metrics
- [ ] Check 3-column layout for action buttons
- [ ] Verify optimal spacing and alignment
- [ ] Test wide table display without horizontal scroll

## Error Handling Testing

### 12. API Error Scenarios
- [ ] Stop backend service
- [ ] Refresh page
- [ ] Verify error message displays: "Failed to load data quality metrics"
- [ ] Check error is styled properly (red alert box)
- [ ] Restart backend and verify recovery

### 13. Loading States
- [ ] Clear browser cache
- [ ] Navigate to Data Quality tab
- [ ] Verify "Loading data quality metrics..." spinner appears
- [ ] Check spinner disappears when data loads
- [ ] Verify smooth transition from loading to loaded state

### 14. Empty Data States
- [ ] If database is empty:
  - [ ] Verify appropriate empty state messages
  - [ ] Check no JavaScript errors in console
  - [ ] Verify UI doesn't break

## Performance Testing

### 15. Load Time
- [ ] Measure time from tab click to full render
- [ ] Should load within 3 seconds on average
- [ ] Check no console errors during load
- [ ] Verify no layout shifts after initial render

### 16. Memory Usage
- [ ] Open browser developer tools
- [ ] Monitor memory usage while using tab
- [ ] Expand/collapse all sections multiple times
- [ ] Verify no memory leaks (increasing memory usage)

## Accessibility Testing

### 17. Keyboard Navigation
- [ ] Tab through all interactive elements
- [ ] Verify focus indicators are visible
- [ ] Test Enter/Space on buttons
- [ ] Check expandable sections work with keyboard

### 18. Screen Reader Compatibility
- [ ] Enable screen reader
- [ ] Verify all sections have proper headings
- [ ] Check metric values are announced correctly
- [ ] Test table navigation with screen reader

## Cross-Browser Testing

### 19. Chrome/Chromium
- [ ] Test all functionality in Chrome
- [ ] Verify Plotly charts render correctly
- [ ] Check no console errors

### 20. Firefox
- [ ] Test all functionality in Firefox
- [ ] Verify CSS gradients work
- [ ] Check responsive design

### 21. Safari (if on macOS)
- [ ] Test all functionality in Safari
- [ ] Verify no webkit-specific issues
- [ ] Check touch gestures on trackpad

## Edge Cases

### 22. Large Dataset
- [ ] Test with millions of datapoints
- [ ] Verify tables paginate or scroll smoothly
- [ ] Check performance doesn't degrade

### 23. Extreme Values
- [ ] Test with 0% completeness
- [ ] Test with 100% completeness
- [ ] Test with very large gap counts
- [ ] Verify formatting remains correct

### 24. Concurrent Usage
- [ ] Open multiple browser tabs
- [ ] Navigate to Data Quality in each
- [ ] Verify all tabs update independently
- [ ] Check no session conflicts

## Final Verification

### 25. Overall Quality Check
- [ ] No JavaScript errors in console
- [ ] All animations are smooth
- [ ] Color scheme is consistent
- [ ] Text is readable (proper contrast)
- [ ] No broken images or icons
- [ ] Help tooltips are helpful and accurate

## Test Results Summary

**Date Tested:** _______________
**Tester Name:** _______________
**Environment:** _______________

**Overall Result:** [ ] PASS  [ ] FAIL

**Issues Found:**
1. _________________________________
2. _________________________________
3. _________________________________

**Notes:**
_____________________________________
_____________________________________
_____________________________________