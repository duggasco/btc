# OHLCV Data Upload Fix

## Issue Summary
The data upload functionality was only expecting `price` and `volume` fields for OHLCV data, instead of the full OHLCV dataset (Open, High, Low, Close, Volume).

## Root Cause
1. The `DataUploadService` was treating OHLCV data as simple "price" data with only timestamp and price as required fields
2. OHLCV fields were marked as "optional" instead of required
3. Data was being stored in a generic `historical_data` table with JSON instead of the proper `ohlcv_data` table

## Fix Applied

### 1. Updated Required Fields (`src/backend/services/data_upload_service.py`)
Changed the data type template for "price" data to require all OHLCV fields:
```python
"price": {
    "required": ["timestamp", "open", "high", "low", "close", "volume"],
    "optional": ["price"],
    "description": "OHLCV data - requires all standard OHLCV fields"
}
```

### 2. Updated Storage Method
Modified `_store_in_database` method to:
- Store OHLCV data in the proper `ohlcv_data` table
- Create the table with correct schema if it doesn't exist
- Insert data with proper column mapping

### 3. Updated Summary Generation
Enhanced `_generate_summary` method to:
- Use close price for statistics (instead of generic "price")
- Add OHLCV-specific statistics (high_max, low_min, avg_range)

## Expected Upload Format
CSV files should now include all OHLCV fields:
```csv
timestamp,open,high,low,close,volume
2025-01-11 00:00:00,95000,95500,94500,95200,1000
2025-01-11 01:00:00,95200,95700,95000,95400,1100
```

## Testing
The fix was tested with a sample OHLCV dataset and successfully:
- Validated all required fields
- Processed the data correctly
- Generated appropriate statistics

## Deployment Note
To apply this fix in production:
1. Rebuild the backend container with the updated code
2. The fix will automatically create the proper table structure if needed
3. Existing data in the generic table remains unaffected