# Data Upload User Guide

This guide explains how to prepare and upload historical data files to the Bitcoin Trading System.

## Quick Start

1. Navigate to **Settings** → **Data Upload** tab
2. Select your data type
3. Upload CSV or Excel file
4. Map columns if needed
5. Submit for processing

## Supported File Formats

- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel format
- **Maximum size**: 50 MB per file

## Data Types and Required Columns

### 1. Price Data (OHLCV)
Historical price data with candlestick information.

**Required columns:**
- `timestamp` - Date/time of the data point
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price  
- `close` - Closing price
- `volume` - Trading volume

**Sample CSV format:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.00,42500.00,41800.00,42300.00,1234.56
2024-01-01 01:00:00,42300.00,42400.00,42100.00,42250.00,987.65
```

### 2. Volume Data
Trading volume metrics over time.

**Required columns:**
- `timestamp` - Date/time of the data point
- `volume` - Trading volume

**Optional columns:**
- `price` - Price at the time
- `trades_count` - Number of trades

**Sample CSV format:**
```csv
timestamp,volume,price,trades_count
2024-01-01 00:00:00,1234.56,42000.00,1500
2024-01-01 01:00:00,987.65,42300.00,1200
```

### 3. On-chain Metrics
Blockchain network data.

**Required columns:**
- `timestamp` - Date/time of the data point

**Optional columns:**
- `active_addresses` - Number of active addresses
- `transaction_count` - Number of transactions
- `hash_rate` - Network hash rate
- `difficulty` - Mining difficulty
- `fees` - Average transaction fees

**Sample CSV format:**
```csv
timestamp,active_addresses,transaction_count,hash_rate
2024-01-01,850000,350000,500.5
2024-01-02,875000,360000,510.2
```

### 4. Sentiment Data
Market sentiment indicators.

**Required columns:**
- `timestamp` - Date/time of the data point

**Optional columns:**
- `sentiment_score` - Overall sentiment (-1 to 1)
- `fear_greed_index` - Fear & Greed Index (0-100)
- `social_volume` - Social media activity volume
- `mentions` - Number of mentions

**Sample CSV format:**
```csv
timestamp,sentiment_score,fear_greed_index,social_volume
2024-01-01,0.65,72,15000
2024-01-02,0.70,75,18000
```

### 5. Macro Indicators
Macroeconomic data affecting Bitcoin.

**Required columns:**
- `timestamp` - Date/time of the data point

**Optional columns:**
- `dxy` - US Dollar Index
- `gold` - Gold price
- `sp500` - S&P 500 index
- `vix` - Volatility Index
- `bond_yield` - 10-year bond yield

**Sample CSV format:**
```csv
timestamp,dxy,gold,sp500,vix
2024-01-01,102.5,2050.00,4700.50,15.2
2024-01-02,102.3,2055.00,4710.25,14.8
```

## Column Mapping

The system automatically suggests column mappings based on common patterns:

### Automatic Detection Patterns

**Timestamp columns:**
- date, time, timestamp, datetime, ts, created_at

**Price columns:**
- price, close, last, value, btc_price, bitcoin_price

**Volume columns:**
- volume, vol, quantity, amount, btc_volume

### Manual Mapping

If automatic detection fails:
1. Review the preview table
2. Use dropdown menus to map your columns
3. Ensure all required columns are mapped

## Data Sources

Select the appropriate source for your data:
- `binance` - Binance exchange data
- `coingecko` - CoinGecko API data
- `cryptowatch` - Cryptowatch data
- `glassnode` - Glassnode on-chain data
- `santiment` - Santiment metrics
- `custom` - Your own data source

## Date/Time Formats

Supported timestamp formats:
- ISO 8601: `2024-01-01T00:00:00Z`
- Standard: `2024-01-01 00:00:00`
- Date only: `2024-01-01` (assumes 00:00:00)
- Unix timestamp: `1704067200`

## Validation and Error Handling

### Pre-upload Validation
- File size check (< 50 MB)
- File format verification
- Basic structure validation

### Processing Validation
- Required columns check
- Data type validation
- Date range verification
- Duplicate detection

### Common Errors

**"Missing required columns"**
- Ensure all required columns exist
- Check column mapping is correct

**"Invalid timestamp format"**
- Use supported date formats
- Ensure consistent formatting

**"Duplicate data detected"**
- System prevents duplicate entries
- Shows count of skipped duplicates

## Best Practices

1. **Data Quality**
   - Ensure data is clean and accurate
   - Remove any corrupt or incomplete rows
   - Verify timestamps are in correct timezone

2. **File Preparation**
   - Use consistent date formatting
   - Include column headers
   - Sort data by timestamp (optional but recommended)

3. **Large Datasets**
   - Split files larger than 50 MB
   - Upload in chronological order
   - Use the Data Quality tab to verify completeness

4. **Testing**
   - Start with a small sample file
   - Verify data appears correctly
   - Check Data Quality tab after upload

## Troubleshooting

### Upload Fails Immediately
- Check file size (< 50 MB)
- Verify file extension (.csv, .xlsx, .xls)
- Ensure file is not corrupted

### Column Mapping Issues
- Column names are case-sensitive
- Remove special characters from headers
- Ensure no duplicate column names

### Data Not Appearing
- Check Data Quality tab for verification
- Ensure timestamp range is valid
- Verify source selection is correct

### Performance Issues
- Large files may take time to process
- Progress bar shows upload status
- System processes in batches for efficiency

## API Integration

For programmatic uploads, use the API endpoints:

```bash
# Preview file
curl -X POST http://localhost:8090/data/upload/preview \
  -F "file=@data.csv" \
  -F "data_type=price"

# Upload and process
curl -X POST http://localhost:8090/data/upload \
  -F "file=@data.csv" \
  -F "source=custom" \
  -F "data_type=price" \
  -F "column_mappings={\"Date\":\"timestamp\",\"Close\":\"close\"}"
```

## Post-Upload Verification

After uploading:
1. Go to **Settings** → **Data Quality** tab
2. Check completeness percentages
3. Review gap analysis
4. Verify data range matches expectations

## Support

For issues or questions:
- Check error messages for specific guidance
- Review this guide for requirements
- Ensure data format matches examples
- Use "custom" source for non-standard data