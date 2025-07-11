# Sample Data Format Reference

This document provides sample CSV formats for each data type supported by the upload feature.

## Price Data (OHLCV)

### Basic Format
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.00,42500.00,41800.00,42300.00,1234.56
2024-01-01 01:00:00,42300.00,42400.00,42100.00,42250.00,987.65
2024-01-01 02:00:00,42250.00,42600.00,42200.00,42550.00,1456.78
```

### Alternative Column Names (Auto-Mapped)
```csv
Date,Open Price,High Price,Low Price,Closing Price,24h Volume
2024-01-01,42000.00,42500.00,41800.00,42300.00,1234.56
2024-01-02,42300.00,43000.00,42000.00,42800.00,1567.89
```

### Daily Data Format
```csv
timestamp,open,high,low,close,volume
2024-01-01,42000,43000,41500,42500,150000
2024-01-02,42500,44000,42000,43500,175000
2024-01-03,43500,45000,43000,44500,200000
```

## Volume Data

### Simple Volume
```csv
timestamp,volume
2024-01-01 00:00:00,1234.56
2024-01-01 01:00:00,987.65
2024-01-01 02:00:00,1456.78
```

### Volume with Price and Trade Count
```csv
timestamp,volume,price,trades_count
2024-01-01 00:00:00,1234.56,42000.00,1500
2024-01-01 01:00:00,987.65,42300.00,1200
2024-01-01 02:00:00,1456.78,42250.00,1800
```

## On-chain Metrics

### Network Activity
```csv
timestamp,active_addresses,transaction_count,fees
2024-01-01,850000,350000,125.50
2024-01-02,875000,360000,130.25
2024-01-03,900000,375000,135.75
```

### Mining Metrics
```csv
timestamp,hash_rate,difficulty
2024-01-01,500.5,65000000000000
2024-01-02,510.2,65500000000000
2024-01-03,505.8,65250000000000
```

### Combined On-chain Data
```csv
timestamp,active_addresses,transaction_count,hash_rate,difficulty,fees
2024-01-01,850000,350000,500.5,65000000000000,125.50
2024-01-02,875000,360000,510.2,65500000000000,130.25
```

## Sentiment Data

### Basic Sentiment
```csv
timestamp,sentiment_score
2024-01-01,0.65
2024-01-02,0.70
2024-01-03,0.55
```

### Fear & Greed with Social Volume
```csv
timestamp,sentiment_score,fear_greed_index,social_volume,mentions
2024-01-01,0.65,72,15000,3500
2024-01-02,0.70,75,18000,4200
2024-01-03,0.55,68,12000,2800
```

### Hourly Sentiment
```csv
timestamp,sentiment_score,fear_greed_index
2024-01-01 00:00:00,0.65,72
2024-01-01 01:00:00,0.67,73
2024-01-01 02:00:00,0.63,71
```

## Macro Indicators

### US Dollar and Gold
```csv
timestamp,dxy,gold
2024-01-01,102.5,2050.00
2024-01-02,102.3,2055.00
2024-01-03,102.7,2048.00
```

### Stock Market Indicators
```csv
timestamp,sp500,vix
2024-01-01,4700.50,15.2
2024-01-02,4710.25,14.8
2024-01-03,4705.75,15.5
```

### Complete Macro Dataset
```csv
timestamp,dxy,gold,sp500,vix,bond_yield
2024-01-01,102.5,2050.00,4700.50,15.2,4.25
2024-01-02,102.3,2055.00,4710.25,14.8,4.28
2024-01-03,102.7,2048.00,4705.75,15.5,4.22
```

## Timestamp Format Examples

### ISO 8601
```csv
timestamp,price,volume
2024-01-01T00:00:00Z,42000.00,1234.56
2024-01-01T01:00:00Z,42300.00,987.65
```

### Unix Timestamp
```csv
timestamp,price,volume
1704067200,42000.00,1234.56
1704070800,42300.00,987.65
```

### Date Only (Daily Data)
```csv
timestamp,close,volume
2024-01-01,42000.00,150000
2024-01-02,42500.00,175000
```

## Column Mapping Examples

### Original File
```csv
Date,BTC Price,24h Volume,Fear Index
2024-01-01,42000,150000,72
```

### Column Mapping Required
```json
{
  "Date": "timestamp",
  "BTC Price": "price",
  "24h Volume": "volume",
  "Fear Index": "fear_greed_index"
}
```

## Tips for Data Preparation

1. **Consistent Formatting**
   - Use same date format throughout
   - Decimal points for prices
   - No thousand separators

2. **Clean Data**
   - Remove empty rows
   - Handle missing values
   - Fix obvious errors

3. **Proper Headers**
   - First row must be column names
   - No special characters
   - Clear, descriptive names

4. **Time Zones**
   - UTC recommended
   - Consistent timezone throughout
   - Include timezone in timestamp if not UTC

## Validation Examples

### Good Data
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.00,42500.00,41800.00,42300.00,1234.56
```

### Bad Data - Missing Required Column
```csv
date,open,high,low,close
2024-01-01,42000,42500,41800,42300
```
Missing: `volume` column

### Bad Data - Invalid Values
```csv
timestamp,open,high,low,close,volume
2024-01-01,42000,42500,41800,42300,-100
```
Error: Negative volume

### Bad Data - Wrong Format
```csv
timestamp,open,high,low,close,volume
Jan 1 2024,42,000.00,42,500.00,41,800.00,42,300.00,1,234.56
```
Error: Comma separators in numbers, informal date format