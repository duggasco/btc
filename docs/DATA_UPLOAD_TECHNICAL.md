# Data Upload Technical Documentation

This document provides technical details about the manual data upload implementation.

## Architecture Overview

The data upload feature consists of:
- Frontend component (`DataUploader`) for user interface
- Backend service (`DataUploadService`) for processing
- API endpoints for communication
- SQLite storage in `historical_data.db`

## Database Schema

### Historical Data Table
```sql
CREATE TABLE IF NOT EXISTS historical_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'BTC',
    source TEXT NOT NULL,
    data_type TEXT NOT NULL,
    frequency TEXT NOT NULL,
    
    -- Price data
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    price REAL,
    volume REAL,
    
    -- On-chain metrics
    active_addresses INTEGER,
    transaction_count INTEGER,
    hash_rate REAL,
    difficulty REAL,
    fees REAL,
    
    -- Sentiment data
    sentiment_score REAL,
    fear_greed_index INTEGER,
    social_volume INTEGER,
    mentions INTEGER,
    
    -- Macro indicators
    dxy REAL,
    gold REAL,
    sp500 REAL,
    vix REAL,
    bond_yield REAL,
    
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(timestamp, symbol, source, data_type, frequency)
);

-- Indexes for performance
CREATE INDEX idx_historical_timestamp ON historical_data(timestamp);
CREATE INDEX idx_historical_symbol ON historical_data(symbol);
CREATE INDEX idx_historical_source ON historical_data(source);
CREATE INDEX idx_historical_type ON historical_data(data_type);
CREATE INDEX idx_historical_composite ON historical_data(timestamp, symbol, source);
```

## Data Processing Pipeline

### 1. File Upload Stage
- File received via multipart form data
- Saved to temporary directory (`/tmp/btc_uploads/`)
- Basic validation (size, format)

### 2. Preview Stage
- Read first 10 rows for preview
- Detect column data types
- Generate mapping suggestions
- Return preview to frontend

### 3. Validation Stage
- Validate file format and size
- Check required columns exist
- Verify data types are correct
- Ensure timestamps are valid

### 4. Processing Stage
- Apply column mappings
- Clean and standardize data
- Convert timestamps to UTC
- Handle missing values

### 5. Storage Stage
- Begin database transaction
- Bulk insert with conflict handling
- Calculate summary statistics
- Commit or rollback

## Performance Optimizations

### Bulk Insert Strategy
```python
# Batch size for optimal performance
BATCH_SIZE = 1000

# Use executemany for batch inserts
cursor.executemany(
    "INSERT OR IGNORE INTO historical_data (...) VALUES (...)",
    batch_data
)
```

### Memory Management
- Stream large files instead of loading entirely
- Process in chunks to limit memory usage
- Clean up temporary files immediately

### Database Optimizations
- Compound unique index prevents duplicates
- Batch inserts reduce transaction overhead
- Prepared statements prevent SQL injection
- Connection pooling for concurrent uploads

## Error Handling

### Rollback Mechanism
```python
try:
    conn.execute("BEGIN TRANSACTION")
    # Process and insert data
    conn.commit()
except Exception as e:
    conn.rollback()
    raise
```

### Validation Errors
- Missing required columns
- Invalid data types
- Out-of-range values
- Malformed timestamps

### Recovery
- Automatic cleanup of temp files
- Transaction rollback on errors
- Detailed error messages
- Partial success reporting

## Column Mapping Algorithm

### Automatic Detection
```python
def suggest_mappings(columns):
    # Pattern matching for common column names
    patterns = {
        'timestamp': ['date', 'time', 'timestamp', 'datetime'],
        'price': ['price', 'close', 'last', 'value'],
        'volume': ['volume', 'vol', 'quantity', 'amount']
    }
    
    # Fuzzy matching with similarity threshold
    for col in columns:
        col_lower = col.lower()
        for target, patterns in patterns.items():
            if any(p in col_lower for p in patterns):
                mappings[col] = target
```

### Manual Override
- User can override suggestions
- Validation ensures mappings are complete
- Store mapping preferences for future

## Data Quality Checks

### Pre-Processing
- Remove duplicate rows in file
- Handle missing values appropriately
- Standardize number formats
- Convert timestamps to UTC

### Post-Processing
- Verify row counts match
- Check data ranges are reasonable
- Ensure no data corruption
- Log anomalies for review

## Security Considerations

### File Upload Security
- Validate file extensions
- Check MIME types
- Scan for malicious content
- Limit file sizes

### Data Validation
- Sanitize all inputs
- Use parameterized queries
- Validate data ranges
- Prevent injection attacks

### Access Control
- Authentication required
- Rate limiting applied
- Audit logging enabled
- Temporary file cleanup

## Integration Points

### With Historical Data Manager
```python
# DataUploadService stores in same database
# HistoricalDataManager can query uploaded data
historical_data = manager.get_historical_data(
    symbol='BTC',
    frequency='daily',
    source='custom',
    start_date=start,
    end_date=end
)
```

### With Data Quality Monitor
- Uploaded data included in quality metrics
- Gap analysis considers all sources
- Completeness calculations updated

### With Trading System
- Uploaded data available for backtesting
- Feature engineering uses all data sources
- Model training includes custom data

## Monitoring and Logging

### Upload Metrics
- Files uploaded per day
- Data points inserted
- Error rates
- Processing times

### Audit Trail
```python
logger.info(f"Upload started: {filename} ({file_size} bytes)")
logger.info(f"Data type: {data_type}, Source: {source}")
logger.info(f"Rows processed: {rows_processed}, Inserted: {rows_inserted}")
```

### Performance Tracking
- Upload duration
- Database insert time
- Memory usage
- Disk I/O

## Future Enhancements

### Planned Features
1. **Async Processing** - Background jobs for large files
2. **Data Transformation** - Custom data pipelines
3. **Format Support** - JSON, Parquet, HDF5
4. **Validation Rules** - User-defined validation
5. **Scheduling** - Automated periodic uploads

### Scalability Considerations
- Migrate to PostgreSQL for larger datasets
- Implement data partitioning
- Add caching layer
- Support distributed processing

## Troubleshooting

### Common Issues

**High Memory Usage**
- Reduce batch size
- Enable streaming mode
- Process files in chunks

**Slow Uploads**
- Check database indexes
- Optimize batch size
- Review network latency

**Duplicate Errors**
- Verify unique constraints
- Check timestamp precision
- Review source naming

### Debug Mode
```python
# Enable debug logging
logger.setLevel(logging.DEBUG)

# Verbose SQL logging
conn.set_trace_callback(print)

# Performance profiling
import cProfile
cProfile.run('process_upload(...)')
```

## Testing

### Unit Tests
- Column mapping logic
- Data validation rules
- Error handling paths
- Database operations

### Integration Tests
- End-to-end upload flow
- Large file handling
- Concurrent uploads
- Error recovery

### Performance Tests
- Upload speed benchmarks
- Memory usage profiling
- Database insert rates
- Concurrent user load