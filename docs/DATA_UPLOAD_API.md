# Data Upload API Documentation

This document describes the API endpoints for manual data upload functionality.

## Endpoints Overview

- `POST /data/upload/preview` - Preview file and get column mapping suggestions
- `POST /data/upload` - Process and store uploaded data
- `GET /data/upload/templates/{data_type}` - Get column requirements for data type
- `GET /data/upload/sources` - Get list of valid data sources

## Authentication

All endpoints require proper authentication headers if security is enabled.

## Endpoints

### 1. Preview File Upload

Preview uploaded file contents and get column mapping suggestions.

**Endpoint:** `POST /data/upload/preview`

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Data:**
- `file` (required) - The file to upload (CSV or Excel)
- `data_type` (required) - Type of data: `price`, `volume`, `onchain`, `sentiment`, `macro`

**Response:**
```json
{
  "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
  "row_count": 1000,
  "sample_data": [
    {
      "Date": "2024-01-01",
      "Open": 42000.0,
      "High": 42500.0,
      "Low": 41800.0,
      "Close": 42300.0,
      "Volume": 1234.56
    }
  ],
  "suggested_mappings": {
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
  },
  "data_types": {
    "Date": "object",
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Volume": "float64"
  }
}
```

**Error Responses:**
- `400 Bad Request` - Invalid file format or data type
- `413 Request Entity Too Large` - File exceeds size limit
- `500 Internal Server Error` - Processing error

**Example:**
```bash
curl -X POST http://localhost:8090/data/upload/preview \
  -F "file=@bitcoin_prices.csv" \
  -F "data_type=price"
```

### 2. Process Data Upload

Upload and process data file, storing in the historical database.

**Endpoint:** `POST /data/upload`

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Data:**
- `file` (required) - The file to upload
- `source` (required) - Data source: `binance`, `coingecko`, `cryptowatch`, `glassnode`, `santiment`, `custom`
- `data_type` (required) - Type of data: `price`, `volume`, `onchain`, `sentiment`, `macro`
- `symbol` (optional, default: "BTC") - Asset symbol
- `column_mappings` (optional) - JSON string mapping original columns to required columns

**Column Mappings Format:**
```json
{
  "original_column_name": "required_column_name",
  "Date": "timestamp",
  "Close": "close"
}
```

**Response:**
```json
{
  "success": true,
  "rows_processed": 1000,
  "rows_inserted": 950,
  "duplicate_rows": 50,
  "data_range": {
    "start": "2024-01-01T00:00:00",
    "end": "2024-03-31T23:59:59"
  },
  "summary": {
    "avg_price": 45000.50,
    "min_price": 41000.00,
    "max_price": 50000.00,
    "total_volume": 1234567.89
  }
}
```

**Error Responses:**
- `400 Bad Request` - Invalid parameters or missing required columns
- `409 Conflict` - All data already exists (duplicates)
- `413 Request Entity Too Large` - File too large
- `422 Unprocessable Entity` - Data validation failed
- `500 Internal Server Error` - Processing or database error

**Example:**
```bash
# With automatic column mapping
curl -X POST http://localhost:8090/data/upload \
  -F "file=@bitcoin_prices.csv" \
  -F "source=custom" \
  -F "data_type=price"

# With custom column mapping
curl -X POST http://localhost:8090/data/upload \
  -F "file=@bitcoin_data.csv" \
  -F "source=binance" \
  -F "data_type=price" \
  -F "symbol=BTC" \
  -F 'column_mappings={"Date":"timestamp","BTC Price":"close","24h Volume":"volume"}'
```

### 3. Get Data Type Template

Get column requirements and template for specific data type.

**Endpoint:** `GET /data/upload/templates/{data_type}`

**Parameters:**
- `data_type` (path parameter) - One of: `price`, `volume`, `onchain`, `sentiment`, `macro`

**Response:**
```json
{
  "data_type": "price",
  "required": ["timestamp", "price"],
  "optional": ["volume", "open", "high", "low", "close"],
  "description": "Historical price data with optional OHLCV values",
  "example": {
    "timestamp": "2024-01-01 00:00:00",
    "open": 42000.0,
    "high": 42500.0,
    "low": 41800.0,
    "close": 42300.0,
    "volume": 1234.56
  }
}
```

**Error Responses:**
- `404 Not Found` - Invalid data type

**Example:**
```bash
curl http://localhost:8090/data/upload/templates/price
```

### 4. Get Valid Sources

Get list of valid data sources for uploads.

**Endpoint:** `GET /data/upload/sources`

**Response:**
```json
{
  "sources": [
    {
      "id": "binance",
      "name": "Binance",
      "description": "Binance exchange data"
    },
    {
      "id": "coingecko",
      "name": "CoinGecko",
      "description": "CoinGecko API data"
    },
    {
      "id": "cryptowatch",
      "name": "Cryptowatch",
      "description": "Cryptowatch market data"
    },
    {
      "id": "glassnode",
      "name": "Glassnode",
      "description": "On-chain analytics"
    },
    {
      "id": "santiment",
      "name": "Santiment",
      "description": "Sentiment and social data"
    },
    {
      "id": "custom",
      "name": "Custom",
      "description": "User-provided data source"
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8090/data/upload/sources
```

## Data Validation Rules

### Timestamp Validation
- Must be parseable datetime
- Supported formats: ISO 8601, `YYYY-MM-DD HH:MM:SS`, Unix timestamp
- Must be within reasonable range (not future dates)

### Numeric Validation
- Price values must be positive
- Volume must be non-negative
- Sentiment scores between -1 and 1
- Fear/Greed index between 0 and 100

### Required Fields
- All required fields must be present
- Cannot be null or empty
- Must have correct data type

## Error Response Format

All error responses follow this format:

```json
{
  "detail": "Human-readable error message",
  "error_code": "ERROR_CODE",
  "validation_errors": [
    {
      "field": "column_name",
      "message": "Specific validation error"
    }
  ]
}
```

## Rate Limiting

- Upload endpoints are rate-limited to prevent abuse
- Default: 10 uploads per minute per IP
- Large files count as multiple requests based on size

## Best Practices

1. **File Size**
   - Keep files under 50 MB
   - Split larger datasets into multiple files
   - Compress files before upload if needed

2. **Data Quality**
   - Validate data before upload
   - Ensure consistent formatting
   - Remove invalid or corrupt rows

3. **Performance**
   - Upload during off-peak hours for large files
   - Use batch uploads for multiple files
   - Monitor progress via response data

4. **Error Handling**
   - Check validation errors in response
   - Retry with exponential backoff on 5xx errors
   - Log failed uploads for debugging

## Examples

### Python Upload Script
```python
import requests

# Preview file
with open('bitcoin_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8090/data/upload/preview',
        files={'file': f},
        data={'data_type': 'price'}
    )
    preview = response.json()
    print(f"Found {preview['row_count']} rows")

# Upload with mapping
with open('bitcoin_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8090/data/upload',
        files={'file': f},
        data={
            'source': 'custom',
            'data_type': 'price',
            'column_mappings': json.dumps(preview['suggested_mappings'])
        }
    )
    result = response.json()
    print(f"Inserted {result['rows_inserted']} rows")
```

### JavaScript Upload
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('source', 'custom');
formData.append('data_type', 'price');

fetch('http://localhost:8090/data/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(result => {
    console.log(`Uploaded ${result.rows_inserted} rows`);
});
```