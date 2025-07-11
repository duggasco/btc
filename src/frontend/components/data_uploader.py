"""
Data Upload Component with comprehensive validation

This component provides a user-friendly interface for uploading CSV/Excel files
with real-time validation feedback before submitting to the server.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
import os
from datetime import datetime
import io

class DataUploader:
    """Component for handling data file uploads with validation"""
    
    # Maximum file size in MB
    MAX_FILE_SIZE_MB = 50
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    
    # Data type configurations
    DATA_TYPES = {
        'price': {
            'name': 'Price Data (OHLCV)',
            'description': 'Historical price data with Open, High, Low, Close, and Volume',
            'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'icon': '[PRICE]'
        },
        'volume': {
            'name': 'Volume Data',
            'description': 'Trading volume data over time',
            'required_columns': ['timestamp', 'volume'],
            'icon': '[VOLUME]'
        },
        'onchain': {
            'name': 'On-chain Metrics',
            'description': 'Blockchain metrics like active addresses, transaction count, etc.',
            'required_columns': ['timestamp', 'metric_name', 'metric_value'],
            'icon': '[ONCHAIN]'
        },
        'sentiment': {
            'name': 'Sentiment Data',
            'description': 'Market sentiment scores and social indicators',
            'required_columns': ['timestamp', 'sentiment_score'],
            'icon': '[SENTIMENT]'
        },
        'macro': {
            'name': 'Macro Indicators',
            'description': 'Macroeconomic indicators like DXY, interest rates, etc.',
            'required_columns': ['timestamp', 'indicator_name', 'indicator_value'],
            'icon': '[MACRO]'
        }
    }
    
    def __init__(self, api_client):
        """Initialize the data uploader component"""
        self.api_client = api_client
        
    def render(self):
        """Render the data upload interface"""
        st.markdown("### 📤 Data Upload")
        st.info("Upload historical data to enhance the trading system's analysis capabilities")
        
        # Data type selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            data_type = st.selectbox(
                "Data Type",
                options=list(self.DATA_TYPES.keys()),
                format_func=lambda x: f"{self.DATA_TYPES[x]['icon']} {self.DATA_TYPES[x]['name']}",
                help="Select the type of data you're uploading"
            )
        
        with col2:
            if data_type:
                st.markdown(f"**Description:** {self.DATA_TYPES[data_type]['description']}")
                st.markdown(f"**Required columns:** `{', '.join(self.DATA_TYPES[data_type]['required_columns'])}`")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help=f"Maximum file size: {self.MAX_FILE_SIZE_MB} MB"
        )
        
        if uploaded_file is not None:
            # Validate file before sending to server
            is_valid, validation_msg = self._validate_file_client_side(uploaded_file, data_type)
            
            if not is_valid:
                st.error(f"{validation_msg}")
                return
            
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
                "Type": uploaded_file.type
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**File Details:**")
                for key, value in file_details.items():
                    st.write(f"• {key}: {value}")
            
            with col2:
                # Additional options
                symbol = st.text_input("Symbol", value="BTC", help="Trading symbol for this data")
                source = st.text_input("Source", value="manual_upload", help="Data source identifier")
            
            # Preview data
            if st.checkbox("Preview data", value=True):
                df_preview = self._load_file_preview(uploaded_file)
                if df_preview is not None:
                    st.markdown("**Data Preview (first 10 rows):**")
                    st.dataframe(df_preview.head(10), use_container_width=True)
                    
                    # Basic statistics
                    st.markdown("**Basic Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", f"{len(df_preview):,}")
                    with col2:
                        st.metric("Total Columns", len(df_preview.columns))
                    with col3:
                        st.metric("Memory Usage", f"{df_preview.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Validation and upload buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Validate Only", type="secondary", use_container_width=True):
                    self._validate_file_server_side(uploaded_file, data_type, symbol, source, validate_only=True)
            
            with col2:
                if st.button("📤 Upload & Process", type="primary", use_container_width=True):
                    self._upload_file(uploaded_file, data_type, symbol, source)
        
        # Sample format download
        st.markdown("---")
        st.markdown("### 📋 Sample Formats")
        st.write("Download sample CSV files to see the expected format for each data type:")
        
        cols = st.columns(5)
        for idx, (dtype, config) in enumerate(self.DATA_TYPES.items()):
            with cols[idx % 5]:
                if st.button(f"{config['icon']} {dtype.capitalize()}", use_container_width=True):
                    self._download_sample_format(dtype)
    
    def _validate_file_client_side(self, uploaded_file, data_type: str) -> Tuple[bool, str]:
        """Perform client-side validation of the uploaded file"""
        # Check file size
        if uploaded_file.size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File size exceeds maximum allowed size of {self.MAX_FILE_SIZE_MB} MB"
        
        # Check file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type. Please upload one of: {', '.join(self.SUPPORTED_EXTENSIONS)}"
        
        # Check if file is empty
        if uploaded_file.size == 0:
            return False, "File is empty"
        
        return True, "File passed initial validation"
    
    def _load_file_preview(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load a preview of the uploaded file"""
        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if file_ext == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        uploaded_file.seek(0)
                        return pd.read_csv(uploaded_file, encoding=encoding, nrows=1000)
                    except UnicodeDecodeError:
                        continue
                st.error("Unable to read CSV file with common encodings")
                return None
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(uploaded_file, nrows=1000)
            else:
                return None
        except Exception as e:
            st.error(f"Error loading file preview: {str(e)}")
            return None
    
    def _validate_file_server_side(self, uploaded_file, data_type: str, symbol: str, 
                                  source: str, validate_only: bool = True):
        """Send file to server for validation"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Prepare form data
            files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {
                'data_type': data_type,
                'symbol': symbol,
                'source': source
            }
            
            # Show spinner
            with st.spinner("Validating file..."):
                # Use requests directly for file upload
                import requests
                
                base_url = self.api_client.base_url.rstrip('/')
                response = requests.post(
                    f"{base_url}/data/validate",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._display_validation_results(result)
                else:
                    st.error(f"Validation failed: {response.text}")
                    
        except Exception as e:
            st.error(f"Error during validation: {str(e)}")
    
    def _upload_file(self, uploaded_file, data_type: str, symbol: str, source: str):
        """Upload file to server for processing"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Prepare form data
            files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {
                'data_type': data_type,
                'symbol': symbol,
                'source': source,
                'validate_only': False
            }
            
            # Show spinner
            with st.spinner("Uploading and processing file..."):
                # Use requests directly for file upload
                import requests
                
                base_url = self.api_client.base_url.rstrip('/')
                response = requests.post(
                    f"{base_url}/data/upload",
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes timeout for large files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['status'] == 'success':
                        st.success(f"{result['message']}")
                        
                        # Show summary
                        if 'rows_processed' in result:
                            st.metric("Rows Processed", f"{result['rows_processed']:,}")
                        
                        # Show validation details if available
                        if 'validation' in result:
                            with st.expander("View validation details"):
                                self._display_validation_results(result)
                    else:
                        st.error(f"Upload failed: {result.get('message', 'Unknown error')}")
                        if 'validation' in result:
                            self._display_validation_results(result)
                else:
                    st.error(f"Upload failed: {response.text}")
                    
        except Exception as e:
            st.error(f"Error during upload: {str(e)}")
    
    def _display_validation_results(self, result: Dict[str, Any]):
        """Display validation results in a user-friendly format"""
        validation = result.get('validation', {})
        
        # Overall status
        if validation.get('is_valid'):
            st.success("Validation passed!")
        else:
            st.error("Validation failed!")
        
        # Statistics
        stats = validation.get('statistics', {})
        if stats:
            st.markdown("**Statistics:**")
            cols = st.columns(6)
            
            metrics = [
                ("Total Rows", stats.get('total_rows', 0)),
                ("Valid Rows", stats.get('valid_rows', 0)),
                ("Error Rows", stats.get('error_rows', 0)),
                ("Warning Rows", stats.get('warning_rows', 0)),
                ("Duplicate Rows", stats.get('duplicate_rows', 0)),
                ("Conflict Rows", stats.get('conflict_rows', 0))
            ]
            
            for idx, (label, value) in enumerate(metrics):
                with cols[idx]:
                    st.metric(label, f"{value:,}")
        
        # Errors
        errors = validation.get('errors', [])
        if errors:
            st.markdown("**Errors:**")
            error_df = pd.DataFrame(errors)
            
            # Group by error type
            if 'message' in error_df.columns:
                error_summary = error_df['message'].value_counts().head(10)
                for error_msg, count in error_summary.items():
                    st.write(f"• {error_msg} ({count} occurrences)")
            
            # Show detailed errors
            with st.expander(f"View all errors ({len(errors)} total)"):
                for idx, error in enumerate(errors[:50]):  # Show first 50
                    row = error.get('row', 'N/A')
                    col = error.get('column', 'N/A')
                    msg = error.get('message', 'N/A')
                    val = error.get('value', 'N/A')
                    
                    st.write(f"**Row {row}, Column '{col}':** {msg}")
                    if val != 'N/A' and val is not None:
                        st.code(f"Value: {val}")
                
                if len(errors) > 50:
                    st.info(f"Showing first 50 errors out of {len(errors)} total")
        
        # Warnings
        warnings = validation.get('warnings', [])
        if warnings:
            st.markdown("**Warnings:**")
            
            # Group by warning type
            warning_df = pd.DataFrame(warnings)
            if 'message' in warning_df.columns:
                warning_summary = warning_df['message'].value_counts().head(5)
                for warning_msg, count in warning_summary.items():
                    st.write(f"• {warning_msg} ({count} occurrences)")
            
            # Show detailed warnings
            with st.expander(f"View all warnings ({len(warnings)} total)"):
                for warning in warnings[:20]:  # Show first 20
                    row = warning.get('row', 'N/A')
                    col = warning.get('column', 'N/A')
                    msg = warning.get('message', 'N/A')
                    
                    st.write(f"**Row {row}, Column '{col}':** {msg}")
        
        # Suggestions
        suggestions = validation.get('suggestions', [])
        if suggestions:
            st.markdown("**Suggestions:**")
            for suggestion in suggestions:
                st.info(f"• {suggestion}")
    
    def _download_sample_format(self, data_type: str):
        """Download a sample CSV file for the specified data type"""
        try:
            # Get sample format from API
            result = self.api_client.get(f"/data/sample-format/{data_type}")
            
            if result and result.get('status') == 'success':
                csv_string = result.get('sample_csv', '')
                
                # Create download button
                st.download_button(
                    label=f"Download {data_type} sample.csv",
                    data=csv_string,
                    file_name=f"sample_{data_type}_data.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Failed to get sample format for {data_type}")
                
        except Exception as e:
            st.error(f"Error getting sample format: {str(e)}")