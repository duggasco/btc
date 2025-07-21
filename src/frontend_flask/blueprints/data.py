"""Data Management Blueprint"""
from flask import Blueprint, render_template, jsonify, request, current_app, send_file
import requests
import logging
import os
import json
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import io
import base64

logger = logging.getLogger(__name__)

data_bp = Blueprint('data', __name__, url_prefix='/data')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@data_bp.route('/upload')
def upload_page():
    """Data upload page"""
    return render_template('data/upload.html', 
                         active_page='data_upload',
                         page_title='Data Upload')

@data_bp.route('/quality')
def quality_page():
    """Data quality dashboard"""
    return render_template('data/quality.html', 
                         active_page='data_quality',
                         page_title='Data Quality')

@data_bp.route('/history')
def history_page():
    """Upload history page"""
    return render_template('data/history.html', 
                         active_page='data_history',
                         page_title='Upload History')

@data_bp.route('/manage')
def manage_page():
    """Data management page for deletion"""
    return render_template('data/manage.html', 
                         active_page='data_manage',
                         page_title='Data Management')

@data_bp.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV and XLSX files are allowed'}), 400
        
        # Get upload parameters
        data_type = request.form.get('data_type', 'ohlcv')
        symbol = request.form.get('symbol', 'BTC')
        source = request.form.get('source', 'manual_upload')
        
        # Map frontend data types to backend data types
        backend_data_type_map = {
            'ohlcv': 'price',
            'onchain': 'onchain',
            'sentiment': 'sentiment',
            'macro': 'macro'
        }
        backend_data_type = backend_data_type_map.get(data_type, 'price')
        
        # Read file content
        file_content = file.read()
        filename = secure_filename(file.filename)
        
        # Determine file type from extension
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'csv'
        
        # Send to backend
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        files = {'file': (filename, file_content, file.content_type)}
        data = {
            'file_type': file_ext,
            'data_type': backend_data_type,
            'symbol': symbol,
            'source': source
        }
        
        response = requests.post(
            f"{api_base_url}/data/upload",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Check if backend reports success (handle both boolean and integer)
            backend_success = result.get('success')
            if backend_success is True or backend_success == 1 or backend_success == "1":
                rows_inserted = result.get('rows_inserted', 0)
                logger.info(f"Backend response: success={backend_success} (type: {type(backend_success).__name__}), rows_inserted={rows_inserted}")
                success_message = f"Successfully uploaded {rows_inserted} records"
                logger.info(f"Returning success message to frontend: {success_message}")
                return jsonify({
                    'success': True,
                    'message': success_message,
                    'details': result
                })
            else:
                # Backend returned 200 but success=false
                error_msg = result.get('error', 'Upload failed - no rows inserted')
                logger.warning(f"Backend returned success=false: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'details': result
                }), 400
        else:
            error_data = response.json() if response.text else {'error': 'Upload failed'}
            return jsonify({
                'success': False,
                'error': error_data.get('detail', error_data.get('error', 'Upload failed'))
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/preview', methods=['POST'])
def preview_file():
    """Preview file before upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Read file for preview
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), nrows=10)
        else:  # xlsx
            df = pd.read_excel(io.BytesIO(file_content), nrows=10)
        
        # Convert to preview format
        preview_data = {
            'columns': df.columns.tolist(),
            'rows': df.to_dict('records'),
            'total_rows': len(df),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return jsonify({
            'success': True,
            'preview': preview_data
        })
        
    except Exception as e:
        logger.error(f"Error previewing file: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/quality-metrics')
def get_quality_metrics():
    """Get data quality metrics"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        response = requests.get(f"{api_base_url}/analytics/data-quality", timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to fetch quality metrics'}), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching quality metrics: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/upload-history')
def get_upload_history():
    """Get upload history"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        # Get upload tracking data
        response = requests.get(f"{api_base_url}/data/uploads", timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # Fallback to empty history if endpoint doesn't exist
            return jsonify({
                'uploads': [],
                'total': 0
            })
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching upload history: {e}")
        return jsonify({'uploads': [], 'total': 0})

@data_bp.route('/api/sample-formats')
def get_sample_formats():
    """Get sample data formats"""
    return jsonify({
        'formats': {
            'ohlcv': {
                'columns': ['date', 'open', 'high', 'low', 'close', 'volume'],
                'example': [
                    {
                        'date': '2025-01-01',
                        'open': 95000,
                        'high': 96000,
                        'low': 94000,
                        'close': 95500,
                        'volume': 12345.67
                    }
                ]
            },
            'onchain': {
                'columns': ['date', 'active_addresses', 'transaction_count', 'hash_rate', 'difficulty', 'block_size', 'fees_total', 'supply'],
                'example': [
                    {
                        'date': '2025-01-01',
                        'active_addresses': 950000,
                        'transaction_count': 320000,
                        'hash_rate': 500000000,
                        'difficulty': 70000000000000,
                        'block_size': 1.5,
                        'fees_total': 25.5,
                        'supply': 19700000
                    }
                ]
            },
            'sentiment': {
                'columns': ['timestamp', 'fear_greed_index', 'reddit_sentiment', 'twitter_sentiment', 'news_sentiment', 'google_trends'],
                'example': [
                    {
                        'timestamp': '2025-01-01 00:00:00',
                        'fear_greed_index': 65,
                        'reddit_sentiment': 0.75,
                        'twitter_sentiment': 0.68,
                        'news_sentiment': 0.72,
                        'google_trends': 85
                    }
                ]
            },
            'macro': {
                'columns': ['date', 'symbol', 'value', 'indicator_type'],
                'example': [
                    {
                        'date': '2025-01-01',
                        'symbol': 'DXY',
                        'value': 103.5,
                        'indicator_type': 'currency_index'
                    }
                ]
            }
        }
    })

@data_bp.route('/api/download-template/<data_type>')
def download_template(data_type):
    """Download sample template for data type"""
    try:
        # Create sample data based on type
        if data_type == 'ohlcv':
            df = pd.DataFrame({
                'date': pd.date_range('2025-01-01', periods=5),
                'open': [95000, 95500, 96000, 95800, 96200],
                'high': [95800, 96200, 96500, 96300, 96800],
                'low': [94500, 95200, 95700, 95500, 95900],
                'close': [95500, 96000, 95800, 96200, 96500],
                'volume': [12345.67, 13456.78, 14567.89, 12890.12, 13901.23]
            })
            filename = 'ohlcv_template.csv'
        elif data_type == 'onchain':
            df = pd.DataFrame({
                'date': pd.date_range('2025-01-01', periods=5),
                'active_addresses': [950000, 960000, 955000, 965000, 970000],
                'transaction_count': [320000, 325000, 318000, 330000, 335000],
                'hash_rate': [500000000, 505000000, 510000000, 508000000, 512000000],
                'difficulty': [70000000000000, 70100000000000, 70200000000000, 70150000000000, 70300000000000],
                'block_size': [1.5, 1.48, 1.52, 1.49, 1.51],
                'fees_total': [25.5, 26.2, 24.8, 27.1, 26.5],
                'supply': [19700000, 19700100, 19700200, 19700300, 19700400]
            })
            filename = 'onchain_template.csv'
        elif data_type == 'sentiment':
            df = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=5, freq='H'),
                'fear_greed_index': [65, 68, 63, 70, 72],
                'reddit_sentiment': [0.75, 0.78, 0.72, 0.80, 0.82],
                'twitter_sentiment': [0.68, 0.70, 0.65, 0.72, 0.74],
                'news_sentiment': [0.72, 0.74, 0.70, 0.76, 0.78],
                'google_trends': [85, 87, 83, 88, 90]
            })
            filename = 'sentiment_template.csv'
        elif data_type == 'macro':
            df = pd.DataFrame({
                'date': pd.date_range('2025-01-01', periods=5),
                'symbol': ['DXY', 'GLD', 'VIX', 'TNX', 'DXY'],
                'value': [103.5, 185.2, 15.3, 4.25, 103.8],
                'indicator_type': ['currency_index', 'commodity', 'volatility', 'bond_yield', 'currency_index']
            })
            filename = 'macro_template.csv'
        else:
            return jsonify({'error': 'Invalid data type'}), 400
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/rollback/<upload_id>', methods=['POST'])
def rollback_upload(upload_id):
    """Rollback a specific upload"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        response = requests.post(
            f"{api_base_url}/data/rollback/{upload_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'message': 'Upload rolled back successfully'
            })
        else:
            error_data = response.json() if response.text else {'error': 'Rollback failed'}
            return jsonify({
                'success': False,
                'error': error_data.get('detail', error_data.get('error', 'Rollback failed'))
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error rolling back upload: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/data-gaps')
def get_data_gaps():
    """Get data gaps information"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        response = requests.get(f"{api_base_url}/data/gaps", timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # Return empty gaps if endpoint doesn't exist
            return jsonify({
                'gaps': [],
                'summary': {
                    'total_gaps': 0,
                    'total_missing_days': 0
                }
            })
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data gaps: {e}")
        return jsonify({'gaps': [], 'summary': {'total_gaps': 0, 'total_missing_days': 0}})

@data_bp.route('/api/available')
def get_available_data():
    """Get available data for deletion"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        response = requests.get(f"{api_base_url}/data/available", timeout=10)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to fetch available data'}), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching available data: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/stats/<data_type>')
def get_data_stats(data_type):
    """Get statistics for specific data"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        # Forward query parameters
        source = request.args.get('source')
        symbol = request.args.get('symbol', 'BTC')
        
        response = requests.get(
            f"{api_base_url}/data/stats/{data_type}",
            params={'source': source, 'symbol': symbol},
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to fetch data statistics'}), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data stats: {e}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/api/delete', methods=['DELETE'])
def delete_data():
    """Delete data from database"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        # Forward all query parameters
        params = {
            'data_type': request.args.get('data_type'),
            'source': request.args.get('source'),
            'symbol': request.args.get('symbol', 'BTC'),
            'start_date': request.args.get('start_date'),
            'end_date': request.args.get('end_date'),
            'confirm': request.args.get('confirm', 'false')
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.delete(
            f"{api_base_url}/data/delete",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {'error': 'Deletion failed'}
            return jsonify({
                'success': False,
                'error': error_data.get('detail', error_data.get('error', 'Deletion failed'))
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting data: {e}")
        return jsonify({'error': str(e)}), 500