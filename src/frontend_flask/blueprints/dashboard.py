"""Trading Dashboard Blueprint"""
from flask import Blueprint, render_template, jsonify, request, current_app
import requests
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    """Main trading dashboard page"""
    return render_template('dashboard/index.html', 
                         active_page='dashboard',
                         page_title='Trading Dashboard')

@dashboard_bp.route('/debug')
def debug():
    """Debug page for testing"""
    return render_template('test_debug.html')

@dashboard_bp.route('/api/dashboard-data')
def get_dashboard_data():
    """Get all dashboard data in one request"""
    try:
        # Get backend URL from environment or config
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        # Make parallel requests to backend endpoints
        try:
            # Get current price
            price_resp = requests.get(f"{api_base_url}/price/current", timeout=5)
            price_data = price_resp.json() if price_resp.status_code == 200 else {'price': 0}
            
            # Get BTC latest data for 24h high/low
            btc_resp = requests.get(f"{api_base_url}/btc/latest", timeout=5)
            btc_data = btc_resp.json() if btc_resp.status_code == 200 else {}
            
            # Get latest signal
            signal_resp = requests.get(f"{api_base_url}/signals/latest", timeout=5)
            signal_data = signal_resp.json() if signal_resp.status_code == 200 else {}
            
            # Get portfolio metrics
            portfolio_resp = requests.get(f"{api_base_url}/portfolio/metrics", timeout=5)
            portfolio_metrics = portfolio_resp.json() if portfolio_resp.status_code == 200 else {}
            
            # Get paper trading status
            paper_resp = requests.get(f"{api_base_url}/paper-trading/status", timeout=5)
            paper_data = paper_resp.json() if paper_resp.status_code == 200 else {}
            
            # Get recent trades
            trades_resp = requests.get(f"{api_base_url}/trades/?limit=5", timeout=5)
            recent_trades = trades_resp.json() if trades_resp.status_code == 200 else []
            
            # Get positions
            positions_resp = requests.get(f"{api_base_url}/positions/", timeout=5)
            positions = positions_resp.json() if positions_resp.status_code == 200 else []
            
            # Get system health
            health_resp = requests.get(f"{api_base_url}/health", timeout=5)
            health_data = health_resp.json() if health_resp.status_code == 200 else {}
            
            # Process and combine data
            current_price = float(price_data.get('price', 0))
            
            # Calculate portfolio values based on paper trading status
            if paper_data.get('enabled'):
                portfolio = paper_data.get('portfolio', {})
                btc_balance = portfolio.get('btc_balance', 0)
                usd_balance = portfolio.get('usd_balance', 10000)
                total_value = usd_balance + (btc_balance * current_price)
                pnl = portfolio.get('total_pnl', 0)
                pnl_percentage = (pnl / 10000 * 100) if pnl != 0 else 0
            else:
                btc_balance = 0
                usd_balance = 0
                total_value = portfolio_metrics.get('total_invested', 0)
                pnl = portfolio_metrics.get('total_pnl', 0)
                pnl_percentage = 0
            
            # Build response matching frontend expectations
            data = {
                'price': {
                    'current': current_price,
                    'change_24h': price_data.get('change_24h', 0),
                    'high_24h': btc_data.get('high_24h', current_price * 1.02),
                    'low_24h': btc_data.get('low_24h', current_price * 0.98),
                    'volume_24h': price_data.get('volume', btc_data.get('total_volume', 0))
                },
                'signal': {
                    'current': signal_data.get('signal', 'HOLD').upper(),
                    'confidence': float(signal_data.get('confidence', 0)) * 100,
                    'timestamp': signal_data.get('timestamp', datetime.now().isoformat()),
                    'price': float(signal_data.get('price', current_price))
                },
                'portfolio': {
                    'total_value': total_value,
                    'btc_balance': btc_balance,
                    'usd_balance': usd_balance,
                    'change_24h': 0,  # TODO: Calculate actual 24h change
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage
                },
                'performance': {
                    'win_rate': paper_data.get('performance', {}).get('win_rate', 0),
                    'total_trades': portfolio_metrics.get('total_trades', 0),
                    'winning_trades': 0,  # TODO: Calculate from trades
                    'losing_trades': 0,   # TODO: Calculate from trades
                    'avg_profit': 0,      # TODO: Calculate from trades
                    'sharpe_ratio': paper_data.get('performance', {}).get('sharpe_ratio', 0)
                },
                'positions': positions[:5] if positions else [],
                'recent_trades': recent_trades[:5] if isinstance(recent_trades, list) else [],
                'system_status': {
                    'api': health_data.get('status') == 'healthy',
                    'database': health_data.get('components', {}).get('database') == 'healthy',
                    'model': health_data.get('components', {}).get('signal_generator') == 'healthy',
                    'trading': paper_data.get('enabled', False),
                    'websocket': health_data.get('components', {}).get('websocket_connections', 0) >= 0
                },
                'paper_trading_enabled': paper_data.get('enabled', False)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling backend API: {e}")
            # Return fallback data on API error
            data = {
                'price': {'current': 0, 'change_24h': 0, 'high_24h': 0, 'low_24h': 0, 'volume_24h': 0},
                'signal': {'current': 'ERROR', 'confidence': 0, 'timestamp': datetime.now().isoformat(), 'price': 0},
                'portfolio': {'total_value': 0, 'btc_balance': 0, 'usd_balance': 0, 'change_24h': 0, 'pnl': 0, 'pnl_percentage': 0},
                'performance': {'win_rate': 0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'avg_profit': 0, 'sharpe_ratio': 0},
                'positions': [],
                'recent_trades': [],
                'system_status': {'api': False, 'database': False, 'model': False, 'trading': False, 'websocket': False},
                'paper_trading_enabled': False
            }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/execute-trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    try:
        data = request.get_json()
        trade_type = data.get('type')
        quantity = data.get('quantity')
        price = data.get('price')
        
        # Validate inputs
        if not all([trade_type, quantity]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        # Check if paper trading is enabled
        try:
            paper_status_resp = requests.get(f"{api_base_url}/paper-trading/status", timeout=5)
            paper_data = paper_status_resp.json() if paper_status_resp.status_code == 200 else {}
            
            if paper_data.get('enabled'):
                # Execute paper trade
                trade_data = {
                    'trade_type': trade_type.lower(),
                    'quantity': float(quantity),
                    'price': float(price) if price else None
                }
                
                response = requests.post(f"{api_base_url}/paper-trading/trade", json=trade_data, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    return jsonify({
                        'success': True,
                        'trade_id': result.get('trade_id'),
                        'message': f'{trade_type} order executed successfully',
                        'details': {
                            'type': trade_type,
                            'quantity': quantity,
                            'price': result.get('price', price or 'market'),
                            'timestamp': result.get('timestamp', datetime.now().isoformat())
                        }
                    })
                else:
                    return jsonify({'error': 'Failed to execute trade', 'details': response.text}), response.status_code
            else:
                # Paper trading not enabled
                return jsonify({
                    'success': False,
                    'error': 'Paper trading is not enabled. Please enable it in settings first.'
                }), 400
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing trade: {e}")
            return jsonify({'error': 'Failed to connect to trading backend'}), 500
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/chart-data')
def get_chart_data():
    """Get chart data for price visualization"""
    try:
        timeframe = request.args.get('timeframe', '1D')
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        # Get price history from backend
        try:
            # Convert timeframe to days for backend API
            timeframe_to_days = {
                '1H': 2,    # 2 days for hourly data
                '4H': 7,    # 7 days for 4-hour data
                '1D': 30,   # 30 days for daily data
                '1W': 180,  # 180 days for weekly data
                '1M': 365   # 365 days for monthly data
            }
            days = timeframe_to_days.get(timeframe, 7)
            response = requests.get(f"{api_base_url}/price/history", params={'days': days}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Convert to format expected by frontend
                chart_data = []
                for item in data:
                    chart_data.append({
                        'timestamp': item.get('timestamp', item.get('date')),
                        'open': float(item.get('open', 0)),
                        'high': float(item.get('high', 0)),
                        'low': float(item.get('low', 0)),
                        'close': float(item.get('close', item.get('price', 0))),
                        'volume': float(item.get('volume', 0))
                    })
                return jsonify(chart_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching price history: {e}")
        
        # If we get here, the primary endpoint failed
        # Log the error and return empty array
        logger.error(f"Failed to fetch price history data for timeframe {timeframe}")
        
        # Return empty array on error
        return jsonify([])
    except Exception as e:
        logger.error(f"Error fetching chart data: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/paper-trading/status')
def get_paper_trading_status():
    """Get paper trading status"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        
        response = requests.get(f"{api_base_url}/paper-trading/status", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'enabled': False, 'error': 'Failed to get status'}), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting paper trading status: {e}")
        return jsonify({'enabled': False, 'error': str(e)}), 500

@dashboard_bp.route('/api/paper-trading/toggle', methods=['POST'])
def toggle_paper_trading():
    """Toggle paper trading on/off"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        data = request.get_json()
        action = data.get('action')
        
        if action not in ['enable', 'disable']:
            return jsonify({'error': 'Invalid action'}), 400
        
        response = requests.post(f"{api_base_url}/paper-trading/toggle", json={}, timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to toggle paper trading'}), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error toggling paper trading: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/paper-trading/trade', methods=['POST'])
def execute_paper_trade():
    """Execute a paper trade"""
    try:
        api_base_url = os.environ.get('API_BASE_URL', current_app.config.get('API_BASE_URL', 'http://localhost:8000'))
        data = request.get_json()
        
        # Forward the request to the backend
        response = requests.post(f"{api_base_url}/paper-trading/trade", json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'trade_id': result.get('trade_id'),
                'message': f"Paper {data.get('trade_type', 'order').upper()} executed successfully",
                'details': result
            })
        else:
            error_data = response.json() if response.text else {'error': 'Unknown error'}
            return jsonify({
                'success': False,
                'error': error_data.get('detail', error_data.get('error', 'Failed to execute paper trade'))
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error executing paper trade: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500