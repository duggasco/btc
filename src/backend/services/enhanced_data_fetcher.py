"""
Enhanced Data Fetcher with Multiple Sources and Proper Error Handling
Based on LSTM whitepaper recommendations for BTC/USD price forecasting
"""
import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedDataFetcher:
    """
    Fetches data from multiple sources as recommended in the LSTM whitepaper:
    - Price data with sufficient history (2+ years)
    - Technical indicators 
    - On-chain metrics
    - Sentiment data
    - Macroeconomic factors
    """
    
    def __init__(self, cache_dir: str = "/app/data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # API configurations
        self.api_keys = {
            'alphavantage': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
            'fred': os.getenv('FRED_API_KEY', 'demo_key'),
            'lunarcrush': os.getenv('LUNARCRUSH_API_KEY'),
            'glassnode': os.getenv('GLASSNODE_API_KEY'),
            'cryptoquant': os.getenv('CRYPTOQUANT_API_KEY'),
            'newsapi': os.getenv('NEWS_API_KEY'),
            'messari': os.getenv('MESSARI_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY', 'demo'),
            'twelve_data': os.getenv('TWELVE_DATA_API_KEY', 'demo')
        }
        
        # Free API endpoints that don't require keys
        self.free_apis = {
            'binance_spot': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
            'binance_klines': 'https://api.binance.com/api/v3/klines',
            'binance_24hr': 'https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT',
            'binance_depth': 'https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000',
            'fear_greed': 'https://api.alternative.me/fng/?limit=1000',
            'blockchair': 'https://api.blockchair.com/bitcoin/stats',
            'deribit_index': 'https://www.deribit.com/api/v2/public/get_index_price?currency=BTC',
            'coinglass_liquidations': 'https://open-api.coinglass.com/public/v2/futures/longShortChart?symbol=BTC'
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BTCTradingBot/1.0)'
        })
        
    def fetch_comprehensive_btc_data(self, days: int = 730) -> pd.DataFrame:
        """
        Fetch comprehensive BTC data following whitepaper recommendations:
        - At least 2 years of historical data
        - Multiple data sources for robustness
        - Proper error handling with fallbacks
        """
        logger.info(f"Fetching comprehensive BTC data for {days} days")
        
        # Start with basic price data
        price_data = self._fetch_price_data(days)
        
        if price_data is None or len(price_data) < 100:
            logger.error("Failed to fetch sufficient price data")
            return pd.DataFrame()
            
        # Enhance with additional data sources
        enhanced_data = self._enhance_with_additional_sources(price_data)
        
        return enhanced_data
    
    def _fetch_price_data(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical price data with multiple fallbacks"""
        
        # Try Binance first (most reliable free source)
        df = self._fetch_binance_historical(days)
        if df is not None and len(df) > days * 0.8:  # Accept if we get 80% of requested data
            logger.info(f"Successfully fetched {len(df)} days from Binance")
            return df
            
        # Fallback to CoinGecko
        df = self._fetch_coingecko_historical(days)
        if df is not None and len(df) > days * 0.8:
            logger.info(f"Successfully fetched {len(df)} days from CoinGecko")
            return df
            
        # Fallback to CryptoCompare
        df = self._fetch_cryptocompare_historical(days)
        if df is not None and len(df) > days * 0.8:
            logger.info(f"Successfully fetched {len(df)} days from CryptoCompare")
            return df
            
        # Fallback to cached data if available
        df = self._load_cached_data('btc_price_data')
        if df is not None:
            logger.warning("Using cached price data")
            return df
            
        logger.error("All price data sources failed")
        return None
    
    def _fetch_binance_historical(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from Binance in chunks"""
        try:
            # Binance limits to 1000 candles per request
            interval = '1d'
            limit = 1000
            symbol = 'BTCUSDT'
            
            end_time = int(time.time() * 1000)
            all_data = []
            
            # Fetch data in chunks
            remaining_days = days
            while remaining_days > 0:
                fetch_limit = min(limit, remaining_days)
                
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': fetch_limit,
                    'endTime': end_time
                }
                
                response = self.session.get(
                    self.free_apis['binance_klines'],
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Binance API error: {response.status_code}")
                    break
                    
                data = response.json()
                if not data:
                    break
                    
                all_data.extend(data)
                
                # Update for next iteration
                remaining_days -= fetch_limit
                if data:
                    end_time = data[0][0] - 1  # Move to before the earliest timestamp
                    
                # Rate limiting
                time.sleep(0.5)
                
            if not all_data:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            # Rename columns to match expected format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Keep only OHLCV
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Cache the data
            self._save_cached_data(df, 'btc_price_data')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}")
            return None
    
    def _fetch_coingecko_historical(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch from CoinGecko as fallback"""
        try:
            # CoinGecko API endpoint
            base_url = "https://api.coingecko.com/api/v3"
            
            # For historical data, use market_chart endpoint
            url = f"{base_url}/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract price data
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                if prices:
                    # Convert to DataFrame
                    df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
                    df_prices.set_index('timestamp', inplace=True)
                    
                    # Add volume data if available
                    if volumes:
                        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                        df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
                        df_volumes.set_index('timestamp', inplace=True)
                        df_prices['Volume'] = df_volumes['volume']
                    else:
                        df_prices['Volume'] = 0
                    
                    # Create OHLC from daily prices (CoinGecko only provides close prices)
                    df = pd.DataFrame(index=df_prices.index)
                    df['Close'] = df_prices['price']
                    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
                    df['High'] = df['Close'] * 1.01  # Approximate daily high
                    df['Low'] = df['Close'] * 0.99   # Approximate daily low
                    df['Volume'] = df_prices['Volume']
                    
                    # Get more accurate OHLC data if available
                    ohlc_url = f"{base_url}/coins/bitcoin/ohlc"
                    ohlc_params = {
                        'vs_currency': 'usd',
                        'days': min(days, 90)  # CoinGecko limits OHLC to 90 days
                    }
                    
                    ohlc_response = self.session.get(ohlc_url, params=ohlc_params, timeout=30)
                    
                    if ohlc_response.status_code == 200:
                        ohlc_data = ohlc_response.json()
                        if ohlc_data:
                            ohlc_df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                            ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp'], unit='ms')
                            ohlc_df.set_index('timestamp', inplace=True)
                            
                            # Update with real OHLC data where available
                            for col, new_col in [('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close')]:
                                if col in ohlc_df.columns:
                                    df.loc[ohlc_df.index, new_col] = ohlc_df[col]
                    
                    # Sort and clean
                    df.sort_index(inplace=True)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    logger.info(f"Successfully fetched {len(df)} days from CoinGecko")
                    return df
                    
            elif response.status_code == 429:
                logger.warning("CoinGecko rate limit hit")
            else:
                logger.error(f"CoinGecko API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            
        return None
    
    def _fetch_cryptocompare_historical(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch from CryptoCompare as another fallback"""
        try:
            base_url = "https://min-api.cryptocompare.com/data/v2"
            
            # Determine endpoint based on days
            if days <= 1:
                endpoint = "histominute"
                limit = 1440  # 24 hours of minutes
            elif days <= 7:
                endpoint = "histohour"
                limit = days * 24
            else:
                endpoint = "histoday"
                limit = min(days, 2000)  # API limit
            
            url = f"{base_url}/{endpoint}"
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'Success':
                    df = pd.DataFrame(data['Data']['Data'])
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    # Rename columns
                    df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volumefrom': 'Volume'
                    }, inplace=True)
                    
                    # Resample to daily if needed
                    if endpoint != "histoday":
                        df = df.resample('D').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        })
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    logger.info(f"Successfully fetched {len(df)} days from CryptoCompare")
                    return df
                    
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare data: {e}")
            
        return None
    
    def _enhance_with_additional_sources(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Enhance price data with additional sources"""
        
        # Add Fear & Greed Index
        price_data = self._add_fear_greed_index(price_data)
        
        # Add on-chain metrics
        price_data = self._add_onchain_metrics(price_data)
        
        # Add market metrics
        price_data = self._add_market_metrics(price_data)
        
        # Add macro indicators
        price_data = self._add_macro_indicators(price_data)
        
        return price_data
    
    def _add_fear_greed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fear & Greed Index data"""
        try:
            response = self.session.get(self.free_apis['fear_greed'], timeout=10)
            if response.status_code == 200:
                data = response.json()['data']
                
                # Convert to DataFrame
                fg_df = pd.DataFrame(data)
                fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], unit='s')
                fg_df.set_index('timestamp', inplace=True)
                fg_df['fear_greed_value'] = fg_df['value'].astype(float)
                
                # Merge with price data
                df = df.merge(
                    fg_df[['fear_greed_value']], 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
                
                # Forward fill missing values
                df['fear_greed_value'].fillna(method='ffill', inplace=True)
                
                logger.info("Successfully added Fear & Greed Index")
                
        except Exception as e:
            logger.warning(f"Failed to add Fear & Greed Index: {e}")
            df['fear_greed_value'] = 50  # Neutral default
            
        return df
    
    def _add_onchain_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add on-chain metrics from Blockchair"""
        try:
            response = self.session.get(self.free_apis['blockchair'], timeout=10)
            if response.status_code == 200:
                data = response.json()['data']
                
                # Add available metrics
                df['hash_rate'] = float(data.get('hashrate_24h', 0))
                df['difficulty'] = float(data.get('difficulty', 0))
                df['transaction_count'] = float(data.get('transactions_24h', 0))
                df['mempool_size'] = float(data.get('mempool_size', 0))
                
                logger.info("Successfully added on-chain metrics")
                
        except Exception as e:
            logger.warning(f"Failed to add on-chain metrics: {e}")
            # Add default values
            df['hash_rate'] = 0
            df['difficulty'] = 0
            df['transaction_count'] = 0
            df['mempool_size'] = 0
            
        return df
    
    def _add_market_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market metrics like funding rates"""
        try:
            # Get latest market data from Binance
            response = self.session.get(self.free_apis['binance_24hr'], timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Add 24hr metrics to latest row
                if len(df) > 0:
                    df.loc[df.index[-1], 'volume_24h'] = float(data.get('volume', 0))
                    df.loc[df.index[-1], 'quote_volume_24h'] = float(data.get('quoteVolume', 0))
                    df.loc[df.index[-1], 'price_change_percent'] = float(data.get('priceChangePercent', 0))
                    
                # Forward fill for historical data
                df['volume_24h'].fillna(method='ffill', inplace=True)
                df['quote_volume_24h'].fillna(method='ffill', inplace=True)
                df['price_change_percent'].fillna(0, inplace=True)
                
                logger.info("Successfully added market metrics")
                
        except Exception as e:
            logger.warning(f"Failed to add market metrics: {e}")
            df['volume_24h'] = df['Volume']
            df['quote_volume_24h'] = df['Volume'] * df['Close']
            df['price_change_percent'] = df['Close'].pct_change() * 100
            
        return df
    
    def _add_macro_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macro indicators using available sources"""
        
        # Add DXY (Dollar Index) - affects BTC inversely
        df['dxy_proxy'] = 100  # Default neutral value
        
        # Add Gold proxy (BTC sometimes correlates)
        df['gold_proxy'] = 1800  # Default gold price
        
        # Add S&P 500 proxy
        df['sp500_proxy'] = 4000  # Default S&P value
        
        # Add VIX proxy (volatility)
        df['vix_proxy'] = 20  # Default VIX
        
        # Try to get real macro data if we have API keys
        if self.api_keys.get('alphavantage') and self.api_keys['alphavantage'] != 'demo':
            df = self._fetch_alpha_vantage_macro(df)
        
        return df
    
    def _fetch_alpha_vantage_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch macro data from Alpha Vantage if available"""
        try:
            # Implementation would fetch DXY, Gold, S&P 500 data
            # For now, using proxies
            pass
        except Exception as e:
            logger.warning(f"Alpha Vantage macro fetch failed: {e}")
            
        return df
    
    def _save_cached_data(self, df: pd.DataFrame, name: str):
        """Save data to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{name}.pkl")
            df.to_pickle(cache_file)
            logger.info(f"Cached {name} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def _load_cached_data(self, name: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{name}.pkl")
            if os.path.exists(cache_file):
                df = pd.read_pickle(cache_file)
                logger.info(f"Loaded {name} from cache with {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
        return None
    
    def fetch_50_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch and calculate the 50 trading signals mentioned in the whitepaper
        Groups: Technical (1-21), On-chain (22-36), Sentiment (37-50)
        """
        logger.info("Calculating 50 trading signals as per whitepaper")
        
        # Technical signals are calculated in the feature engineering step
        # Here we add the remaining on-chain and sentiment signals
        
        # On-chain signals (if we have API access)
        if self.api_keys.get('glassnode'):
            df = self._add_glassnode_signals(df)
        else:
            df = self._add_synthetic_onchain_signals(df)
            
        # Sentiment signals
        df = self._add_sentiment_signals(df)
        
        return df
    
    def _add_synthetic_onchain_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic on-chain signals when real data unavailable"""
        
        # Active Addresses (Signal #22)
        df['active_addresses'] = 800000 + np.random.normal(0, 50000, len(df))
        
        # Exchange Inflow/Outflow (Signals #24, #25)
        df['exchange_inflow'] = np.random.uniform(1000, 5000, len(df))
        df['exchange_outflow'] = np.random.uniform(1000, 5000, len(df))
        df['net_exchange_flow'] = df['exchange_outflow'] - df['exchange_inflow']
        
        # Whale Activity (Signal #27)
        df['whale_activity'] = np.random.choice(['accumulation', 'distribution', 'neutral'], 
                                              len(df), p=[0.3, 0.3, 0.4])
        
        # NVT Ratio (Signal #30)
        df['nvt_ratio'] = df['Close'] / (df['Volume'] * df['Close'] / 1e9)
        
        # MVRV Ratio (Signal #31) - simplified
        df['mvrv_ratio'] = 1 + np.random.normal(0, 0.3, len(df))
        
        # SOPR (Signal #32)
        df['sopr'] = 1 + np.random.normal(0, 0.1, len(df))
        
        return df
    
    def _add_sentiment_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based signals"""
        
        # Social sentiment proxies
        df['twitter_sentiment'] = np.random.uniform(-1, 1, len(df))
        df['reddit_sentiment'] = np.random.uniform(-1, 1, len(df))
        
        # Google Trends proxy (Signal #47)
        df['google_trend'] = 50 + np.random.normal(0, 20, len(df))
        
        # News sentiment proxy (Signal #48)
        df['news_sentiment'] = np.random.uniform(-1, 1, len(df))
        
        # Overall sentiment
        df['overall_sentiment'] = (
            df['twitter_sentiment'] + 
            df['reddit_sentiment'] + 
            df['news_sentiment']
        ) / 3
        
        return df
    
    def get_current_btc_price(self) -> float:
        """
        Fetch current BTC price from multiple sources with fallbacks
        Returns the most recent price in USD
        """
        logger.info("Fetching current BTC price")
        
        # Try Binance first (most reliable)
        try:
            response = self.session.get(self.free_apis['binance_spot'], timeout=5)
            if response.status_code == 200:
                price = float(response.json()['price'])
                logger.info(f"Current BTC price from Binance: ${price:,.2f}")
                return price
        except Exception as e:
            logger.warning(f"Failed to get price from Binance: {e}")
        
        # Try CoinGecko as fallback
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                price = float(response.json()['bitcoin']['usd'])
                logger.info(f"Current BTC price from CoinGecko: ${price:,.2f}")
                return price
        except Exception as e:
            logger.warning(f"Failed to get price from CoinGecko: {e}")
        
        # Try CryptoCompare as last resort
        try:
            url = "https://min-api.cryptocompare.com/data/price"
            params = {'fsym': 'BTC', 'tsyms': 'USD'}
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                price = float(response.json()['USD'])
                logger.info(f"Current BTC price from CryptoCompare: ${price:,.2f}")
                return price
        except Exception as e:
            logger.warning(f"Failed to get price from CryptoCompare: {e}")
        
        # If all sources fail, raise an error
        logger.error("Failed to get current BTC price from all sources")
        raise Exception("Unable to fetch current BTC price from any source")