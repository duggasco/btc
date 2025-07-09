"""
Feature Engineering Module implementing 50 trading signals
as per LSTM whitepaper recommendations for BTC/USD forecasting
"""
import pandas as pd
import numpy as np
# Use ta library instead of TA-Lib for better compatibility
import ta

# Create compatibility wrapper for ta library
class TALibCompat:
    """Wrapper to make ta library compatible with TA-Lib function signatures"""
    
    @staticmethod
    def SMA(close, timeperiod):
        """Simple Moving Average"""
        return pd.Series(close).rolling(window=timeperiod).mean()
    
    @staticmethod
    def EMA(close, timeperiod):
        """Exponential Moving Average"""
        return pd.Series(close).ewm(span=timeperiod, adjust=False).mean()
    
    @staticmethod
    def RSI(close, timeperiod=14):
        """Relative Strength Index"""
        return ta.momentum.RSIIndicator(pd.Series(close), window=timeperiod).rsi()
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """MACD"""
        macd_ind = ta.trend.MACD(pd.Series(close), window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
        return macd_ind.macd(), macd_ind.macd_signal(), macd_ind.macd_diff()
    
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
        """Bollinger Bands"""
        bb_ind = ta.volatility.BollingerBands(pd.Series(close), window=timeperiod, window_dev=nbdevup)
        return bb_ind.bollinger_hband(), bb_ind.bollinger_mavg(), bb_ind.bollinger_lband()
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        """Stochastic Oscillator"""
        stoch_ind = ta.momentum.StochasticOscillator(pd.Series(high), pd.Series(low), pd.Series(close), 
                                                     window=fastk_period, smooth_window=slowk_period)
        return stoch_ind.stoch(), stoch_ind.stoch_signal()
    
    @staticmethod
    def ADX(high, low, close, timeperiod=14):
        """Average Directional Index"""
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).adx()
    
    @staticmethod
    def PLUS_DI(high, low, close, timeperiod=14):
        """Plus Directional Indicator"""
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).adx_pos()
    
    @staticmethod
    def MINUS_DI(high, low, close, timeperiod=14):
        """Minus Directional Indicator"""
        return ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).adx_neg()
    
    @staticmethod
    def OBV(close, volume):
        """On Balance Volume"""
        return ta.volume.OnBalanceVolumeIndicator(pd.Series(close), pd.Series(volume)).on_balance_volume()
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """Average True Range"""
        return ta.volatility.AverageTrueRange(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).average_true_range()
    
    @staticmethod
    def ROC(close, timeperiod=10):
        """Rate of Change"""
        return ta.momentum.ROCIndicator(pd.Series(close), window=timeperiod).roc()
    
    @staticmethod
    def SAR(high, low, acceleration=0.02, maximum=0.2):
        """Parabolic SAR"""
        # ta library doesn't have SAR, so we'll use a simple approximation
        close = (pd.Series(high) + pd.Series(low)) / 2
        return close.rolling(window=20).mean()
    
    @staticmethod
    def AROON(high, low, timeperiod=14):
        """Aroon Indicator"""
        # ta library's AroonIndicator only takes close price, so we'll use high-low midpoint
        close = (pd.Series(high) + pd.Series(low)) / 2
        aroon_ind = ta.trend.AroonIndicator(close=close, window=timeperiod)
        return aroon_ind.aroon_up(), aroon_ind.aroon_down()
    
    @staticmethod
    def CCI(high, low, close, timeperiod=14):
        """Commodity Channel Index"""
        return ta.trend.CCIIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=timeperiod).cci()
    
    @staticmethod
    def MFI(high, low, close, volume, timeperiod=14):
        """Money Flow Index"""
        return ta.volume.MFIIndicator(pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume), window=timeperiod).money_flow_index()
    
    # Candlestick patterns - simplified versions
    @staticmethod
    def CDLHAMMER(open_price, high, low, close):
        """Hammer pattern (simplified)"""
        body = abs(close - open_price)
        lower_shadow = np.minimum(open_price, close) - low
        return (lower_shadow > 2 * body).astype(int) * 100
    
    @staticmethod
    def CDLENGULFING(open_price, high, low, close):
        """Engulfing pattern (simplified)"""
        prev_body = abs(close.shift(1) - open_price.shift(1))
        curr_body = abs(close - open_price)
        bullish = ((open_price > close.shift(1)) & (close > open_price.shift(1)) & (curr_body > prev_body))
        return bullish.astype(int) * 100
    
    @staticmethod
    def CDLMORNINGSTAR(open_price, high, low, close):
        """Morning star pattern (simplified)"""
        # Simplified: look for reversal after downtrend
        return ((close > open_price) & (close.shift(2) < open_price.shift(2))).astype(int) * 100
    
    @staticmethod
    def CDLSHOOTINGSTAR(open_price, high, low, close):
        """Shooting star pattern (simplified)"""
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        return (upper_shadow > 2 * body).astype(int) * -100
    
    @staticmethod
    def CDLEVENINGSTAR(open_price, high, low, close):
        """Evening star pattern (simplified)"""
        # Simplified: look for reversal after uptrend
        return ((close < open_price) & (close.shift(2) > open_price.shift(2))).astype(int) * -100

# Use the compatibility wrapper
talib = TALibCompat()
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Implements the 50 trading signals categorized as:
    - Technical Analysis Signals (1-21)
    - On-Chain Fundamentals (22-36) 
    - Sentiment & Derivatives (37-50)
    
    With adaptive feature selection based on available data
    """
    
    def __init__(self, min_periods_ratio: float = 0.8):
        """
        Args:
            min_periods_ratio: Minimum ratio of valid data required after indicator calculation
        """
        self.min_periods_ratio = min_periods_ratio
        self.feature_importance = {}
        
    def engineer_features(self, df: pd.DataFrame, adaptive: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Engineer all features with adaptive selection based on data availability
        """
        logger.info(f"Starting feature engineering with {len(df)} rows")
        
        # Store original length
        original_length = len(df)
        
        # Create a copy to avoid modifying original
        features_df = df.copy()
        
        # Track which features we can calculate
        available_features = []
        
        # 1. Technical Analysis Features (Signals 1-21)
        features_df, tech_features = self._add_technical_features(features_df, adaptive)
        available_features.extend(tech_features)
        
        # 2. On-Chain Features (Signals 22-36) 
        features_df, onchain_features = self._add_onchain_features(features_df)
        available_features.extend(onchain_features)
        
        # 3. Sentiment & Market Structure Features (Signals 37-50)
        features_df, sentiment_features = self._add_sentiment_features(features_df)
        available_features.extend(sentiment_features)
        
        # 4. Additional engineered features
        features_df, eng_features = self._add_engineered_features(features_df)
        available_features.extend(eng_features)
        
        # Remove rows with too many NaN values
        before_dropna = len(features_df)
        threshold = int(len(available_features) * 0.3)  # Allow 30% missing values
        features_df = features_df.dropna(thresh=len(features_df.columns) - threshold)
        
        # Forward fill remaining NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Final fillna with 0 for any remaining
        features_df = features_df.fillna(0)
        
        after_processing = len(features_df)
        logger.info(f"Feature engineering complete. Rows: {original_length} -> {after_processing}")
        logger.info(f"Total features: {len(available_features)}")
        
        return features_df, available_features
    
    def _add_technical_features(self, df: pd.DataFrame, adaptive: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Add technical analysis features (Signals 1-21)"""
        features = []
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        volume = df['Volume']
        
        # Signal 1: Moving Averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period or not adaptive:
                col_name = f'SMA_{period}'
                df[col_name] = talib.SMA(close, timeperiod=period)
                features.append(col_name)
                
                col_name = f'EMA_{period}'
                df[col_name] = talib.EMA(close, timeperiod=period)
                features.append(col_name)
        
        # Only add long MAs if we have enough data
        if len(df) >= 200 or not adaptive:
            df['SMA_100'] = talib.SMA(close, timeperiod=100)
            df['SMA_200'] = talib.SMA(close, timeperiod=200)
            features.extend(['SMA_100', 'SMA_200'])
            
            # Golden/Death Cross
            if 'SMA_50' in df.columns:
                df['golden_cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
                df['death_cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)
                features.extend(['golden_cross', 'death_cross'])
        
        # Signal 2: RSI
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        features.extend(['RSI', 'RSI_oversold', 'RSI_overbought'])
        
        # Signal 3: MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_histogram'] = macd_hist
        df['MACD_bullish_cross'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)
        df['MACD_bearish_cross'] = ((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))).astype(int)
        features.extend(['MACD', 'MACD_signal', 'MACD_histogram', 'MACD_bullish_cross', 'MACD_bearish_cross'])
        
        # Signal 4: Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        df['BB_width'] = bb_upper - bb_lower
        df['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        df['BB_squeeze'] = df['BB_width'] / df['BB_middle']
        features.extend(['BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position', 'BB_squeeze'])
        
        # Signal 5: Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        df['STOCH_oversold'] = (slowk < 20).astype(int)
        df['STOCH_overbought'] = (slowk > 80).astype(int)
        features.extend(['STOCH_K', 'STOCH_D', 'STOCH_oversold', 'STOCH_overbought'])
        
        # Signal 6: ADX
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['ADX_strong_trend'] = (df['ADX'] > 25).astype(int)
        df['ADX_bullish'] = ((df['PLUS_DI'] > df['MINUS_DI']) & (df['ADX'] > 25)).astype(int)
        df['ADX_bearish'] = ((df['MINUS_DI'] > df['PLUS_DI']) & (df['ADX'] > 25)).astype(int)
        features.extend(['ADX', 'PLUS_DI', 'MINUS_DI', 'ADX_strong_trend', 'ADX_bullish', 'ADX_bearish'])
        
        # Signal 7: Fibonacci Retracement Levels (simplified)
        if len(df) >= 50:
            rolling_high = df['High'].rolling(50).max()
            rolling_low = df['Low'].rolling(50).min()
            fib_range = rolling_high - rolling_low
            
            df['FIB_0'] = rolling_low
            df['FIB_236'] = rolling_low + 0.236 * fib_range
            df['FIB_382'] = rolling_low + 0.382 * fib_range
            df['FIB_500'] = rolling_low + 0.500 * fib_range
            df['FIB_618'] = rolling_low + 0.618 * fib_range
            df['FIB_786'] = rolling_low + 0.786 * fib_range
            df['FIB_1000'] = rolling_high
            
            features.extend(['FIB_0', 'FIB_236', 'FIB_382', 'FIB_500', 'FIB_618', 'FIB_786', 'FIB_1000'])
        
        # Signal 8: VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['price_above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)
        features.extend(['VWAP', 'price_above_VWAP'])
        
        # Signal 9: Ichimoku Cloud (simplified for daily data)
        if len(df) >= 52:
            # Tenkan-sen (Conversion Line)
            high_9 = df['High'].rolling(9).max()
            low_9 = df['Low'].rolling(9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = df['High'].rolling(26).max()
            low_26 = df['Low'].rolling(26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            high_52 = df['High'].rolling(52).max()
            low_52 = df['Low'].rolling(52).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            df['chikou_span'] = df['Close'].shift(-26)
            
            # Price position relative to cloud
            df['price_above_cloud'] = (
                (df['Close'] > df['senkou_span_a']) & 
                (df['Close'] > df['senkou_span_b'])
            ).astype(int)
            
            df['bullish_cloud'] = (df['senkou_span_a'] > df['senkou_span_b']).astype(int)
            
            features.extend(['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                           'chikou_span', 'price_above_cloud', 'bullish_cloud'])
        
        # Signal 10: OBV (On-Balance Volume)
        df['OBV'] = talib.OBV(close, volume)
        df['OBV_trend'] = (df['OBV'] > df['OBV'].rolling(20).mean()).astype(int)
        features.extend(['OBV', 'OBV_trend'])
        
        # Signal 11: Candlestick Patterns (using TA-Lib pattern recognition)
        # Bullish patterns
        df['HAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
        df['BULLISH_ENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
        df['MORNING_STAR'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        
        # Bearish patterns
        df['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        df['BEARISH_ENGULFING'] = talib.CDLENGULFING(open_price, high, low, close) * -1
        df['EVENING_STAR'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        
        # Aggregate patterns
        df['bullish_pattern'] = (
            (df['HAMMER'] > 0) | 
            (df['BULLISH_ENGULFING'] > 0) | 
            (df['MORNING_STAR'] > 0)
        ).astype(int)
        
        df['bearish_pattern'] = (
            (df['SHOOTING_STAR'] < 0) | 
            (df['BEARISH_ENGULFING'] < 0) | 
            (df['EVENING_STAR'] < 0)
        ).astype(int)
        
        features.extend(['bullish_pattern', 'bearish_pattern'])
        
        # Signal 12: Support and Resistance (using pivot points)
        df['pivot'] = (high + low + close) / 3
        df['resistance_1'] = 2 * df['pivot'] - low
        df['support_1'] = 2 * df['pivot'] - high
        df['resistance_2'] = df['pivot'] + (high - low)
        df['support_2'] = df['pivot'] - (high - low)
        features.extend(['pivot', 'resistance_1', 'support_1', 'resistance_2', 'support_2'])
        
        # Signal 13: Volume Spikes
        df['volume_spike'] = (volume > volume.rolling(20).mean() * 2).astype(int)
        df['volume_ma'] = talib.SMA(volume, timeperiod=20)
        features.extend(['volume_spike', 'volume_ma'])
        
        # Signal 14: ATR (Average True Range)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_normalized'] = df['ATR'] / close
        features.extend(['ATR', 'ATR_normalized'])
        
        # Signal 15: ROC (Rate of Change)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['ROC_positive'] = (df['ROC'] > 0).astype(int)
        features.extend(['ROC', 'ROC_positive'])
        
        # Signal 16: Parabolic SAR
        df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        df['SAR_trend'] = (close > df['SAR']).astype(int)
        features.extend(['SAR', 'SAR_trend'])
        
        # Signal 17: Aroon
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
        df['AROON_up'] = aroon_up
        df['AROON_down'] = aroon_down
        df['AROON_bullish'] = (aroon_up > aroon_down).astype(int)
        features.extend(['AROON_up', 'AROON_down', 'AROON_bullish'])
        
        # Signal 18: CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['CCI_overbought'] = (df['CCI'] > 100).astype(int)
        df['CCI_oversold'] = (df['CCI'] < -100).astype(int)
        features.extend(['CCI', 'CCI_overbought', 'CCI_oversold'])
        
        # Signal 19: CMF (Chaikin Money Flow)
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle division by zero
        mf_volume = mfm * volume
        df['CMF'] = mf_volume.rolling(20).sum() / volume.rolling(20).sum()
        df['CMF_positive'] = (df['CMF'] > 0).astype(int)
        features.extend(['CMF', 'CMF_positive'])
        
        # Signal 20: MFI (Money Flow Index)
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['MFI_overbought'] = (df['MFI'] > 80).astype(int)
        df['MFI_oversold'] = (df['MFI'] < 20).astype(int)
        features.extend(['MFI', 'MFI_overbought', 'MFI_oversold'])
        
        # Signal 21: Divergences (simplified)
        if 'RSI' in df.columns and len(df) >= 20:
            # RSI divergence
            price_higher = (close > close.shift(20)) & (close.shift(20) > 0)
            rsi_lower = (df['RSI'] < df['RSI'].shift(20)) & (df['RSI'].shift(20) > 0)
            df['RSI_bearish_divergence'] = (price_higher & rsi_lower).astype(int)
            
            price_lower = (close < close.shift(20)) & (close.shift(20) > 0)
            rsi_higher = (df['RSI'] > df['RSI'].shift(20)) & (df['RSI'].shift(20) > 0)
            df['RSI_bullish_divergence'] = (price_lower & rsi_higher).astype(int)
            
            features.extend(['RSI_bearish_divergence', 'RSI_bullish_divergence'])
            
            # MACD divergence
            if 'MACD_histogram' in df.columns:
                macd_lower = (df['MACD_histogram'] < df['MACD_histogram'].shift(20))
                df['MACD_bearish_divergence'] = (price_higher & macd_lower).astype(int)
                
                macd_higher = (df['MACD_histogram'] > df['MACD_histogram'].shift(20))
                df['MACD_bullish_divergence'] = (price_lower & macd_higher).astype(int)
                
                features.extend(['MACD_bearish_divergence', 'MACD_bullish_divergence'])
        
        return df, features
    
    def _add_onchain_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add on-chain features (Signals 22-36)"""
        features = []
        
        # These would come from the data fetcher
        # For now, we'll use what's available or create proxies
        
        # Signal 22: Active Addresses
        if 'active_addresses' in df.columns:
            df['active_addresses_ma'] = df['active_addresses'].rolling(7).mean()
            df['active_addresses_growth'] = df['active_addresses'].pct_change(7)
            features.extend(['active_addresses', 'active_addresses_ma', 'active_addresses_growth'])
        
        # Signal 23: On-chain Transaction Volume
        if 'transaction_count' in df.columns:
            df['tx_volume_ma'] = df['transaction_count'].rolling(7).mean()
            df['tx_volume_growth'] = df['transaction_count'].pct_change(7)
            features.extend(['transaction_count', 'tx_volume_ma', 'tx_volume_growth'])
        
        # Signal 24 & 25: Exchange Flows
        if 'exchange_inflow' in df.columns and 'exchange_outflow' in df.columns:
            df['net_exchange_flow'] = df['exchange_outflow'] - df['exchange_inflow']
            df['exchange_flow_ratio'] = df['exchange_outflow'] / (df['exchange_inflow'] + 1)
            features.extend(['exchange_inflow', 'exchange_outflow', 'net_exchange_flow', 'exchange_flow_ratio'])
        
        # Signal 26: Stablecoin Flows (proxy)
        if 'volume_24h' in df.columns:
            # Use volume as proxy for stablecoin activity
            df['stablecoin_proxy'] = df['volume_24h'].rolling(3).mean()
            features.append('stablecoin_proxy')
        
        # Signal 27: Whale Activity
        if 'whale_activity' in df.columns:
            df['whale_accumulation'] = (df['whale_activity'] == 'accumulation').astype(int)
            df['whale_distribution'] = (df['whale_activity'] == 'distribution').astype(int)
            features.extend(['whale_accumulation', 'whale_distribution'])
        
        # Signal 28 & 29: Hash Rate and Mining
        if 'hash_rate' in df.columns:
            df['hash_rate_ma'] = df['hash_rate'].rolling(7).mean()
            df['hash_rate_growth'] = df['hash_rate'].pct_change(30)
            features.extend(['hash_rate', 'hash_rate_ma', 'hash_rate_growth'])
        
        if 'difficulty' in df.columns:
            features.append('difficulty')
        
        # Signal 30: NVT Ratio
        if 'nvt_ratio' in df.columns:
            df['nvt_signal'] = df['nvt_ratio'].rolling(14).mean()
            features.extend(['nvt_ratio', 'nvt_signal'])
        
        # Signal 31: MVRV Ratio
        if 'mvrv_ratio' in df.columns:
            df['mvrv_extreme'] = ((df['mvrv_ratio'] > 3) | (df['mvrv_ratio'] < 1)).astype(int)
            features.extend(['mvrv_ratio', 'mvrv_extreme'])
        
        # Signal 32: SOPR
        if 'sopr' in df.columns:
            df['sopr_profit'] = (df['sopr'] > 1).astype(int)
            features.extend(['sopr', 'sopr_profit'])
        
        # Signal 33: Exchange Reserves (use net flow as proxy)
        if 'net_exchange_flow' in df.columns:
            df['exchange_reserves_proxy'] = df['net_exchange_flow'].cumsum()
            features.append('exchange_reserves_proxy')
        
        # Signal 34: DeFi TVL (not available for BTC, skip)
        
        # Signal 35: Bitcoin Dominance (would need market cap data)
        # Using volume dominance as proxy
        if 'volume_24h' in df.columns:
            df['volume_dominance'] = df['volume_24h'] / df['volume_24h'].rolling(30).mean()
            features.append('volume_dominance')
        
        # Signal 36: Developer Activity (not applicable for Bitcoin)
        
        return df, features
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add sentiment and market structure features (Signals 37-50)"""
        features = []
        
        # Signal 37-42: Derivatives (would need futures/options data)
        # For now, using price/volume patterns as proxies
        
        # Funding rate proxy (using price momentum)
        df['funding_proxy'] = df['Close'].pct_change(1) * 100
        features.append('funding_proxy')
        
        # Open interest proxy (using volume patterns)
        if 'Volume' in df.columns:
            df['oi_proxy'] = df['Volume'].rolling(7).mean() / df['Volume'].rolling(30).mean()
            features.append('oi_proxy')
        
        # Volatility as proxy for options activity
        df['volatility_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(365)
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(60).mean()
        features.extend(['volatility_20', 'volatility_ratio'])
        
        # Signal 43: Fear & Greed Index
        if 'fear_greed_value' in df.columns:
            df['fear_extreme'] = (df['fear_greed_value'] < 25).astype(int)
            df['greed_extreme'] = (df['fear_greed_value'] > 75).astype(int)
            features.extend(['fear_greed_value', 'fear_extreme', 'greed_extreme'])
        
        # Signal 44-46: Social Sentiment
        if 'twitter_sentiment' in df.columns:
            features.append('twitter_sentiment')
        if 'reddit_sentiment' in df.columns:
            features.append('reddit_sentiment')
        if 'overall_sentiment' in df.columns:
            features.append('overall_sentiment')
        
        # Signal 47: Google Trends
        if 'google_trend' in df.columns:
            df['google_trend_ma'] = df['google_trend'].rolling(7).mean()
            features.extend(['google_trend', 'google_trend_ma'])
        
        # Signal 48: News Sentiment
        if 'news_sentiment' in df.columns:
            features.append('news_sentiment')
        
        # Signal 49: Whale Alerts (covered in on-chain)
        
        # Signal 50: Leverage Ratio proxy
        if 'Volume' in df.columns:
            # High volume relative to price movement suggests leverage
            price_change = df['Close'].pct_change().abs()
            df['leverage_proxy'] = df['Volume'] / (price_change + 0.001)
            df['leverage_proxy'] = df['leverage_proxy'] / df['leverage_proxy'].rolling(30).mean()
            features.append('leverage_proxy')
        
        return df, features
    
    def _add_engineered_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Add additional engineered features"""
        features = []
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Price position features
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Trend features
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['trend_strength'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
            features.append('trend_strength')
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['price_volume_trend'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        # Market microstructure
        df['spread'] = df['High'] - df['Low']
        df['spread_pct'] = df['spread'] / df['Close']
        
        # Momentum features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Higher highs and lower lows
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        features.extend([
            'returns', 'log_returns', 'volatility', 'high_low_ratio', 
            'close_open_ratio', 'volume_ratio', 'price_volume_trend',
            'spread', 'spread_pct', 'momentum_5', 'momentum_10', 
            'momentum_20', 'higher_high', 'lower_low'
        ])
        
        return df, features
    
    def select_features(self, df: pd.DataFrame, features: List[str], 
                       target_col: str = 'Close', max_features: int = 50) -> List[str]:
        """
        Select most important features based on correlation and mutual information
        """
        # Remove target column from features if present
        features = [f for f in features if f != target_col and f in df.columns]
        
        # Calculate correlations with target
        correlations = {}
        for feature in features:
            if df[feature].std() > 0:  # Skip constant features
                corr = df[feature].corr(df[target_col])
                if not np.isnan(corr):
                    correlations[feature] = abs(corr)
        
        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected = [f[0] for f in sorted_features[:max_features]]
        
        logger.info(f"Selected {len(selected)} features based on correlation")
        
        return selected