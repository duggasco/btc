"""
Enhanced Backtesting System with 50+ Trading Signals
Maintains ALL original functionality while adding comprehensive signal suite
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import optuna
from sklearn.preprocessing import MinMaxScaler
import logging
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== ORIGINAL CLASSES (PRESERVED) ==========

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    training_window_days: int = 1008  # 3-4 years as per research
    test_window_days: int = 90  # 3 months
    purge_days: int = 2  # Prevent information leakage
    retraining_frequency_days: int = 90  # Quarterly retraining
    min_train_test_ratio: float = 0.7  # 70% training minimum
    transaction_cost: float = 0.0025  # 0.25% per trade
    max_drawdown_threshold: float = 0.25  # 25% max acceptable drawdown
    target_sortino_ratio: float = 2.0  # Target Sortino ratio
    # Trading strategy parameters (can be set dynamically)
    position_size: Optional[float] = None
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None
    sell_percentage: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
@dataclass
class SignalWeights:
    """Optimizable signal weights for feature importance"""
    technical_weight: float = 0.40  # 40% technical indicators
    onchain_weight: float = 0.35   # 35% on-chain metrics
    sentiment_weight: float = 0.15  # 15% sentiment data
    macro_weight: float = 0.10      # 10% macroeconomic factors
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = self.technical_weight + self.onchain_weight + self.sentiment_weight + self.macro_weight
        if total > 0:
            self.technical_weight /= total
            self.onchain_weight /= total
            self.sentiment_weight /= total
            self.macro_weight /= total

# ========== NEW ENHANCED SIGNAL WEIGHTS ==========

@dataclass
class EnhancedSignalWeights(SignalWeights):
    """Extended signal weights including sub-categories from 50 signals"""
    # Technical sub-weights
    momentum_weight: float = 0.30      # RSI, ROC, Stochastic
    trend_weight: float = 0.40         # MA, MACD, ADX
    volatility_weight: float = 0.15    # Bollinger, ATR
    volume_weight: float = 0.15        # OBV, Volume spikes
    
    # On-chain sub-weights  
    flow_weight: float = 0.40          # Exchange in/outflows
    network_weight: float = 0.30       # Active addresses, tx volume
    holder_weight: float = 0.30        # Whale activity, HODL metrics
    
    # Sentiment sub-weights
    social_weight: float = 0.50        # Twitter, Reddit sentiment
    derivatives_weight: float = 0.30   # Funding, OI, Put/Call
    fear_greed_weight: float = 0.20    # F&G index, Google trends
    
    def normalize_subcategories(self):
        """Normalize sub-category weights"""
        # Technical
        tech_total = self.momentum_weight + self.trend_weight + self.volatility_weight + self.volume_weight
        if tech_total > 0:
            self.momentum_weight /= tech_total
            self.trend_weight /= tech_total
            self.volatility_weight /= tech_total
            self.volume_weight /= tech_total
            
        # On-chain
        chain_total = self.flow_weight + self.network_weight + self.holder_weight
        if chain_total > 0:
            self.flow_weight /= chain_total
            self.network_weight /= chain_total
            self.holder_weight /= chain_total
            
        # Sentiment
        sent_total = self.social_weight + self.derivatives_weight + self.fear_greed_weight
        if sent_total > 0:
            self.social_weight /= sent_total
            self.derivatives_weight /= sent_total
            self.fear_greed_weight /= sent_total

# ========== ENHANCED SIGNAL CALCULATOR ==========

class ComprehensiveSignalCalculator:
    """Calculate all 50+ trading signals from the research"""
    
    def __init__(self):
        self.lookback_periods = {
            'rsi': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_period': 14,
            'adx_period': 14,
            'atr_period': 14,
            'roc_period': 10,
            'aroon_period': 25,
            'cci_period': 20,
            'cmf_period': 20,
            'mfi_period': 14
        }
        
    def calculate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical, on-chain, and sentiment signals"""
        signals = data.copy()
        
        # Technical Indicators
        signals = self._calculate_technical_signals(signals)
        
        # On-chain signals (if available in data)
        signals = self._calculate_onchain_signals(signals)
        
        # Sentiment signals (if available in data)  
        signals = self._calculate_sentiment_signals(signals)
        
        # Derivative signals (if available in data)
        signals = self._calculate_derivatives_signals(signals)
        
        return signals
    
    def _calculate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        # 1. Moving Averages
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MA Crossovers
        df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
        
        # 2. RSI
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70
        
        # 3. MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['Close'])
        df['macd_bullish_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_bearish_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # 4. Bollinger Bands
        df['bb_middle'], df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['Close'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 5. Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        df['stoch_oversold'] = df['stoch_k'] < 20
        df['stoch_overbought'] = df['stoch_k'] > 80
        
        # 6. ADX
        df['adx'], df['plus_di'], df['minus_di'] = self._calculate_adx(df)
        df['adx_strong_trend'] = df['adx'] > 25
        df['adx_bullish'] = (df['plus_di'] > df['minus_di']) & df['adx_strong_trend']
        df['adx_bearish'] = (df['minus_di'] > df['plus_di']) & df['adx_strong_trend']
        
        # 7. Fibonacci Retracements (dynamic)
        df = self._calculate_fibonacci_levels(df)
        
        # 8. VWAP (if volume available)
        if 'Volume' in df.columns:
            df['vwap'] = self._calculate_vwap(df)
            df['price_above_vwap'] = df['Close'] > df['vwap']
        
        # 9. Ichimoku Cloud (simplified)
        df = self._calculate_ichimoku(df)
        
        # 10. OBV
        if 'Volume' in df.columns:
            df['obv'] = self._calculate_obv(df)
            df['obv_trend'] = df['obv'].rolling(20).mean()
        
        # 11. ATR
        df['atr'] = self._calculate_atr(df)
        df['atr_normalized'] = df['atr'] / df['Close']
        
        # 12. ROC
        df['roc'] = self._calculate_roc(df['Close'])
        df['roc_positive'] = df['roc'] > 0
        
        # 13. Parabolic SAR
        df['sar'], df['sar_trend'] = self._calculate_parabolic_sar(df)
        
        # 14. Aroon
        df['aroon_up'], df['aroon_down'] = self._calculate_aroon(df)
        df['aroon_bullish'] = df['aroon_up'] > df['aroon_down']
        
        # 15. CCI
        df['cci'] = self._calculate_cci(df)
        df['cci_overbought'] = df['cci'] > 100
        df['cci_oversold'] = df['cci'] < -100
        
        # 16. CMF
        if 'Volume' in df.columns:
            df['cmf'] = self._calculate_cmf(df)
            df['cmf_positive'] = df['cmf'] > 0
        
        # 17. MFI
        if 'Volume' in df.columns:
            df['mfi'] = self._calculate_mfi(df)
            df['mfi_overbought'] = df['mfi'] > 80
            df['mfi_oversold'] = df['mfi'] < 20
        
        # 18. Divergences
        df = self._calculate_divergences(df)
        
        # 19. Support/Resistance Levels
        df = self._calculate_support_resistance(df)
        
        # 20. Volume Analysis
        if 'Volume' in df.columns:
            df['volume_spike'] = df['Volume'] > df['Volume'].rolling(20).mean() * 2
            df['volume_ma'] = df['Volume'].rolling(20).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=self.lookback_periods['macd_fast']).mean()
        ema_slow = prices.ewm(span=self.lookback_periods['macd_slow']).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.lookback_periods['macd_signal']).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(self.lookback_periods['bb_period']).mean()
        std = prices.rolling(self.lookback_periods['bb_period']).std()
        upper = middle + (std * self.lookback_periods['bb_std'])
        lower = middle - (std * self.lookback_periods['bb_std'])
        return middle, upper, lower
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        period = self.lookback_periods['stoch_period']
        low_min = df['Low'].rolling(period).min()
        high_max = df['High'].rolling(period).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(3).mean()
        return k, d
    
    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX"""
        period = self.lookback_periods['adx_period']
        
        # True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        """Calculate dynamic Fibonacci levels"""
        # Find recent swing high and low
        recent_high = df['High'].rolling(lookback).max()
        recent_low = df['Low'].rolling(lookback).min()
        
        # Fibonacci levels
        diff = recent_high - recent_low
        df['fib_0'] = recent_low
        df['fib_236'] = recent_low + 0.236 * diff
        df['fib_382'] = recent_low + 0.382 * diff
        df['fib_500'] = recent_low + 0.500 * diff
        df['fib_618'] = recent_low + 0.618 * diff
        df['fib_786'] = recent_low + 0.786 * diff
        df['fib_1000'] = recent_high
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud"""
        # Conversion Line
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        # Base Line
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        # Leading Span A
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Leading Span B
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Lagging Span
        df['chikou_span'] = df['Close'].shift(-26)
        
        # Cloud signals
        df['price_above_cloud'] = df['Close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['bullish_cloud'] = df['senkou_span_a'] > df['senkou_span_b']
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.lookback_periods['atr_period']).mean()
        
        return atr
    
    def _calculate_roc(self, prices: pd.Series) -> pd.Series:
        """Calculate Rate of Change"""
        period = self.lookback_periods['roc_period']
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Parabolic SAR"""
        # Simplified implementation
        sar = df['Close'].copy()
        trend = pd.Series(1, index=df.index)  # 1 for uptrend, -1 for downtrend
        
        # This is a simplified version - full implementation would track EP and AF
        for i in range(1, len(df)):
            if trend.iloc[i-1] == 1:  # Uptrend
                if df['Low'].iloc[i] < sar.iloc[i-1]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = df['High'].iloc[i]
                else:
                    sar.iloc[i] = sar.iloc[i-1]
            else:  # Downtrend
                if df['High'].iloc[i] > sar.iloc[i-1]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = df['Low'].iloc[i]
                else:
                    sar.iloc[i] = sar.iloc[i-1]
        
        return sar, trend
    
    def _calculate_aroon(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon Indicator"""
        period = self.lookback_periods['aroon_period']
        
        aroon_up = df['High'].rolling(period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100, raw=True
        )
        aroon_down = df['Low'].rolling(period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100, raw=True
        )
        
        return aroon_up, aroon_down
    
    def _calculate_cci(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Commodity Channel Index"""
        period = self.lookback_periods['cci_period']
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    def _calculate_cmf(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        period = self.lookback_periods['cmf_period']
        
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        mf_volume = mf_multiplier * df['Volume']
        
        cmf = mf_volume.rolling(period).sum() / df['Volume'].rolling(period).sum()
        return cmf
    
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Money Flow Index"""
        period = self.lookback_periods['mfi_period']
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi
    
    def _calculate_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect bullish and bearish divergences"""
        # RSI Divergence
        price_lows = df['Low'].rolling(20).min() == df['Low']
        price_highs = df['High'].rolling(20).max() == df['High']
        
        rsi_lows = df['rsi'].rolling(20).min() == df['rsi']
        rsi_highs = df['rsi'].rolling(20).max() == df['rsi']
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        df['rsi_bullish_divergence'] = price_lows & (df['rsi'] > df['rsi'].shift(20))
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        df['rsi_bearish_divergence'] = price_highs & (df['rsi'] < df['rsi'].shift(20))
        
        # Similar for MACD
        df['macd_bullish_divergence'] = price_lows & (df['macd'] > df['macd'].shift(20))
        df['macd_bearish_divergence'] = price_highs & (df['macd'] < df['macd'].shift(20))
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        # Simple implementation using recent highs/lows
        df['resistance_1'] = df['High'].rolling(lookback).max()
        df['support_1'] = df['Low'].rolling(lookback).min()
        
        # Pivot points
        df['pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['resistance_2'] = 2 * df['pivot'] - df['Low']
        df['support_2'] = 2 * df['pivot'] - df['High']
        
        return df
    
    def _calculate_onchain_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate on-chain signals if data is available"""
        # Active Addresses
        if 'active_addresses' in df.columns:
            df['active_addr_ma'] = df['active_addresses'].rolling(7).mean()
            df['active_addr_growth'] = df['active_addresses'].pct_change(7)
            df['active_addr_spike'] = df['active_addresses'] > df['active_addr_ma'] * 1.5
        
        # Transaction Volume
        if 'tx_volume' in df.columns:
            df['tx_volume_ma'] = df['tx_volume'].rolling(7).mean()
            df['tx_volume_growth'] = df['tx_volume'].pct_change(7)
        
        # Exchange Flows
        if 'exchange_inflow' in df.columns:
            df['net_exchange_flow'] = df.get('exchange_outflow', 0) - df['exchange_inflow']
            df['exchange_flow_ma'] = df['net_exchange_flow'].rolling(7).mean()
            df['bullish_exchange_flow'] = df['net_exchange_flow'] > 0
        
        # Whale Activity
        if 'whale_count' in df.columns:
            df['whale_accumulation'] = df['whale_count'].diff() > 0
            df['whale_distribution'] = df['whale_count'].diff() < 0
        
        # Hash Rate
        if 'hash_rate' in df.columns:
            df['hash_rate_ma'] = df['hash_rate'].rolling(30).mean()
            df['hash_rate_growth'] = df['hash_rate'].pct_change(30)
        
        # Miner Activity
        if 'miner_outflow' in df.columns:
            df['miner_selling'] = df['miner_outflow'] > df['miner_outflow'].rolling(30).mean() * 2
        
        # NVT Ratio
        if 'network_value' in df.columns and 'tx_volume' in df.columns:
            df['nvt_ratio'] = df['network_value'] / (df['tx_volume'] + 1e-10)
            df['nvt_high'] = df['nvt_ratio'] > df['nvt_ratio'].rolling(90).quantile(0.8)
            df['nvt_low'] = df['nvt_ratio'] < df['nvt_ratio'].rolling(90).quantile(0.2)
        
        # MVRV Ratio
        if 'mvrv_ratio' in df.columns:
            df['mvrv_high'] = df['mvrv_ratio'] > 3.0
            df['mvrv_low'] = df['mvrv_ratio'] < 1.0
        
        # SOPR
        if 'sopr' in df.columns:
            df['sopr_bullish'] = (df['sopr'] < 1.0) & (df['sopr'].shift() >= 1.0)
            df['sopr_bearish'] = (df['sopr'] > 1.0) & (df['sopr'].shift() <= 1.0)
        
        # Stablecoin Flows
        if 'stablecoin_supply' in df.columns:
            df['stablecoin_growth'] = df['stablecoin_supply'].pct_change(7)
            df['stablecoin_bullish'] = df['stablecoin_growth'] > 0.01
        
        return df
    
    def _calculate_sentiment_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment signals if data is available"""
        # Fear & Greed Index
        if 'fear_greed' in df.columns:
            df['extreme_fear'] = df['fear_greed'] < 25
            df['extreme_greed'] = df['fear_greed'] > 75
            df['fear_greed_ma'] = df['fear_greed'].rolling(7).mean()
        
        # Social Volume
        if 'social_volume' in df.columns:
            df['social_volume_ma'] = df['social_volume'].rolling(24).mean()  # 24h for hourly data
            df['social_spike'] = df['social_volume'] > df['social_volume_ma'] * 2
        
        # Twitter Sentiment
        if 'twitter_sentiment' in df.columns:
            df['twitter_bullish'] = df['twitter_sentiment'] > 0.6
            df['twitter_bearish'] = df['twitter_sentiment'] < 0.4
            df['twitter_sentiment_ma'] = df['twitter_sentiment'].rolling(24).mean()
        
        # Reddit Activity
        if 'reddit_mentions' in df.columns:
            df['reddit_spike'] = df['reddit_mentions'] > df['reddit_mentions'].rolling(24).mean() * 1.5
        
        # Google Trends
        if 'google_trends' in df.columns:
            df['google_interest_high'] = df['google_trends'] > 80
            df['google_interest_low'] = df['google_trends'] < 20
        
        # News Sentiment
        if 'news_sentiment' in df.columns:
            df['news_positive'] = df['news_sentiment'] > 0.6
            df['news_negative'] = df['news_sentiment'] < 0.4
        
        return df
    
    def _calculate_derivatives_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derivatives market signals if data is available"""
        # Funding Rate
        if 'funding_rate' in df.columns:
            df['funding_extreme_positive'] = df['funding_rate'] > 0.001  # 0.1% per 8h
            df['funding_extreme_negative'] = df['funding_rate'] < -0.0005
            df['funding_ma'] = df['funding_rate'].rolling(24).mean()
        
        # Open Interest
        if 'open_interest' in df.columns:
            df['oi_growth'] = df['open_interest'].pct_change(24)
            df['oi_spike'] = df['oi_growth'] > 0.1  # 10% growth
            df['oi_decline'] = df['oi_growth'] < -0.1
        
        # Long/Short Ratio
        if 'long_short_ratio' in df.columns:
            df['extreme_longs'] = df['long_short_ratio'] > 1.5
            df['extreme_shorts'] = df['long_short_ratio'] < 0.67
        
        # Futures Basis
        if 'futures_basis' in df.columns:
            df['contango'] = df['futures_basis'] > 0
            df['backwardation'] = df['futures_basis'] < 0
            df['extreme_contango'] = df['futures_basis'] > 0.1  # 10% annualized
        
        # Put/Call Ratio
        if 'put_call_ratio' in df.columns:
            df['high_put_call'] = df['put_call_ratio'] > 1.2
            df['low_put_call'] = df['put_call_ratio'] < 0.8
        
        # Implied Volatility
        if 'implied_volatility' in df.columns:
            df['iv_high'] = df['implied_volatility'] > df['implied_volatility'].rolling(30).quantile(0.8)
            df['iv_low'] = df['implied_volatility'] < df['implied_volatility'].rolling(30).quantile(0.2)
        
        # Leverage Ratio
        if 'leverage_ratio' in df.columns:
            df['high_leverage'] = df['leverage_ratio'] > df['leverage_ratio'].rolling(30).quantile(0.9)
        
        return df

# ========== ORIGINAL PERFORMANCE METRICS CLASS (PRESERVED) ==========

class PerformanceMetrics:
    """Calculate crypto-specific performance metrics"""
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio focusing on downside deviation"""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        annual_return = np.mean(excess_returns) * 252
        
        return annual_return / downside_deviation if downside_deviation > 0 else float('inf')
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    @staticmethod
    def maximum_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        if len(cumulative_returns) == 0:
            return 0.0
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if len(returns) == 0:
            return 1.0
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else float('inf')
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate win rate percentage"""
        if len(returns) == 0:
            return 0.5
        return (returns > 0).sum() / len(returns)
    
    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        active_returns = returns - benchmark_returns
        if active_returns.std() == 0:
            return 0
        return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))

# ========== ENHANCED PERFORMANCE METRICS ==========

class EnhancedPerformanceMetrics(PerformanceMetrics):
    """Extended metrics including portfolio-specific measures"""
    
    @staticmethod
    def realized_pnl(trades: pd.DataFrame) -> float:
        """Calculate realized P&L from completed trades"""
        if trades.empty:
            return 0.0
        
        buy_value = trades[trades['side'] == 'buy']['value'].sum()
        sell_value = trades[trades['side'] == 'sell']['value'].sum()
        return sell_value - buy_value
    
    @staticmethod
    def unrealized_pnl(positions: pd.DataFrame, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L on open positions"""
        if positions.empty:
            return 0.0
        
        total_pnl = 0.0
        for _, pos in positions.iterrows():
            current_price = current_prices.get(pos['symbol'], pos['entry_price'])
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            total_pnl += pnl
        
        return total_pnl
    
    @staticmethod
    def portfolio_volatility(returns: np.ndarray, window: int = 30) -> pd.Series:
        """Calculate rolling portfolio volatility"""
        return pd.Series(returns).rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def conditional_value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        var = EnhancedPerformanceMetrics.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return float('inf')
        
        return gains.sum() / losses.sum()

# ========== ENHANCED WALK-FORWARD BACKTESTER ==========

class WalkForwardBacktester:
    """Implements walk-forward analysis for LSTM crypto trading"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_history = []
        
    def create_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward train/test splits with adaptive sizing"""
        splits = []
        total_days = len(data)
        
        # Adaptive parameters based on data size
        if total_days < 200:
            training_window = max(60, int(total_days * 0.6))
            test_window = max(20, int(total_days * 0.2))
        else:
            training_window = min(self.config.training_window_days, int(total_days * 0.7))
            test_window = min(self.config.test_window_days, int(total_days * 0.2))
        
        # FIXED: Move start_idx and while loop outside the if/else
        start_idx = 0
        
        while start_idx + training_window + self.config.purge_days + test_window <= total_days:
            # Training data
            train_end = start_idx + training_window
            train_data = data.iloc[start_idx:train_end]
            
            # Purge period to prevent information leakage
            test_start = train_end + self.config.purge_days
            test_end = test_start + test_window
            test_data = data.iloc[test_start:test_end]
            
            splits.append((train_data, test_data))
            
            # Move forward by retraining frequency
            start_idx += self.config.retraining_frequency_days
                
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def backtest_strategy(self, model, data: pd.DataFrame, signal_weights: SignalWeights) -> Dict:
        """Run complete walk-forward backtest"""
        splits = self.create_walk_forward_splits(data)
        all_results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Apply signal weights to features
            weighted_features = self._apply_signal_weights(train_data, signal_weights)
            
            # Train model on this split
            # FIX: Ensure target is 1D by using .ravel()
            target = train_data['target'].values.ravel() if 'target' in train_data else np.zeros(len(train_data))
            model.fit(weighted_features, target)
            
            # Generate predictions
            test_features = self._apply_signal_weights(test_data, signal_weights)
            predictions = model.predict(test_features)
            
            # Calculate returns with transaction costs
            returns = self._calculate_returns(test_data, predictions)
            
            # Store results
            split_metrics = self._calculate_split_metrics(returns)
            all_results.append(split_metrics)
        
        # Aggregate results
        return self._aggregate_results(all_results)
    
    def _apply_signal_weights(self, data: pd.DataFrame, weights: SignalWeights) -> np.ndarray:
        """Apply optimized weights to different signal categories"""
        weighted_features_list = []
        
        # Process each row individually
        for idx, row in data.iterrows():
            row_features = []
            
            # Handle technical features
            if 'technical_features' in data.columns:
                tech_feat = row['technical_features']
                if isinstance(tech_feat, (list, np.ndarray)):
                    tech_feat = np.array(tech_feat).flatten()
                    weighted_tech = tech_feat * weights.technical_weight
                    row_features.extend(weighted_tech)
                else:
                    row_features.append(float(tech_feat) * weights.technical_weight)
            
            # Handle onchain features
            if 'onchain_features' in data.columns:
                onchain_feat = row['onchain_features']
                if isinstance(onchain_feat, (list, np.ndarray)):
                    onchain_feat = np.array(onchain_feat).flatten()
                    weighted_onchain = onchain_feat * weights.onchain_weight
                    row_features.extend(weighted_onchain)
                else:
                    row_features.append(float(onchain_feat) * weights.onchain_weight)
            
            # Handle sentiment features
            if 'sentiment_features' in data.columns:
                sent_feat = row['sentiment_features']
                if isinstance(sent_feat, (list, np.ndarray)):
                    sent_feat = np.array(sent_feat).flatten()
                    weighted_sent = sent_feat * weights.sentiment_weight
                    row_features.extend(weighted_sent)
                else:
                    row_features.append(float(sent_feat) * weights.sentiment_weight)
            
            # Handle macro features
            if 'macro_features' in data.columns:
                macro_feat = row['macro_features']
                if isinstance(macro_feat, (list, np.ndarray)):
                    macro_feat = np.array(macro_feat).flatten()
                    weighted_macro = macro_feat * weights.macro_weight
                    row_features.extend(weighted_macro)
                else:
                    row_features.append(float(macro_feat) * weights.macro_weight)
            
            weighted_features_list.append(row_features)
        
        # Convert to numpy array
        weighted_features = np.array(weighted_features_list)
        
        # Ensure we return a 2D array
        if weighted_features.ndim == 1:
            weighted_features = weighted_features.reshape(-1, 1)
        
        return weighted_features
    
    def _calculate_returns(self, data: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Calculate returns including transaction costs"""
        # Use config values or defaults
        buy_threshold = self.config.buy_threshold if self.config.buy_threshold is not None else 0.02
        sell_threshold = self.config.sell_threshold if self.config.sell_threshold is not None else 0.02
        
        # Convert predictions to positions (-1, 0, 1)
        positions = np.where(predictions > buy_threshold, 1, 
                           np.where(predictions < -sell_threshold, -1, 0))
        
        # Debug logging
        logger.info(f"Thresholds - buy: {buy_threshold}, sell: {sell_threshold}")
        logger.info(f"Positions - long: {np.sum(positions == 1)}, short: {np.sum(positions == -1)}, neutral: {np.sum(positions == 0)}")
        
        # Calculate raw returns - use target column if available for proper alignment
        if 'target' in data.columns:
            # Target already contains next period returns
            price_returns = data['target'].fillna(0).values
            # Align positions with returns (predictions[i] → returns[i])
            strategy_returns = positions[:len(price_returns)] * price_returns
        elif 'Close' in data.columns:
            # Calculate next period returns manually
            price_returns = data['Close'].pct_change().shift(-1).fillna(0).values
            # Align positions with returns
            strategy_returns = positions[:len(price_returns)] * price_returns
        else:
            logger.warning("No price data available for return calculation")
            return np.zeros(len(predictions) - 1)
        
        # Apply transaction costs on position changes
        if len(positions) > 1:
            position_changes = np.diff(positions[:len(strategy_returns) + 1])
            transaction_costs = np.abs(position_changes[:len(strategy_returns)]) * self.config.transaction_cost
            net_returns = strategy_returns - transaction_costs
        else:
            net_returns = strategy_returns
        
        return net_returns
    
    def _calculate_split_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate all performance metrics for a split"""
        cumulative_returns = (1 + returns).cumprod()
        
        metrics = {
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(
                returns, 
                PerformanceMetrics.maximum_drawdown(cumulative_returns)
            ),
            'max_drawdown': PerformanceMetrics.maximum_drawdown(cumulative_returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'total_return': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'avg_daily_return': np.mean(returns),
            'volatility': np.std(returns) * np.sqrt(252)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return np.mean(excess_returns) * 252 / (np.std(returns) * np.sqrt(252))
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all splits"""
        if not results or len(results) == 0:
            logger.warning("No results to aggregate - insufficient data")
            return {
                'sortino_ratio_mean': 0.0,
                'calmar_ratio_mean': 0.0,
                'max_drawdown_mean': 0.0,
                'profit_factor_mean': 0.0,
                'win_rate_mean': 0.0,
                'total_return_mean': 0.0,
                'sharpe_ratio_mean': 0.0,
                'volatility_mean': 0.0,
                'composite_score': 0.0,
                'periods_tested': 0,
                'success': False,
                'error_message': 'Insufficient data for backtesting'
            }
            
        aggregated = {}
        
        for metric in results[0].keys():
            values = [r[metric] for r in results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        # Calculate composite score as per research
        aggregated['composite_score'] = self._calculate_composite_score(aggregated)
        
        return aggregated
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate composite score based on research recommendations"""
        # Risk Component (40%)
        risk_score = (
            min(metrics['sortino_ratio_mean'] / self.config.target_sortino_ratio, 2.0) * 0.20 +
            max(0, 1 - metrics['max_drawdown_mean'] / self.config.max_drawdown_threshold) * 0.20
        )
        
        # Return Component (35%)
        return_score = (
            min(metrics['calmar_ratio_mean'] / 3.0, 1.0) * 0.20 +
            min(metrics['profit_factor_mean'] / 2.0, 1.0) * 0.15
        )
        
        # Consistency Component (25%)
        consistency_score = (
            metrics['win_rate_mean'] * 0.15 +
            min(max(metrics.get('information_ratio_mean', 0) / 1.0, 0), 1.0) * 0.10
        )
        
        return risk_score + return_score + consistency_score


class EnhancedWalkForwardBacktester(WalkForwardBacktester):
    """Enhanced backtester with comprehensive signal integration"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.signal_calculator = ComprehensiveSignalCalculator()
        self.enhanced_metrics = EnhancedPerformanceMetrics()
        self.signal_performance = {}  # Track individual signal performance
        
    def backtest_strategy(self, model, data: pd.DataFrame, signal_weights: SignalWeights) -> Dict:
        """Enhanced backtest with all 50+ signals"""
        # Calculate all signals
        logger.info("Calculating comprehensive signal suite...")
        
        # Check if data has required columns for signal calculation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for signal calculation: {missing_cols}")
            logger.info(f"Available columns: {list(data.columns)[:20]}...")
            # If essential price data is missing, skip signal calculation
            if 'Close' not in data.columns:
                logger.error("Cannot calculate signals without Close price data")
                return self._aggregate_results([])
            data_with_signals = data.copy()
        else:
            data_with_signals = self.signal_calculator.calculate_all_signals(data)
        
        # Run original backtest logic
        splits = self.create_walk_forward_splits(data_with_signals)
        
        if len(splits) == 0:
            logger.warning("No valid splits created - using single period backtest")
            split_point = int(len(data_with_signals) * 0.8)
            train_data = data_with_signals.iloc[:split_point]
            test_data = data_with_signals.iloc[split_point:]
            if len(test_data) > 0:
                splits = [(train_data, test_data)]
            else:
                logger.error("Insufficient data for backtesting")
                return {
                    'sortino_ratio_mean': 0.0,
                    'calmar_ratio_mean': 0.0,
                    'max_drawdown_mean': 0.0,
                    'profit_factor_mean': 0.0,
                    'win_rate_mean': 0.0,
                    'total_return_mean': 0.0,
                    'sharpe_ratio_mean': 0.0,
                    'volatility_mean': 0.0,
                    'composite_score': 0.0,
                    'periods_tested': 0,
                    'success': False,
                    'error_message': 'Insufficient data for backtesting'
                }
        
        all_results = []
        signal_contributions = {}
        
        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}: train={len(train_data)}, test={len(test_data)}")
            
            try:
                # Apply enhanced signal weights
                weighted_features = self._apply_enhanced_signal_weights(train_data, signal_weights)
                
                # Prepare target
                if 'target' in train_data.columns:
                    target = train_data['target'].values.ravel()
                else:
                    if 'Close' in train_data.columns:
                        target = train_data['Close'].pct_change().shift(-1).fillna(0).values
                    else:
                        target = np.zeros(len(train_data))
                
                # Train model
                model.fit(weighted_features, target)
                
                # Generate predictions
                test_features = self._apply_enhanced_signal_weights(test_data, signal_weights)
                predictions = model.predict(test_features)
                
                # Log prediction statistics
                logger.info(f"Predictions - min: {np.min(predictions):.4f}, max: {np.max(predictions):.4f}, mean: {np.mean(predictions):.4f}, std: {np.std(predictions):.4f}")
                logger.info(f"Non-zero predictions: {np.sum(predictions != 0)} out of {len(predictions)}")
                
                # Calculate returns with enhanced logic
                returns, positions = self._calculate_enhanced_returns(test_data, predictions)
                
                # Track signal contributions
                split_contributions = self._analyze_signal_contributions(test_data, predictions, returns)
                for signal, contrib in split_contributions.items():
                    if signal not in signal_contributions:
                        signal_contributions[signal] = []
                    signal_contributions[signal].append(contrib)
                
                # Calculate enhanced metrics
                split_metrics = self._calculate_enhanced_split_metrics(returns, positions, test_data)
                all_results.append(split_metrics)
                
            except Exception as e:
                logger.error(f"Error in split {i+1}: {e}")
                all_results.append({
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'profit_factor': 1.0,
                    'win_rate': 0.5,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'avg_daily_return': 0.0,
                    'volatility': 0.2
                })
        
        # Aggregate results with signal analysis
        aggregated_results = self._aggregate_enhanced_results(all_results, signal_contributions)
        
        return aggregated_results
    
    def _apply_enhanced_signal_weights(self, data: pd.DataFrame, weights: SignalWeights) -> np.ndarray:
        """Apply weights to comprehensive signal set"""
        if isinstance(weights, EnhancedSignalWeights):
            return self._apply_granular_weights(data, weights)
        else:
            # Fall back to original weight application
            return self._apply_signal_weights(data, weights)
    
    def _apply_granular_weights(self, data: pd.DataFrame, weights: EnhancedSignalWeights) -> np.ndarray:
        """Apply granular sub-category weights"""
        # First, extract all features into a consistent structure
        weighted_features = []
        
        # Define feature groups with fixed order
        tech_features = {
            'momentum': ['rsi', 'roc', 'stoch_k', 'mfi'],
            'trend': ['macd', 'adx', 'aroon_bullish', 'sar_trend'],
            'volatility': ['bb_position', 'atr_normalized', 'cci'],
            'volume': ['obv_trend', 'volume_spike', 'cmf']
        }
        
        onchain_features = {
            'flow': ['net_exchange_flow', 'bullish_exchange_flow', 'stablecoin_bullish'],
            'network': ['active_addr_growth', 'tx_volume_growth', 'hash_rate_growth'],
            'holder': ['whale_accumulation', 'mvrv_low', 'sopr_bullish']
        }
        
        sentiment_features = {
            'social': ['twitter_bullish', 'reddit_spike', 'extreme_fear'],
            'derivatives': ['funding_extreme_negative', 'extreme_shorts', 'high_put_call'],
            'fear_greed': ['extreme_fear', 'extreme_greed', 'google_interest_high']
        }
        
        # Process each row
        for idx in range(len(data)):
            row_features = []
            
            # Technical signals
            for category, signals in tech_features.items():
                cat_weight = getattr(weights, f'{category}_weight', 0.25)
                cat_values = []
                for signal in signals:
                    col_name = f'tech_{signal}' if not signal.startswith('tech_') else signal
                    if col_name in data.columns:
                        val = data.iloc[idx].get(col_name, 0)
                        if pd.notna(val):
                            cat_values.append(float(val))
                
                weighted_val = np.mean(cat_values) * cat_weight * weights.technical_weight if cat_values else 0.0
                row_features.append(weighted_val)
            
            # On-chain signals with sub-weights
            if weights.onchain_weight > 0:
                for category, signals in onchain_features.items():
                    cat_weight = getattr(weights, f'{category}_weight', 0.33)
                    cat_values = []
                    for signal in signals:
                        if signal in data.columns:
                            val = data.iloc[idx].get(signal, 0)
                            if pd.notna(val):
                                cat_values.append(float(val))
                    
                    weighted_val = np.mean(cat_values) * cat_weight * weights.onchain_weight if cat_values else 0.0
                    row_features.append(weighted_val)
            else:
                # Add zeros for onchain features if weight is 0
                for _ in onchain_features:
                    row_features.append(0.0)
            
            # Sentiment signals with sub-weights
            if weights.sentiment_weight > 0:
                for category, signals in sentiment_features.items():
                    cat_weight = getattr(weights, f'{category}_weight', 0.33)
                    cat_values = []
                    for signal in signals:
                        if signal in data.columns:
                            val = data.iloc[idx].get(signal, 0)
                            if pd.notna(val):
                                cat_values.append(float(val))
                    
                    weighted_val = np.mean(cat_values) * cat_weight * weights.sentiment_weight if cat_values else 0.0
                    row_features.append(weighted_val)
            else:
                # Add zeros for sentiment features if weight is 0
                for _ in sentiment_features:
                    row_features.append(0.0)
            
            # Macro features (original)
            if weights.macro_weight > 0 and 'macro_features' in data.columns:
                macro_val = data.iloc[idx].get('macro_features', 0)
                if isinstance(macro_val, (list, np.ndarray)):
                    macro_val = np.mean(macro_val) if len(macro_val) > 0 else 0.0
                else:
                    macro_val = float(macro_val) if pd.notna(macro_val) else 0.0
                row_features.append(macro_val * weights.macro_weight)
            else:
                row_features.append(0.0)
            
            weighted_features.append(row_features)
        
        # Convert to numpy array and ensure 2D shape
        feature_matrix = np.array(weighted_features)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(-1, 1)
        
        return feature_matrix
    
    def _calculate_enhanced_returns(self, data: pd.DataFrame, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced return calculation with sophisticated position sizing"""
        # Use config values or defaults
        buy_threshold = self.config.buy_threshold if self.config.buy_threshold is not None else 0.02
        sell_threshold = self.config.sell_threshold if self.config.sell_threshold is not None else 0.02
        position_size = self.config.position_size if self.config.position_size is not None else 1.0
        
        # Get price returns - properly aligned with predictions
        if 'target' in data.columns:
            # Target already contains next period returns
            price_returns = data['target'].fillna(0).values
        elif 'Close' in data.columns:
            # Calculate next period returns
            price_returns = data['Close'].pct_change().shift(-1).fillna(0).values
        else:
            logger.warning("No price data available for return calculation")
            price_returns = np.zeros(len(data))
        
        # Enhanced position sizing based on signal strength and market conditions
        positions = np.zeros(len(predictions))
        
        for i in range(len(predictions)):
            base_position = 0
            
            # Base position from prediction
            if predictions[i] > buy_threshold:
                base_position = position_size
            elif predictions[i] < -sell_threshold:
                base_position = -position_size
            
            # Adjust position based on additional signals
            if i < len(data):
                # Volatility adjustment
                if 'atr_normalized' in data.columns:
                    vol_adjust = 1 / (1 + data.iloc[i].get('atr_normalized', 0.02))
                    base_position *= vol_adjust
                
                # Trend confirmation
                if 'adx_strong_trend' in data.columns and data.iloc[i].get('adx_strong_trend', False):
                    base_position *= 1.2
                
                # Sentiment adjustment
                if 'extreme_fear' in data.columns and data.iloc[i].get('extreme_fear', False):
                    base_position *= 0.8 if base_position > 0 else 1.2
                
                # Volume confirmation
                if 'volume_spike' in data.columns and data.iloc[i].get('volume_spike', False):
                    base_position *= 1.1
            
            positions[i] = np.clip(base_position, -1.5, 1.5)  # Allow some leverage
        
        # Calculate strategy returns - properly aligned
        min_len = min(len(positions), len(price_returns))
        strategy_returns = positions[:min_len] * price_returns[:min_len]
        
        # Apply transaction costs with consideration for position changes
        if len(positions) > 1:
            position_changes = np.diff(positions)
            transaction_costs = np.abs(position_changes) * self.config.transaction_cost
            # Pad transaction costs to match returns length
            tc_padded = np.zeros(len(strategy_returns))
            tc_padded[1:min(len(transaction_costs)+1, len(tc_padded))] = transaction_costs[:len(tc_padded)-1]
            net_returns = strategy_returns - tc_padded
        else:
            net_returns = strategy_returns
        
        return net_returns[:min_len], positions[:min_len]
    
    def _analyze_signal_contributions(self, data: pd.DataFrame, predictions: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Analyze which signals contributed most to returns"""
        contributions = {}
        
        # List of key signals to analyze
        key_signals = [
            'rsi_oversold', 'rsi_overbought', 'macd_bullish_cross', 'macd_bearish_cross',
            'golden_cross', 'death_cross', 'bb_position', 'volume_spike',
            'extreme_fear', 'extreme_greed', 'whale_accumulation', 'net_exchange_flow'
        ]
        
        for signal in key_signals:
            if signal in data.columns:
                # Calculate returns when signal is active
                # Ensure proper alignment between signal data and returns
                signal_data = data[signal].values
                min_len = min(len(signal_data), len(returns))
                signal_active = signal_data[:min_len]
                aligned_returns = returns[:min_len]
                
                # Get returns when signal is active
                if isinstance(signal_active[0], (bool, np.bool_)):
                    signal_returns = aligned_returns[signal_active]
                else:
                    # For numeric signals, consider them active when > 0
                    signal_returns = aligned_returns[signal_active > 0]
                
                if len(signal_returns) > 0:
                    contributions[signal] = {
                        'mean_return': np.mean(signal_returns),
                        'total_return': np.sum(signal_returns),
                        'count': len(signal_returns),
                        'win_rate': (signal_returns > 0).sum() / len(signal_returns)
                    }
        
        return contributions
    
    def _calculate_enhanced_split_metrics(self, returns: np.ndarray, positions: np.ndarray, data: pd.DataFrame) -> Dict:
        """Calculate enhanced metrics including portfolio-specific measures"""
        # Original metrics
        base_metrics = self._calculate_split_metrics(returns)
        
        # Additional portfolio metrics
        cumulative_returns = (1 + returns).cumprod()
        
        # Position statistics
        long_positions = (positions > 0).sum()
        short_positions = (positions < 0).sum()
        neutral_positions = (positions == 0).sum()
        
        # Risk metrics
        var_95 = self.enhanced_metrics.value_at_risk(returns, 0.95)
        cvar_95 = self.enhanced_metrics.conditional_value_at_risk(returns, 0.95)
        omega = self.enhanced_metrics.omega_ratio(returns)
        
        # Trading statistics
        trades = np.diff(positions) != 0
        num_trades = trades.sum()
        
        enhanced_metrics = {
            **base_metrics,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'omega_ratio': omega,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'neutral_positions': neutral_positions,
            'num_trades': num_trades,
            'avg_position_size': np.abs(positions).mean(),
            'position_turnover': np.abs(np.diff(positions)).sum() / len(positions)
        }
        
        return enhanced_metrics
    
    def _aggregate_enhanced_results(self, results: List[Dict], signal_contributions: Dict[str, List]) -> Dict:
        """Aggregate results with signal analysis"""
        # Original aggregation
        base_aggregated = self._aggregate_results(results)
        
        # Aggregate signal contributions
        aggregated_signals = {}
        for signal, contrib_list in signal_contributions.items():
            if contrib_list:
                aggregated_signals[signal] = {
                    'avg_mean_return': np.mean([c['mean_return'] for c in contrib_list]),
                    'total_contribution': sum([c['total_return'] for c in contrib_list]),
                    'total_count': sum([c['count'] for c in contrib_list]),
                    'avg_win_rate': np.mean([c['win_rate'] for c in contrib_list])
                }
        
        # Sort signals by contribution
        top_signals = sorted(
            aggregated_signals.items(),
            key=lambda x: x[1]['total_contribution'],
            reverse=True
        )[:10]
        
        enhanced_aggregated = {
            **base_aggregated,
            'top_contributing_signals': dict(top_signals),
            'signal_performance': aggregated_signals,
            'avg_var_95': np.mean([r.get('var_95', 0) for r in results]),
            'avg_cvar_95': np.mean([r.get('cvar_95', 0) for r in results]),
            'avg_omega_ratio': np.mean([r.get('omega_ratio', 1) for r in results if r.get('omega_ratio', 1) != float('inf')]),
            'total_long_positions': sum([r.get('long_positions', 0) for r in results]),
            'total_short_positions': sum([r.get('short_positions', 0) for r in results]),
            'total_num_trades': sum([r.get('num_trades', 0) for r in results]),
            'avg_position_turnover': np.mean([r.get('position_turnover', 0) for r in results])
        }
        
        return enhanced_aggregated

# ========== ENHANCED BAYESIAN OPTIMIZER ==========
class BayesianOptimizer:
    """Implements Bayesian optimization for hyperparameter and signal weight tuning"""
    
    def __init__(self, backtester: WalkForwardBacktester):
        self.backtester = backtester
        self.study = None
        
    def optimize_signal_weights(self, model, data: pd.DataFrame, n_trials: int = 100) -> SignalWeights:
        """Optimize signal weights using Optuna"""
        
        def objective(trial):
            # Suggest signal weights
            weights = SignalWeights(
                technical_weight=trial.suggest_float('technical_weight', 0.1, 0.6),
                onchain_weight=trial.suggest_float('onchain_weight', 0.1, 0.5),
                sentiment_weight=trial.suggest_float('sentiment_weight', 0.05, 0.3),
                macro_weight=trial.suggest_float('macro_weight', 0.05, 0.3)
            )
            weights.normalize()
            
            # Run backtest with these weights
            results = self.backtester.backtest_strategy(model, data, weights)
            
            # Return negative composite score (Optuna minimizes)
            return -results['composite_score']
        
        # Create and run study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best weights
        best_params = self.study.best_params
        best_weights = SignalWeights(
            technical_weight=best_params['technical_weight'],
            onchain_weight=best_params['onchain_weight'],
            sentiment_weight=best_params['sentiment_weight'],
            macro_weight=best_params['macro_weight']
        )
        best_weights.normalize()
        
        logger.info(f"Best signal weights found: {best_weights}")
        return best_weights
    
    def optimize_lstm_architecture(self, data: pd.DataFrame, n_trials: int = 50) -> Dict:
        """Optimize LSTM architecture hyperparameters"""
        
        def objective(trial):
            # Suggest hyperparameters based on research
            params = {
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'hidden_size': trial.suggest_categorical('hidden_size', [50, 128, 256, 512]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 192])
            }
            
            # Create and train model with these parameters
            model = self._create_lstm_model(params)
            
            # Simplified training and evaluation
            train_loss = self._train_model(model, data, params)
            
            return train_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def _create_lstm_model(self, params: Dict) -> nn.Module:
        """Create LSTM model with given parameters"""
        # This should be integrated with your existing LSTM model
        # Placeholder for model creation
        pass
    
    def _train_model(self, model, data, params) -> float:
        """Train model and return validation loss"""
        # Placeholder for training logic
        pass
        
class EnhancedBayesianOptimizer(BayesianOptimizer):
    """Enhanced optimizer for both main and sub-category weights"""
    
    def __init__(self, backtester: WalkForwardBacktester):
        super().__init__(backtester)
        
    def optimize_enhanced_signal_weights(self, model, data: pd.DataFrame, n_trials: int = 50) -> EnhancedSignalWeights:
        """Optimize both main category and sub-category weights"""
        logger.info(f"Starting enhanced Bayesian optimization with {n_trials} trials")
        
        def objective(trial):
            # Main category weights
            weights = EnhancedSignalWeights(
                technical_weight=trial.suggest_float('technical_weight', 0.2, 0.5),
                onchain_weight=trial.suggest_float('onchain_weight', 0.2, 0.4),
                sentiment_weight=trial.suggest_float('sentiment_weight', 0.1, 0.3),
                macro_weight=trial.suggest_float('macro_weight', 0.05, 0.2),
                
                # Technical sub-weights
                momentum_weight=trial.suggest_float('momentum_weight', 0.15, 0.35),
                trend_weight=trial.suggest_float('trend_weight', 0.25, 0.45),
                volatility_weight=trial.suggest_float('volatility_weight', 0.1, 0.25),
                volume_weight=trial.suggest_float('volume_weight', 0.1, 0.25),
                
                # On-chain sub-weights
                flow_weight=trial.suggest_float('flow_weight', 0.25, 0.5),
                network_weight=trial.suggest_float('network_weight', 0.2, 0.4),
                holder_weight=trial.suggest_float('holder_weight', 0.2, 0.4),
                
                # Sentiment sub-weights
                social_weight=trial.suggest_float('social_weight', 0.3, 0.6),
                derivatives_weight=trial.suggest_float('derivatives_weight', 0.2, 0.4),
                fear_greed_weight=trial.suggest_float('fear_greed_weight', 0.1, 0.3)
            )
            
            # Normalize weights
            weights.normalize()
            weights.normalize_subcategories()
            
            # Run backtest
            try:
                results = self.backtester.backtest_strategy(model, data, weights)
                
                # Multi-objective optimization: maximize composite score while minimizing risk
                composite_score = results.get('composite_score', 0)
                max_drawdown = abs(results.get('max_drawdown_mean', -1))
                
                # Weighted objective (80% return, 20% risk)
                objective_value = 0.8 * composite_score - 0.2 * max_drawdown
                
                return -objective_value  # Negative for minimization
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return 0
        
        # Create and run study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best weights
        best_params = self.study.best_params
        best_weights = EnhancedSignalWeights(**best_params)
        best_weights.normalize()
        best_weights.normalize_subcategories()
        
        logger.info(f"Best enhanced signal weights found with objective value: {-self.study.best_value:.4f}")
        return best_weights

# ========== ORIGINAL REMAINING CLASSES (PRESERVED) ==========

class ConceptDriftDetector:
    """Detect concept drift and trigger retraining"""
    
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_stats = None
        
    def detect_drift(self, recent_errors: np.ndarray) -> bool:
        """Detect if concept drift has occurred using ADWIN-inspired approach"""
        if len(recent_errors) < self.window_size:
            return False
            
        if self.baseline_stats is None:
            self.baseline_stats = {
                'mean': np.mean(recent_errors[:self.window_size]),
                'std': np.std(recent_errors[:self.window_size])
            }
            return False
        
        # Calculate recent statistics
        recent_mean = np.mean(recent_errors[-self.window_size:])
        
        # Check if drift occurred (simplified Page-Hinkley test)
        z_score = abs(recent_mean - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
        
        if z_score > self.threshold:
            logger.warning(f"Concept drift detected! Z-score: {z_score:.2f}")
            # Update baseline
            self.baseline_stats = {
                'mean': recent_mean,
                'std': np.std(recent_errors[-self.window_size:])
            }
            return True
            
        return False

class AdaptiveRetrainingScheduler:
    """Manages adaptive retraining based on market conditions"""
    
    def __init__(self, base_frequency_days: int = 90):
        self.base_frequency = base_frequency_days
        self.last_retrain_date = datetime.now()
        self.volatility_threshold = 0.5  # 50% annualized volatility
        self.drift_detector = ConceptDriftDetector()
        
    def should_retrain(self, recent_data: pd.DataFrame, recent_errors: np.ndarray) -> bool:
        """Determine if model should be retrained"""
        
        # Check base frequency
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        if days_since_retrain >= self.base_frequency:
            logger.info("Retraining due to scheduled frequency")
            return True
        
        # Check volatility-based retraining
        if self._check_high_volatility(recent_data):
            logger.info("Retraining due to high volatility")
            return True
        
        # Check concept drift
        if self.drift_detector.detect_drift(recent_errors):
            logger.info("Retraining due to concept drift")
            return True
        
        return False
    
    def _check_high_volatility(self, data: pd.DataFrame) -> bool:
        """Check if recent volatility exceeds threshold"""
        if 'Close' not in data.columns or len(data) < 30:
            return False
            
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 30:
            return False
            
        recent_volatility = returns.tail(30).std() * np.sqrt(252)
        return recent_volatility > self.volatility_threshold
    
    def mark_retrained(self):
        """Mark that retraining has occurred"""
        self.last_retrain_date = datetime.now()

# ========== ENHANCED BACKTESTING PIPELINE ==========

# ========== ADD THIS BEFORE LINE 1549 (before EnhancedBacktestingPipeline) ==========

class BacktestingPipeline:
    """Main pipeline for backtesting and optimization"""
    
    def __init__(self, model, config: BacktestConfig = None):
        self.model = model
        self.config = config or BacktestConfig()
        self.backtester = WalkForwardBacktester(self.config)
        self.optimizer = BayesianOptimizer(self.backtester)
        self.scheduler = AdaptiveRetrainingScheduler()
        
    def run_full_backtest(self, data: pd.DataFrame, optimize_weights: bool = True) -> Dict:
        """Run complete backtesting pipeline with optimization"""
        logger.info("Starting full backtesting pipeline")
        
        # Step 1: Optimize signal weights if requested
        if optimize_weights:
            logger.info("Optimizing signal weights...")
            optimal_weights = self.optimizer.optimize_signal_weights(self.model, data)
        else:
            logger.info("Using default signal weights")
            optimal_weights = SignalWeights()
            optimal_weights.normalize()
        
        # Step 2: Run final backtest with optimal weights
        logger.info("Running final backtest with optimal weights...")
        final_results = self.backtester.backtest_strategy(self.model, data, optimal_weights)
        
        # Step 3: Generate report
        report = self._generate_report(final_results, optimal_weights)
        
        return report
    
    def _generate_report(self, results: Dict, weights: SignalWeights) -> Dict:
        """Generate comprehensive backtest report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': results,
            'optimal_weights': {
                'technical': weights.technical_weight,
                'onchain': weights.onchain_weight,
                'sentiment': weights.sentiment_weight,
                'macro': weights.macro_weight
            },
            'risk_assessment': self._assess_risk(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _assess_risk(self, results: Dict) -> Dict:
        """Assess risk levels based on metrics"""
        risk_assessment = {
            'overall_risk': 'Low',
            'max_drawdown_risk': 'Acceptable' if results.get('max_drawdown_mean', 0) < self.config.max_drawdown_threshold else 'High',
            'volatility_risk': 'Normal' if results.get('volatility_mean', 0) < 1.0 else 'High',
            'concentration_risk': 'Balanced'  # Based on signal weights
        }
        
        # Handle potential missing keys with .get()
        max_dd = results.get('max_drawdown_mean', 0)
        sortino = results.get('sortino_ratio_mean', 2.0)
        
        if max_dd > 0.4 or sortino < 1.0:
            risk_assessment['overall_risk'] = 'High'
        elif max_dd > 0.25 or sortino < 1.5:
            risk_assessment['overall_risk'] = 'Medium'
            
        return risk_assessment
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if results.get('sortino_ratio_mean', 0) < self.config.target_sortino_ratio:
            recommendations.append("Consider reducing position sizes to improve risk-adjusted returns")
            
        if results.get('max_drawdown_mean', 0) > self.config.max_drawdown_threshold:
            recommendations.append("Implement tighter stop-loss rules to reduce maximum drawdown")
            
        if results.get('win_rate_mean', 0) < 0.45:
            recommendations.append("Review entry signals - win rate is below optimal threshold")
            
        if results.get('volatility_mean', 0) > 1.0:
            recommendations.append("High volatility detected - consider volatility-based position sizing")
            
        return recommendations

# ========== NOW EnhancedBacktestingPipeline CAN INHERIT FROM IT ==========


class EnhancedBacktestingPipeline(BacktestingPipeline):
    """Enhanced pipeline with comprehensive signal support"""
    
    def __init__(self, model, config: BacktestConfig = None):
        self.model = model
        self.config = config or BacktestConfig()
        self.backtester = EnhancedWalkForwardBacktester(self.config)
        self.optimizer = EnhancedBayesianOptimizer(self.backtester)
        self.scheduler = AdaptiveRetrainingScheduler()
        
    def run_full_backtest(self, data: pd.DataFrame, optimize_weights: bool = True, use_enhanced_weights: bool = True) -> Dict:
        """Run complete backtesting pipeline with enhanced signal optimization"""
        logger.info("Starting enhanced backtesting pipeline")
        
        try:
            # Step 1: Optimize signal weights if requested
            if optimize_weights and len(data) >= 200:
                logger.info("Optimizing signal weights...")
                if use_enhanced_weights:
                    optimal_weights = self.optimizer.optimize_enhanced_signal_weights(self.model, data)
                else:
                    optimal_weights = self.optimizer.optimize_signal_weights(self.model, data)
            else:
                logger.info("Using default signal weights")
                optimal_weights = EnhancedSignalWeights() if use_enhanced_weights else SignalWeights()
            
            # Step 2: Run final backtest with optimal weights
            logger.info("Running final backtest with optimal weights...")
            final_results = self.backtester.backtest_strategy(self.model, data, optimal_weights)
            
            # Step 3: Generate enhanced report
            report = self._generate_enhanced_report(final_results, optimal_weights)
            
            return report
            
        except Exception as e:
            logger.error(f"Backtesting pipeline failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': {
                    'sortino_ratio_mean': 0.0,
                    'calmar_ratio_mean': 0.0,
                    'max_drawdown_mean': 0.0,
                    'profit_factor_mean': 0.0,
                    'win_rate_mean': 0.0,
                    'total_return_mean': 0.0,
                    'sharpe_ratio_mean': 0.0,
                    'volatility_mean': 0.0,
                    'composite_score': 0.0,
                    'periods_tested': 0,
                    'success': False,
                    'error_message': str(e)
                },
                'optimal_weights': self._get_default_weights_dict(optimal_weights),
                'risk_assessment': {'overall_risk': 'Unknown'},
                'recommendations': [f"Backtesting failed: {str(e)}"],
                'error': str(e)
            }
    
    def _generate_enhanced_report(self, results: Dict, weights: SignalWeights) -> Dict:
        """Generate comprehensive backtest report with signal analysis"""
        base_report = self._generate_report(results, weights)
        
        # Add enhanced metrics
        enhanced_report = {
            **base_report,
            'signal_analysis': {
                'top_signals': results.get('top_contributing_signals', {}),
                'signal_performance': results.get('signal_performance', {})
            },
            'risk_metrics': {
                'var_95': results.get('avg_var_95', 0),
                'cvar_95': results.get('avg_cvar_95', 0),
                'omega_ratio': results.get('avg_omega_ratio', 1)
            },
            'trading_statistics': {
                'total_trades': results.get('total_num_trades', 0),
                'long_positions': results.get('total_long_positions', 0),
                'short_positions': results.get('total_short_positions', 0),
                'avg_position_turnover': results.get('avg_position_turnover', 0)
            }
        }
        
        # Add signal-specific recommendations
        signal_recommendations = self._generate_signal_recommendations(results)
        enhanced_report['recommendations'].extend(signal_recommendations)
        
        return enhanced_report
    
    def _get_default_weights_dict(self, weights: SignalWeights) -> Dict:
        """Get weights as dictionary"""
        if isinstance(weights, EnhancedSignalWeights):
            return {
                'technical': weights.technical_weight,
                'onchain': weights.onchain_weight,
                'sentiment': weights.sentiment_weight,
                'macro': weights.macro_weight,
                'technical_sub': {
                    'momentum': weights.momentum_weight,
                    'trend': weights.trend_weight,
                    'volatility': weights.volatility_weight,
                    'volume': weights.volume_weight
                },
                'onchain_sub': {
                    'flow': weights.flow_weight,
                    'network': weights.network_weight,
                    'holder': weights.holder_weight
                },
                'sentiment_sub': {
                    'social': weights.social_weight,
                    'derivatives': weights.derivatives_weight,
                    'fear_greed': weights.fear_greed_weight
                }
            }
        else:
            return {
                'technical': weights.technical_weight,
                'onchain': weights.onchain_weight,
                'sentiment': weights.sentiment_weight,
                'macro': weights.macro_weight
            }
    
    def _generate_signal_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on signal performance"""
        recommendations = []
        
        # Analyze top contributing signals
        top_signals = results.get('top_contributing_signals', {})
        if top_signals:
            best_signal = list(top_signals.keys())[0] if top_signals else None
            if best_signal:
                recommendations.append(f"Focus on {best_signal} signal which showed highest contribution")
        
        # Check signal diversity
        signal_perf = results.get('signal_performance', {})
        if len(signal_perf) < 10:
            recommendations.append("Consider incorporating more diverse signals for robustness")
        
        # Risk-based recommendations
        if results.get('avg_omega_ratio', 1) < 1.5:
            recommendations.append("Omega ratio suggests adjusting threshold for better risk/reward")
        
        if results.get('avg_position_turnover', 0) > 0.5:
            recommendations.append("High position turnover - consider longer holding periods to reduce costs")
        
        return recommendations

# Integration example remains the same
if __name__ == "__main__":
    print("Enhanced Backtesting system with 50+ signals initialized successfully")