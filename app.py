# -*- coding: utf-8 -*-

# Standard library imports
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import os
import time

# Third-party imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.stats import norm, stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types"""
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def parse_symbol(symbol: str) -> Dict:
    """
    Parse different types of symbols (index, stock, or option)
    """
    try:
        # First try to parse as option
        option_info = parse_option_symbol(symbol)
        if option_info:
            return {
                'type': 'option',
                **option_info
            }

        # If not an option, check if it's an index
        KNOWN_INDICES = {'NIFTY 50', 'NIFTY BANK', 'NIFTY IT', 'SENSEX'}
        if symbol.upper() in KNOWN_INDICES:
            return {
                'type': 'index',
                'symbol': symbol.upper()
            }

        # Otherwise treat as stock
        return {
            'type': 'stock',
            'symbol': symbol.upper()
        }

    except Exception as e:
        logger.error(f"Error parsing symbol: {str(e)}")
        return None

def parse_option_symbol(symbol: str) -> Dict:
    """
    Parse option trading symbol to extract strike, expiry, and option type
    Example: NIFTY14JAN2523500PE -> {
        'strike': 23500,
        'expiry_date': '2025-01-14',
        'option_type': 'put',
        'days_to_expiry': <calculated>
    }
    """
    try:
        # Constants for parsing
        MONTHS = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }

        # Extract components using regex
        import re
        pattern = r'([A-Z]+)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)'
        match = re.match(pattern, symbol)

        if not match:
            raise ValueError(f"Invalid option symbol format: {symbol}")

        _, day, month, year, strike, option_type = match.groups()

        # Convert to proper date format
        year = '20' + year
        month = MONTHS[month]
        expiry_date = f"{year}-{month}-{day}"

        # Calculate days to expiry
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.now()
        days_to_expiry = (expiry - today).days

        return {
            'strike': float(strike),
            'expiry_date': expiry_date,
            'days_to_expiry': max(0, days_to_expiry),  # Ensure non-negative
            'option_type': 'put' if option_type == 'PE' else 'call'
        }
    except Exception as e:
        logger.error(f"Error parsing option symbol: {str(e)}")
        return None

def calculate_historical_volatility(prices: pd.Series, window: int = 20) -> float:
    """Calculate historical volatility from a series of prices"""
    try:
        returns = np.log(prices / prices.shift(1))
        rolling_std = returns.rolling(window=window).std()
        annualized_vol = rolling_std.iloc[-1] * np.sqrt(252)
        return float(annualized_vol)
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {str(e)}")
        return 0.3  # Return default 30% volatility if calculation fails


# Initialize Flask app
app = Flask(__name__)
CORS(app)

@dataclass
class MarketContext:
    """Class to hold market context"""
    index_price: float
    index_history: pd.DataFrame
    option_history: pd.DataFrame
    oi_history: pd.DataFrame

class TechnicalIndicators:
    """Enhanced Technical Indicators"""

    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI with enhanced accuracy"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.Series, fast_period=12, slow_period=26, signal_period=9) -> Dict:
        """Enhanced MACD calculation"""
        fast_ema = TechnicalIndicators.calculate_ema(data, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window=20, std_dev=2) -> Dict:
        """Enhanced Bollinger Bands calculation"""
        sma = TechnicalIndicators.calculate_sma(data, window)
        std = data.rolling(window=window).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bandwidth = (upper_band - lower_band) / sma

        return {
            'middle_band': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'bandwidth': bandwidth
        }

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window=14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()


"""
Options Analysis Server with Enhanced Analysis and Ngrok Integration - Part 2: Market Regime Detector
"""

class MarketRegimeDetector:
    """Enhanced market regime detection"""

    def __init__(self):
        self.tech_indicators = TechnicalIndicators()
        self.lookback_short = 20
        self.lookback_long = 50
        self.volatility_window = 20

    def detect_regime(self, context: MarketContext) -> Dict:
        """Detect market regime using multiple indicators"""
        try:
            # Calculate indicators for index
            index_indicators = self.calculate_indicators(context.index_history)

            # Calculate indicators for option
            option_indicators = self.calculate_indicators(context.option_history)

            # Detect trend
            trend = self.detect_trend(index_indicators)

            # Detect volatility
            volatility = self.detect_volatility(index_indicators, option_indicators)

            # Detect momentum
            momentum = self.detect_momentum(index_indicators, option_indicators)

            # Analyze OI patterns
            oi_analysis = self.analyze_oi_patterns(context.oi_history)

            return {
                'trend': trend,
                'volatility': volatility,
                'momentum': momentum,
                'oi_analysis': oi_analysis,
                'regime_classification': self.classify_regime(trend, volatility, momentum, oi_analysis)
            }

        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            result_df = df.copy()

            # Price-based indicators
            result_df['sma_short'] = TechnicalIndicators.calculate_sma(df['close'], self.lookback_short)
            result_df['sma_long'] = TechnicalIndicators.calculate_sma(df['close'], self.lookback_long)
            result_df['ema_short'] = TechnicalIndicators.calculate_ema(df['close'], self.lookback_short)
            result_df['ema_long'] = TechnicalIndicators.calculate_ema(df['close'], self.lookback_long)

            # Momentum indicators
            result_df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            macd_data = TechnicalIndicators.calculate_macd(df['close'])
            result_df['macd'] = macd_data['macd_line']
            result_df['macd_signal'] = macd_data['signal_line']
            result_df['macd_hist'] = macd_data['histogram']

            # Volatility indicators
            bb_data = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            result_df['bb_middle'] = bb_data['middle_band']
            result_df['bb_upper'] = bb_data['upper_band']
            result_df['bb_lower'] = bb_data['lower_band']
            result_df['bb_bandwidth'] = bb_data['bandwidth']

            result_df['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])

            # Volume and OI based calculations
            if 'volume' in df.columns:
                result_df['volume_sma'] = TechnicalIndicators.calculate_sma(df['volume'], self.lookback_short)
                result_df['volume_ratio'] = df['volume'] / result_df['volume_sma']

            return result_df

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def detect_trend(self, df: pd.DataFrame) -> Dict:
        """Enhanced trend detection"""
        try:
            latest = df.iloc[-1]
            trend_signals = {
                'ema_trend': latest['ema_short'] > latest['ema_long'],
                'price_above_sma': latest['close'] > latest['sma_long'],
                'macd_positive': latest['macd_hist'] > 0,
                'rsi_trend': latest['rsi'] > 50
            }

            bullish_signals = sum(trend_signals.values())
            trend_strength = bullish_signals / len(trend_signals)

            trend_type = 'STRONG_UPTREND' if trend_strength > 0.75 else \
                        'UPTREND' if trend_strength > 0.5 else \
                        'DOWNTREND' if trend_strength < 0.5 else \
                        'STRONG_DOWNTREND'

            return {
                'type': trend_type,
                'strength': trend_strength,
                'signals': trend_signals
            }

        except Exception as e:
            logger.error(f"Error detecting trend: {str(e)}")
            return None

    def detect_volatility(self, index_df: pd.DataFrame, option_df: pd.DataFrame) -> Dict:
        """Enhanced volatility detection"""
        try:
            # Calculate historical volatility
            index_returns = np.log(index_df['close'] / index_df['close'].shift(1))
            option_returns = np.log(option_df['close'] / option_df['close'].shift(1))

            index_vol = index_returns.std() * np.sqrt(252)  # Annualized
            option_vol = option_returns.std() * np.sqrt(252)

            # Get latest ATR values
            index_atr = index_df['atr'].iloc[-1] / index_df['close'].iloc[-1]  # Normalized ATR
            option_atr = option_df['atr'].iloc[-1] / option_df['close'].iloc[-1]

            # Bollinger Band analysis
            bb_width = index_df['bb_bandwidth'].iloc[-1]

            volatility_state = 'HIGH' if (option_vol > index_vol * 1.5 or bb_width > 0.1) else \
                              'LOW' if (option_vol < index_vol * 0.5 or bb_width < 0.02) else \
                              'NORMAL'

            return {
                'state': volatility_state,
                'metrics': {
                    'index_volatility': float(index_vol),
                    'option_volatility': float(option_vol),
                    'index_atr': float(index_atr),
                    'option_atr': float(option_atr),
                    'bb_width': float(bb_width)
                }
            }

        except Exception as e:
            logger.error(f"Error detecting volatility: {str(e)}")
            return None

    def detect_momentum(self, index_df: pd.DataFrame, option_df: pd.DataFrame) -> Dict:
        """Enhanced momentum detection"""
        try:
            latest_index = index_df.iloc[-1]
            latest_option = option_df.iloc[-1]

            # Momentum signals
            signals = {
                'index_rsi': 'BULLISH' if latest_index['rsi'] > 50 else 'BEARISH',
                'option_rsi': 'BULLISH' if latest_option['rsi'] > 50 else 'BEARISH',
                'index_macd': 'BULLISH' if latest_index['macd_hist'] > 0 else 'BEARISH',
                'option_macd': 'BULLISH' if latest_option['macd_hist'] > 0 else 'BEARISH'
            }

            # Calculate momentum score
            bullish_signals = sum(1 for signal in signals.values() if signal == 'BULLISH')
            momentum_score = bullish_signals / len(signals)

            return {
                'score': momentum_score,
                'signals': signals,
                'strength': 'STRONG' if abs(momentum_score - 0.5) > 0.3 else 'WEAK'
            }

        except Exception as e:
            logger.error(f"Error detecting momentum: {str(e)}")
            return None

    def analyze_oi_patterns(self, oi_df: pd.DataFrame) -> Dict:
        """Analyze Open Interest patterns"""
        try:
            if len(oi_df) < 2:
                return {'trend': 'INSUFFICIENT_DATA'}

            # Calculate OI changes
            oi_df['oi_change'] = oi_df['oi'].pct_change()
            oi_df['oi_ma'] = TechnicalIndicators.calculate_sma(oi_df['oi'], 5)

            latest = oi_df.iloc[-1]

            # Detect OI trend
            oi_trend = 'INCREASING' if latest['oi'] > latest['oi_ma'] else 'DECREASING'

            # Calculate buildup
            price_up = latest['close'] > oi_df['close'].iloc[-2] if 'close' in oi_df.columns else None
            oi_up = latest['oi'] > oi_df['oi'].iloc[-2]

            if price_up is not None:
                buildup = 'LONG_BUILDUP' if price_up and oi_up else \
                         'SHORT_BUILDUP' if not price_up and oi_up else \
                         'LONG_UNWINDING' if not price_up and not oi_up else \
                         'SHORT_COVERING'
            else:
                buildup = 'UNKNOWN'

            return {
                'trend': oi_trend,
                'buildup': buildup,
                'change_percentage': float(latest['oi_change'] * 100)
            }

        except Exception as e:
            logger.error(f"Error analyzing OI patterns: {str(e)}")
            return None

    def classify_regime(self, trend: Dict, volatility: Dict,
                       momentum: Dict, oi_analysis: Dict) -> Dict:
        """Classify market regime"""
        try:
            # Calculate regime score
            regime_score = 0

            # Trend component
            if trend['type'] == 'STRONG_UPTREND':
                regime_score += 2
            elif trend['type'] == 'UPTREND':
                regime_score += 1
            elif trend['type'] == 'DOWNTREND':
                regime_score -= 1
            elif trend['type'] == 'STRONG_DOWNTREND':
                regime_score -= 2

            # Volatility component
            if volatility['state'] == 'HIGH':
                regime_score *= 0.8  # Reduce confidence in high volatility
            elif volatility['state'] == 'LOW':
                regime_score *= 1.2  # Increase confidence in low volatility

            # Momentum component
            regime_score += (momentum['score'] - 0.5) * 2

            # OI confirmation
            if oi_analysis['trend'] == 'INCREASING':
                regime_score *= 1.2
            elif oi_analysis['trend'] == 'DECREASING':
                regime_score *= 0.8

            # Classify regime
            regime_type = 'STRONGLY_BULLISH' if regime_score > 2 else \
                         'BULLISH' if regime_score > 0.5 else \
                         'NEUTRAL' if abs(regime_score) <= 0.5 else \
                         'BEARISH' if regime_score < -0.5 else \
                         'STRONGLY_BEARISH'

            return {
                'type': regime_type,
                'score': float(regime_score),
                'confidence': float(min(abs(regime_score/3), 1.0))  # Normalize to 0-1
            }

        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            return None

class EnhancedOptionsAnalyzer:
    """Enhanced options analysis with market regime integration"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.tech_indicators = TechnicalIndicators()
        self.greeks_analyzer = OptionsGreeksAnalyzer()
        self.oi_analyzer = EnhancedOIAnalyzer()
        self.risk_reward_ratio = 1.5
        self.min_volume = 100

    def _process_data(self, data: List[List], is_index: bool = False) -> pd.DataFrame:
        """Process raw OHLCV data with special handling for index data"""
        try:
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            if is_index:
                df.drop('volume', axis=1, inplace=True)
            else:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                df['oi'] = df['volume'].cumsum()

            return df

        except Exception as e:
            logger.error(f"Error processing {'index' if is_index else 'option'} data: {str(e)}")
            raise

    def analyze_instrument(self,
                         index_data: List[List],
                         option_data: List[List],
                         symbol: str) -> Dict:
        """
        Analyze any financial instrument (index, stock, or option)
        """
        try:
            # Parse the symbol
            instrument_info = parse_symbol(symbol)
            if not instrument_info:
                raise ValueError(f"Could not parse symbol: {symbol}")

            # Process index/base data
            index_df = self._process_data(index_data, is_index=True)

            if instrument_info['type'] == 'option':
                # For options, process option data separately
                option_df = self._process_data(option_data, is_index=False)
            else:
                # For non-options, use the same data
                option_df = self._process_data(index_data, is_index=False)

            # Create market context
            context = MarketContext(
                index_price=float(index_df['close'].iloc[-1]),
                index_history=index_df,
                option_history=option_df,
                oi_history=pd.DataFrame()  # Will be updated for options
            )

            # Detect market regime
            regime = self.regime_detector.detect_regime(context)

            # Generate signals
            signals = self.generate_signals(context, regime)

            # Calculate risk parameters with instrument info
            risk_params = self.calculate_risk_parameters(
                context=context,
                signals=signals,
                instrument_info=instrument_info
            )

            # Initialize optional analyses
            greeks = None
            oi_analysis = None

            # Calculate Greeks and OI analysis only for options
            if instrument_info['type'] == 'option':
                # Calculate historical volatility
                hist_vol = calculate_historical_volatility(option_df['close'])

                # Create proper OI data structure
                if 'oi' in option_df.columns:
                    oi_data = pd.DataFrame({
                        'time': option_df.index,
                        'oi': option_df['oi'],
                        'close': option_df['close']
                    }).reset_index(drop=True)

                    context.oi_history = oi_data

                # Calculate Greeks
                greeks = self.greeks_analyzer.calculate_greeks(
                    S=context.index_price,
                    K=instrument_info['strike'],
                    T=instrument_info['days_to_expiry']/365,
                    r=self.greeks_analyzer.risk_free_rate,
                    sigma=hist_vol,
                    option_type=instrument_info['option_type']
                )

                # OI analysis if available
                if not context.oi_history.empty:
                    option_chain = pd.DataFrame([{
                        'strike': instrument_info['strike'],
                        'days_to_expiry': instrument_info['days_to_expiry'],
                        'implied_vol': hist_vol,
                        'option_type': instrument_info['option_type'],
                        'open_interest': option_df['oi'].iloc[-1]
                    }])

                    oi_analysis = self.oi_analyzer.analyze_oi_patterns(
                        option_chain=option_chain,
                        historical_oi=context.oi_history
                    )

            # Generate recommendations
            recommendations = self.generate_recommendations(
                signals=signals,
                risk_params=risk_params
            )

            return {
                'status': 'success',
                'analysis': {
                    'instrument_info': instrument_info,
                    'regime': regime,
                    'signals': signals,
                    'risk_parameters': risk_params,
                    'greeks': greeks,
                    'oi_analysis': oi_analysis,
                    'recommendations': recommendations
                }
            }

        except Exception as e:
            logger.error(f"Error in instrument analysis: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    def _process_oi_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process OI data"""
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # Add closing price from option data if available
        if 'close' not in df.columns:
            df['close'] = None

        return df

    def generate_signals(self, context: MarketContext, regime: Dict) -> Dict:
        """Generate trading signals with optional OI data"""
        try:
            # Price action signals (always available)
            price_signals = self._analyze_price_action(context)

            # Volume analysis (always available)
            volume_signals = self._analyze_volume(context)

            # OI analysis (optional)
            oi_signals = None
            if not context.oi_history.empty:
                oi_signals = self._analyze_oi(context)
            else:
                oi_signals = {
                    'trend': 'NO_DATA',
                    'buildup': 'UNKNOWN',
                    'current_oi': 0,
                    'oi_change': 0
                }

            # Calculate score with available data
            combined_score = self._calculate_signal_score(
                price_signals,
                volume_signals,
                oi_signals,
                regime
            )

            return {
                'price_action': price_signals,
                'volume_analysis': volume_signals,
                'oi_analysis': oi_signals,
                'combined_score': combined_score,
                'primary_signal': self._determine_primary_signal(combined_score)
            }

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return None

    def _analyze_price_action(self, context: MarketContext) -> Dict:
        """Analyze price action patterns"""
        try:
            option_df = context.option_history

            # Calculate indicators
            indicators = self.tech_indicators
            rsi = indicators.calculate_rsi(option_df['close'])
            macd = indicators.calculate_macd(option_df['close'])
            bb = indicators.calculate_bollinger_bands(option_df['close'])

            # Get latest values
            latest = option_df.iloc[-1]
            latest_rsi = rsi.iloc[-1]
            latest_macd = macd['histogram'].iloc[-1]

            # Check Bollinger Band position
            price = latest['close']
            bb_position = 'MIDDLE'
            if price > bb['upper_band'].iloc[-1]:
                bb_position = 'ABOVE'
            elif price < bb['lower_band'].iloc[-1]:
                bb_position = 'BELOW'

            # Generate signals
            signals = {
                'rsi': {
                    'value': float(latest_rsi),
                    'signal': 'OVERSOLD' if latest_rsi < 30 else 'OVERBOUGHT' if latest_rsi > 70 else 'NEUTRAL'
                },
                'macd': {
                    'value': float(latest_macd),
                    'signal': 'BULLISH' if latest_macd > 0 else 'BEARISH'
                },
                'bollinger': {
                    'position': bb_position,
                    'bandwidth': float(bb['bandwidth'].iloc[-1])
                }
            }

            return signals

        except Exception as e:
            logger.error(f"Error in price action analysis: {str(e)}")
            return None

    def _analyze_volume(self, context: MarketContext) -> Dict:
        """Analyze volume patterns"""
        try:
            df = context.option_history

            # Calculate volume metrics
            volume_sma = self.tech_indicators.calculate_sma(df['volume'], 20)
            volume_ratio = df['volume'] / volume_sma

            # Get latest values
            latest_volume = df['volume'].iloc[-1]
            latest_ratio = volume_ratio.iloc[-1]

            # Volume trend
            volume_trend = 'HIGH' if latest_ratio > 1.5 else \
                          'LOW' if latest_ratio < 0.5 else \
                          'NORMAL'

            return {
                'current_volume': float(latest_volume),
                'volume_ratio': float(latest_ratio),
                'trend': volume_trend,
                'is_valid': latest_volume >= self.min_volume
            }

        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return None

    def _analyze_oi(self, context: MarketContext) -> Dict:
        """Analyze Open Interest patterns"""
        try:
            df = context.oi_history

            if len(df) < 2:
                return {'trend': 'INSUFFICIENT_DATA'}

            # Calculate OI changes
            latest_oi = df['oi'].iloc[-1]
            prev_oi = df['oi'].iloc[-2]
            oi_change = (latest_oi - prev_oi) / prev_oi * 100

            # Price change if available
            price_change = None
            if 'close' in df.columns and not df['close'].isnull().all():
                latest_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                price_change = (latest_price - prev_price) / prev_price * 100

            # Determine buildup
            if price_change is not None:
                buildup = 'LONG_BUILDUP' if price_change > 0 and oi_change > 0 else \
                         'SHORT_BUILDUP' if price_change < 0 and oi_change > 0 else \
                         'LONG_UNWINDING' if not price_change and not oi_change else \
                         'SHORT_COVERING'
            else:
                buildup = 'UNKNOWN'

            return {
                'current_oi': float(latest_oi),
                'oi_change': float(oi_change),
                'buildup': buildup,
                'trend': 'INCREASING' if oi_change > 5 else 'DECREASING' if oi_change < -5 else 'STABLE'
            }

        except Exception as e:
            logger.error(f"Error in OI analysis: {str(e)}")
            return None

    def calculate_risk_parameters(self, context: MarketContext, signals: Dict, instrument_info: Dict) -> Dict:
        """Calculate risk parameters based on instrument type and signal direction"""
        try:
            # Get regime data safely
            regime = self.regime_detector.detect_regime(context)
            volatility_state = regime.get('volatility', {}).get('state', 'NORMAL') if regime else 'NORMAL'

            # Get current price and signal
            price_data = context.option_history if instrument_info['type'] == 'option' else context.index_history
            current_price = round(float(price_data['close'].iloc[-1]), 2)
            combined_score = signals.get('combined_score', 0) if signals else 0
            is_sell_signal = combined_score < 0

            # Set stop-loss percentage based on volatility
            if volatility_state == 'HIGH':
                sl_percentage = 0.07  # 7% for high volatility
            elif volatility_state == 'LOW':
                sl_percentage = 0.03  # 3% for low volatility
            else:
                sl_percentage = 0.05  # 5% for normal volatility

            # For options, handle sell signals differently
            if instrument_info['type'] == 'option':
                if is_sell_signal:
                    return {
                        'entry_price': current_price,
                        'action': 'DO_NOT_BUY',
                        'reason': 'Bearish signal detected - avoid buying call option',
                        'price_source': 'option',
                        'volatility_state': volatility_state
                    }
                else:
                    # For buy signals on options
                    stop_loss = round(current_price * (1 - sl_percentage), 2)
                    target = round(current_price * (1 + (sl_percentage * self.risk_reward_ratio)), 2)
                    risk_per_trade = round(abs(current_price - stop_loss), 2)
                    potential_reward = round(abs(target - current_price), 2)

                    return {
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'action': 'BUY',
                        'risk_reward_ratio': self.risk_reward_ratio,
                        'risk_per_trade': risk_per_trade,
                        'potential_reward': potential_reward,
                        'price_source': 'option',
                        'volatility_state': volatility_state
                    }
            else:
                # For stocks/indices - keep existing logic
                if is_sell_signal:
                    stop_loss = round(current_price * (1 + sl_percentage), 2)
                    target = round(current_price * (1 - (sl_percentage * self.risk_reward_ratio)), 2)
                else:
                    stop_loss = round(current_price * (1 - sl_percentage), 2)
                    target = round(current_price * (1 + (sl_percentage * self.risk_reward_ratio)), 2)

                risk_per_trade = round(abs(current_price - stop_loss), 2)
                potential_reward = round(abs(target - current_price), 2)

                return {
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'action': 'SELL' if is_sell_signal else 'BUY',
                    'risk_reward_ratio': self.risk_reward_ratio,
                    'risk_per_trade': risk_per_trade,
                    'potential_reward': potential_reward,
                    'price_source': 'index/stock',
                    'volatility_state': volatility_state
                }

        except Exception as e:
            logger.error(f"Error calculating risk parameters: {str(e)}")
            return {
                'entry_price': current_price,
                'action': 'ERROR',
                'error': str(e),
                'price_source': 'unknown'
            }
    def _calculate_signal_score(self, price_signals: Dict,
                          volume_signals: Dict,
                          oi_signals: Dict,
                          regime: Dict) -> float:
        """Calculate signal score with enhanced reversal detection"""
        try:
            # Extract key metrics
            rsi_value = price_signals['rsi']['value']
            macd_value = price_signals['macd']['value']
            trend_type = regime.get('trend', {}).get('type', 'NEUTRAL')
            volatility_state = regime.get('volatility', {}).get('state', 'NORMAL')
            vol_metrics = regime.get('volatility', {}).get('metrics', {})

            # Calculate volatility ratio
            option_vol = vol_metrics.get('option_volatility', 0)
            index_vol = vol_metrics.get('index_volatility', 1)
            vol_ratio = option_vol / max(index_vol, 0.001)

            score = 0.0
            conditions = []

            # 1. Trend Score (30% weight)
            trend_score = 0
            if trend_type in ['STRONG_DOWNTREND', 'DOWNTREND']:
                trend_score = -1.0
                # Check for potential reversal
                if rsi_value < 40 and macd_value > 1.5:  # Strong MACD in oversold
                    trend_score *= 0.3  # Significantly reduce bearish weight
                    conditions.append('POTENTIAL_REVERSAL')
            elif trend_type in ['STRONG_UPTREND', 'UPTREND']:
                trend_score = 1.0
                # Check for potential top
                if rsi_value > 60 and macd_value < -1.5:
                    trend_score *= 0.3
                    conditions.append('POTENTIAL_TOP')

            score += trend_score * 0.3

            # 2. Momentum Score (40% weight)
            momentum_score = 0

            # RSI Component
            if rsi_value < 30:
                momentum_score += 1.0
                conditions.append('STRONG_OVERSOLD')
            elif rsi_value < 40:
                momentum_score += 0.5
                conditions.append('NEAR_OVERSOLD')
            elif rsi_value > 70:
                momentum_score -= 1.0
                conditions.append('STRONG_OVERBOUGHT')
            elif rsi_value > 60:
                momentum_score -= 0.5
                conditions.append('NEAR_OVERBOUGHT')

            # MACD Component
            if macd_value > 1.5:  # Strong bullish MACD
                momentum_score += 1.0
                conditions.append('STRONG_BULLISH_MACD')
            elif macd_value > 0:  # Moderate bullish MACD
                momentum_score += 0.5
                conditions.append('BULLISH_MACD')
            elif macd_value < -1.5:  # Strong bearish MACD
                momentum_score -= 1.0
                conditions.append('STRONG_BEARISH_MACD')
            elif macd_value < 0:  # Moderate bearish MACD
                momentum_score -= 0.5
                conditions.append('BEARISH_MACD')

            score += (momentum_score / 2) * 0.4  # Average and apply weight

            # 3. Volatility Adjustment (20% weight)
            volatility_score = 0
            if vol_ratio > 50:  # Extremely high relative volatility
                volatility_score = -0.5  # Strong bias towards neutral
                conditions.append('EXTREME_VOLATILITY')
            elif vol_ratio > 20:  # Very high relative volatility
                volatility_score = -0.3
                conditions.append('HIGH_VOLATILITY')

            score += volatility_score * 0.2

            # 4. Volume Confirmation (10% weight)
            volume_score = 0
            vol_ratio = volume_signals['volume_ratio']

            if vol_ratio > 1.2:
                if momentum_score > 0:  # Confirming bullish momentum
                    volume_score = 0.5
                    conditions.append('VOLUME_CONFIRMS_BULLISH')
                else:  # Confirming bearish momentum
                    volume_score = -0.5
                    conditions.append('VOLUME_CONFIRMS_BEARISH')

            score += volume_score * 0.1

            # Special Cases
            if 'POTENTIAL_REVERSAL' in conditions:
                if 'STRONG_BULLISH_MACD' in conditions and 'NEAR_OVERSOLD' in conditions:
                    score = max(score, 0.2)  # Ensure at least weak buy signal

            if 'EXTREME_VOLATILITY' in conditions:
                if abs(score) < 0.4:  # Weak signals in extreme volatility
                    score = 0  # Force neutral

            # Final scaling and rounding
            final_score = round(np.tanh(score), 2)

            # Log conditions for debugging
            logger.info(f"Signal conditions: {conditions}")
            logger.info(f"Raw score: {score}, Final score: {final_score}")

            return final_score

        except Exception as e:
            logger.error(f"Error calculating signal score: {str(e)}")
            return 0.0

    def _determine_primary_signal(self, score: float) -> str:
        """Determine primary trading signal based on score thresholds"""
        abs_score = abs(score)

        if abs_score < 0.15:
            return 'NEUTRAL'
        elif score > 0:
            if abs_score > 0.5:
                return 'STRONG_BUY'
            return 'BUY'
        else:
            if abs_score > 0.5:
                return 'STRONG_SELL'
            return 'SELL'


    def generate_recommendations(self, signals: Dict, risk_params: Dict) -> Dict:
        """Generate enhanced recommendations with detailed context"""
        try:
            # Get base metrics
            signal_strength = abs(signals.get('combined_score', 0))
            primary_signal = signals.get('primary_signal', 'NEUTRAL')
            price_action = signals.get('price_action', {})
            regime_data = signals.get('regime', {})

            conditions = []

            # 1. Market Context
            trend_type = regime_data.get('trend', {}).get('type')
            if trend_type:
                conditions.append(f"Market: {trend_type}")

            # 2. Technical Signals
            rsi_value = price_action.get('rsi', {}).get('value', 0)
            macd_value = price_action.get('macd', {}).get('value', 0)

            # RSI Context
            if rsi_value < 30:
                conditions.append(f"Strong Oversold (RSI: {rsi_value:.1f})")
            elif rsi_value < 40:
                conditions.append(f"Approaching Oversold (RSI: {rsi_value:.1f})")
                if macd_value > 1.5:
                    conditions.append("Potential Bullish Reversal Setup")
            elif rsi_value > 70:
                conditions.append(f"Strong Overbought (RSI: {rsi_value:.1f})")
            elif rsi_value > 60:
                conditions.append(f"Approaching Overbought (RSI: {rsi_value:.1f})")
                if macd_value < -1.5:
                    conditions.append("Potential Bearish Reversal Setup")

            # MACD Context
            macd_strength = "Strong" if abs(macd_value) > 1.5 else "Moderate"
            conditions.append(f"{macd_strength} MACD {price_action['macd']['signal']} ({macd_value:.2f})")

            # 3. Volume Context
            vol_analysis = signals.get('volume_analysis', {})
            vol_ratio = vol_analysis.get('volume_ratio', 1.0)
            if vol_ratio > 1.2:
                conditions.append(f"High Volume (Ratio: {vol_ratio:.2f})")
            elif vol_ratio < 0.8:
                conditions.append(f"Low Volume (Ratio: {vol_ratio:.2f})")

            # 4. Volatility Context
            vol_metrics = regime_data.get('volatility', {}).get('metrics', {})
            opt_vol = vol_metrics.get('option_volatility', 0)
            idx_vol = vol_metrics.get('index_volatility', 1)
            vol_ratio = opt_vol / idx_vol

            if vol_ratio > 50:
                conditions.append(f"Extreme Volatility ({vol_ratio:.0f}x Index)")
            elif vol_ratio > 20:
                conditions.append(f"High Volatility ({vol_ratio:.0f}x Index)")

            # 5. Trade Parameters
            entry = risk_params.get('entry_price', 0)
            if entry > 0:
                sl_pct = round(abs(risk_params['stop_loss'] - entry) / entry * 100, 2)
                target_pct = round(abs(risk_params['target'] - entry) / entry * 100, 2)

                if primary_signal != 'NEUTRAL':
                    conditions.append(f"Risk: {sl_pct}% / Reward: {target_pct}%")

                # Add warning for high volatility risk
                if vol_ratio > 20:
                    conditions.append("Caution: Wide stops recommended due to high volatility")

            return {
                'signal': primary_signal,
                'confidence': signal_strength,
                'entry_price': risk_params.get('entry_price', 0),
                'stop_loss': risk_params.get('stop_loss', 0),
                'stop_loss_percentage': sl_pct,
                'target': risk_params.get('target', 0),
                'target_percentage': target_pct,
                'risk_reward': risk_params.get('risk_reward_ratio', 1.5),
                'conditions': conditions
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'entry_price': risk_params.get('entry_price', 0),
                'stop_loss': risk_params.get('stop_loss', 0),
                'stop_loss_percentage': 0,
                'target': risk_params.get('target', 0),
                'target_percentage': 0,
                'risk_reward': risk_params.get('risk_reward_ratio', 1.5),
                'conditions': ["Error generating recommendations"]
            }

class OptionsGreeksAnalyzer:
    """Enhanced options analysis with Greeks calculations"""

    def __init__(self):
        self.risk_free_rate = 0.05  # Can be updated with actual rates

    def calculate_greeks(self,
                        S: float,  # Spot price
                        K: float,  # Strike price
                        T: float,  # Time to expiration (in years)
                        r: float,  # Risk-free rate
                        sigma: float,  # Volatility
                        option_type: str = 'call') -> Dict:
        """Calculate option Greeks using Black-Scholes model"""
        try:
            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            # Adjust for put options
            sign = 1 if option_type.lower() == 'call' else -1

            # Calculate Greeks
            delta = sign * norm.cdf(sign * d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - sign * r * K * np.exp(-r*T) * norm.cdf(sign * d2)
            vega = S * np.sqrt(T) * norm.pdf(d1)
            rho = sign * K * T * np.exp(-r*T) * norm.cdf(sign * d2)

            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta/365),  # Daily theta
                'vega': float(vega/100),    # 1% volatility change
                'rho': float(rho/100)       # 1% rate change
            }

        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return None

    def analyze_implied_volatility(self,
                                 option_chain: pd.DataFrame,
                                 spot_price: float) -> Dict:
        """Analyze volatility skew and term structure"""
        try:
            # Group by expiration
            expirations = option_chain.groupby('expiration')

            volatility_analysis = {
                'skew': {},
                'term_structure': {},
                'average_iv': float(option_chain['implied_vol'].mean()),
                'iv_range': {
                    'min': float(option_chain['implied_vol'].min()),
                    'max': float(option_chain['implied_vol'].max())
                }
            }

            # Analyze volatility skew
            for exp, group in expirations:
                atm_strike = group.iloc[(group['strike'] - spot_price).abs().argsort()[:1]].index[0]
                skew_data = group.sort_values('strike')

                volatility_analysis['skew'][exp] = {
                    'strikes': skew_data['strike'].tolist(),
                    'implied_vols': skew_data['implied_vol'].tolist(),
                    'atm_vol': float(group.loc[atm_strike, 'implied_vol'])
                }

            return volatility_analysis

        except Exception as e:
            logger.error(f"Error analyzing implied volatility: {str(e)}")
            return None

class EnhancedOIAnalyzer:
    """Enhanced Open Interest analysis"""

    def analyze_oi_patterns(self,
                          option_chain: pd.DataFrame,
                          historical_oi: pd.DataFrame) -> Dict:
        """Analyze OI patterns and concentration"""
        try:
            # Current OI analysis
            total_oi = option_chain['open_interest'].sum()
            max_oi_strike = option_chain.loc[option_chain['open_interest'].idxmax()]

            # OI concentration analysis
            oi_concentration = option_chain.groupby('strike')['open_interest'].sum()
            top_strikes = oi_concentration.nlargest(5)

            # Historical OI trend
            if not historical_oi.empty:
                oi_trend = self._analyze_historical_oi(historical_oi)
            else:
                oi_trend = {'trend': 'INSUFFICIENT_DATA'}

            return {
                'current_analysis': {
                    'total_oi': int(total_oi),
                    'max_oi_strike': float(max_oi_strike['strike']),
                    'max_oi': int(max_oi_strike['open_interest']),
                    'concentration': {
                        'strikes': top_strikes.index.tolist(),
                        'values': top_strikes.values.tolist()
                    }
                },
                'historical_trend': oi_trend,
                'put_call_ratio': self._calculate_put_call_ratio(option_chain)
            }

        except Exception as e:
            logger.error(f"Error analyzing OI patterns: {str(e)}")
            return None

    def _analyze_historical_oi(self, historical_oi: pd.DataFrame) -> Dict:
        """Analyze historical OI trends"""
        try:
            # Calculate daily changes
            historical_oi['oi_change'] = historical_oi['open_interest'].diff()

            # Calculate trend metrics
            recent_trend = historical_oi['oi_change'].tail(5).mean()

            trend_strength = abs(recent_trend) / historical_oi['open_interest'].mean()

            return {
                'trend': 'INCREASING' if recent_trend > 0 else 'DECREASING',
                'strength': float(trend_strength),
                'daily_changes': historical_oi['oi_change'].tail(5).tolist()
            }

        except Exception as e:
            logger.error(f"Error analyzing historical OI: {str(e)}")
            return None

    def _calculate_put_call_ratio(self, option_chain: pd.DataFrame) -> float:
        """Calculate Put/Call ratio"""
        try:
            puts_oi = option_chain[option_chain['option_type'] == 'put']['open_interest'].sum()
            calls_oi = option_chain[option_chain['option_type'] == 'call']['open_interest'].sum()

            return float(puts_oi / calls_oi) if calls_oi > 0 else 0

        except Exception as e:
            logger.error(f"Error calculating P/C ratio: {str(e)}")
            return 0.0

class OptionsDataValidator:
    """Validator for options data"""

    @staticmethod
    def validate_timestamp(timestamp: str) -> bool:
        """Validate timestamp format"""
        try:
            datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_ohlcv(data: List[List]) -> bool:
        """Validate OHLCV data format"""
        if not data or not isinstance(data, list):
            return False

        for row in data:
            if not isinstance(row, list) or len(row) != 6:
                return False

            # Validate timestamp
            if not OptionsDataValidator.validate_timestamp(row[0]):
                return False

            # Validate numeric values
            try:
                all(float(x) >= 0 for x in row[1:])
            except ValueError:
                return False

        return True

    @staticmethod
    def validate_oi_data(data: List[Dict]) -> bool:
        """Validate OI data format"""
        if not data or not isinstance(data, list):
            return False

        for entry in data:
            if not isinstance(entry, dict):
                return False

            if 'time' not in entry or 'oi' not in entry:
                return False

            if not OptionsDataValidator.validate_timestamp(entry['time']):
                return False

            try:
                float(entry['oi'])
            except ValueError:
                return False

        return True


# Flask routes welcome message
@app.route('/')
def welcome():
    """Welcome page route"""
    return jsonify({
        "message": "Welcome to Enhanced Options Analytics Platform",
        "version": "2.0.0",
        "status": "active"
    })

# Flask routes and utility functions

# Update the Flask route to handle both payload types
@app.route('/analyze', methods=['POST'])
def analyze_instrument():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        # Extract data - note the change from option_symbol to symbol
        index_data = data.get('index_data')
        option_data = data.get('option_data')
        symbol = data.get('option_symbol') or data.get('symbol')  # Try both fields

        # Debug logging
        logger.info(f"Received data: index_data={bool(index_data)}, option_data={bool(option_data)}, symbol={symbol}")

        # Validate minimum required fields
        if not index_data or not symbol:
            return jsonify({
                'status': 'error',
                'message': f'Missing required data fields. Need index_data and symbol. Got: {list(data.keys())}'
            }), 400

        # If option_data is not provided, use index_data
        if not option_data:
            option_data = index_data

        # Initialize analyzer and process data
        analyzer = EnhancedOptionsAnalyzer()
        result = analyzer.analyze_instrument(
            index_data=index_data,
            option_data=option_data,
            symbol=symbol
        )

        # Ensure result is serializable
        return jsonify(convert_to_serializable(result))

    except Exception as e:
        logger.error(f"Error in analysis endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
def calculate_option_returns(entry_price: float,
                           exit_price: float,
                           position_type: str = 'LONG') -> float:
    """Calculate returns for an option trade"""
    if position_type == 'LONG':
        return (exit_price - entry_price) / entry_price * 100
    else:  # SHORT
        return (entry_price - exit_price) / entry_price * 100

def calculate_position_size(account_size: float,
                          risk_per_trade: float,
                          entry_price: float,
                          stop_loss: float) -> int:
    """Calculate position size based on risk management"""
    risk_amount = account_size * (risk_per_trade / 100)
    per_unit_risk = abs(entry_price - stop_loss)

    if per_unit_risk == 0:
        return 0

    position_size = risk_amount / per_unit_risk
    return int(position_size)

def calculate_risk_metrics(entry: float,
                         stop_loss: float,
                         target: float) -> Dict:
    """Calculate risk metrics for a trade"""
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)

    return {
        'risk_amount': risk,
        'reward_amount': reward,
        'risk_reward_ratio': reward / risk if risk != 0 else 0,
        'risk_percentage': (risk / entry) * 100,
        'reward_percentage': (reward / entry) * 100
    }

def format_signal_message(signal: str,
                         entry: float,
                         stop_loss: float,
                         target: float,
                         conditions: List[str]) -> str:
    """Format trading signal message"""
    return f"""
    Signal: {signal}
    Entry Price: {entry:.2f}
    Stop Loss: {stop_loss:.2f} ({((stop_loss - entry) / entry * 100):.1f}%)
    Target: {target:.2f} ({((target - entry) / entry * 100):.1f}%)
    Conditions: {', '.join(conditions)}
    """
def main():
    """Main function to run the server"""
    try:
        print("Starting server setup...")
        
        # Get port from Heroku environment, default to 5000
        port = int(os.environ.get('PORT', 5000))
        
        print(f"\nServer starting on port {port}")
        print("\nAvailable endpoints:")
        print("  POST /analyze - Analyze options data")
        print("  GET / - Check API status")

        # Start Flask app
        app.run(host='0.0.0.0', port=port)

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run main function
    main()

