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
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.stats import norm, stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types"""
    # Handle None/null values
    if obj is None:
        return None
        
    # Updated NumPy integer types
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8,
                       np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
        
    # Updated NumPy float types
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
        
    # NumPy boolean
    if isinstance(obj, np.bool_):
        return bool(obj)
        
    # NumPy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
        
    # Handle datetime
    if isinstance(obj, np.datetime64):
        return str(obj)
        
    # Recursive handling for dictionaries
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
        
    # Recursive handling for lists/tuples
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
        
    # Return unchanged if no conversion needed
    return obj

def parse_option_symbol(symbol: str) -> Dict:
    """
    Parse option symbol focusing on month format first:
    SENSEX30JAN2576000CE format:
        SENSEX - Instrument
        30     - Day
        JAN    - Month
        25     - Year (2025)
        76000  - Strike
        CE     - Option Type
    """
    try:
        # First check if it's an option
        if not (symbol.endswith('CE') or symbol.endswith('PE')):
            return {'type': 'stock', 'symbol': symbol}
            
        # Extract option type
        option_type = symbol[-2:]
        remaining = symbol[:-2]  # Remove CE/PE
        
        # Define months - this is our primary check
        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        # Look for month name first
        found_month = None
        month_position = -1
        for month in months:
            if month in remaining:
                found_month = month
                month_position = remaining.index(month)
                break
        
        if not found_month:
            logger.error(f"No valid month found in symbol {symbol}")
            return {'type': 'stock', 'symbol': symbol}
            
        try:
            # Everything before month position (excluding 2 chars for day)
            instrument = remaining[:month_position-2].strip()
            
            # Get the day (2 chars before month)
            day = int(remaining[month_position-2:month_position])
            
            # Get year (2 chars after month)
            year = 2000 + int(remaining[month_position+3:month_position+5])
            
            # Get strike price (everything after year)
            strike = float(remaining[month_position+5:])
            
            # Validate date
            expiry_date = datetime(year, months[found_month], day)
            days_to_expiry = (expiry_date - datetime.now()).days
            
            return {
                'type': 'option',
                'instrument': instrument,
                'strike': strike,
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'days_to_expiry': max(0, days_to_expiry),
                'option_type': 'put' if option_type == 'PE' else 'call',
                'year': year,
                'month': months[found_month],
                'day': day,
                'month_name': found_month
            }
            
        except ValueError as e:
            logger.error(f"Error parsing date components: {str(e)}")
            return {'type': 'stock', 'symbol': symbol}
            
    except Exception as e:
        logger.error(f"Error parsing symbol {symbol}: {str(e)}")
        return {'type': 'stock', 'symbol': symbol}

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
    """Class to hold market context with separate index and option data"""
    # Index/Underlying data
    index_price: float
    index_history: pd.DataFrame
    
    # Option specific data
    option_price: float
    option_history: pd.DataFrame
    option_chain: pd.DataFrame  # Full option chain data
    strike_price: float
    days_to_expiry: int
    option_type: str  # 'call' or 'put'
    
    # OI data
    oi_history: pd.DataFrame

    @property
    def moneyness(self) -> float:
        """Calculate option moneyness percentage"""
        if self.option_type == 'call':
            return (self.strike_price - self.index_price) / self.index_price * 100
        else:
            return (self.index_price - self.strike_price) / self.index_price * 100
            
    @property
    def is_otm(self) -> bool:
        """Check if option is OTM"""
        if self.option_type == 'call':
            return self.strike_price > self.index_price
        else:
            return self.strike_price < self.index_price

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

            # Analyze OI patterns - use the class method instead of undefined reference
            oi_analysis = self.analyze_oi_patterns_regime(context.oi_history)

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

    def analyze_oi_patterns_regime(self, oi_df: pd.DataFrame) -> Dict:
        """Analyze Open Interest patterns for regime detection"""
        try:
            if len(oi_df) < 2:
                return {'trend': 'INSUFFICIENT_DATA'}

            # Create a copy of the DataFrame
            df = oi_df.copy()
            
            # Calculate OI changes using loc
            df.loc[:, 'oi_change'] = df['oi'].pct_change()
            df.loc[:, 'oi_ma'] = TechnicalIndicators.calculate_sma(df['oi'], 5)

            latest = df.iloc[-1]

            # Detect OI trend
            oi_trend = 'INCREASING' if latest['oi'] > latest['oi_ma'] else 'DECREASING'

            return {
                'trend': oi_trend,
                'buildup': 'UNKNOWN',
                'change_percentage': float(latest['oi_change'] * 100) if 'oi_change' in latest else 0.0
            }

        except Exception as e:
            logger.error(f"Error analyzing OI patterns: {str(e)}")
            return {'trend': 'ERROR', 'error': str(e)}


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

    def _process_data(self, data: List[List], *, is_index: bool = False) -> pd.DataFrame:
        """
        Process raw OHLCV data with special handling for index data.
        Note: is_index parameter is keyword-only to prevent argument conflicts.
        """
        try:
            # Create DataFrame with proper column names
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Convert all price columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            if not is_index:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                df['oi'] = df['volume'].cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {'index' if is_index else 'option'} data: {str(e)}")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def analyze_instrument(self, index_data: List[List], option_data: List[List], symbol: str) -> Dict:
        """Analyze any financial instrument (index, stock, or option)"""
        try:
            # Parse the symbol
            instrument_info = parse_option_symbol(symbol)
            
            # Process index data
            index_df = self._process_data(data=index_data, is_index=True)
            if index_df.empty:
                raise ValueError("Invalid index data provided")
                
            # Process option data if available
            option_df = self._process_data(data=option_data, is_index=False) if option_data else pd.DataFrame()
            
            # Create market context
            context = MarketContext(
                index_price=float(index_df['close'].iloc[-1]),
                index_history=index_df,
                option_price=float(option_df['close'].iloc[-1]) if not option_df.empty else 0.0,
                option_history=option_df,
                option_chain=pd.DataFrame(),
                strike_price=instrument_info.get('strike', 0.0),
                days_to_expiry=instrument_info.get('days_to_expiry', 0),
                option_type=instrument_info.get('option_type', ''),
                oi_history=option_df[['volume', 'oi']] if not option_df.empty else pd.DataFrame()
            )
            
            # Generate signals based on available data
            signals = self.generate_signals(context, {})  # Empty regime dict for now
            
            # Calculate risk parameters
            risk_params = self.calculate_risk_parameters(context, signals, instrument_info)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(signals, risk_params)
            
            return {
                'status': 'success',
                'analysis': {
                    'instrument_info': instrument_info,
                    'signals': signals,
                    'risk_parameters': risk_params,
                    'recommendations': recommendations
                }
            }
            
        except Exception as e:
            logger.error(f"Error in instrument analysis: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

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
        """Calculate risk parameters with focus on option positions using option data"""
        try:
            # Get regime data safely
            regime = self.regime_detector.detect_regime(context)
            volatility_state = regime.get('volatility', {}).get('state', 'NORMAL') if regime else 'NORMAL'

            # Get current prices - use option price for options
            if instrument_info['type'] == 'option':
                # Use option price for entry if available
                if not context.option_history.empty:
                    entry_price = float(context.option_history['close'].iloc[-1])
                else:
                    raise ValueError("Option data required for option analysis")
            else:
                # Use index price for stocks/indices
                entry_price = context.index_price

            combined_score = signals.get('combined_score', 0) if signals else 0

            # For options specifically
            if instrument_info['type'] == 'option':
                # Determine position side based on option type and market view
                is_bearish = combined_score < 0
                is_put = instrument_info['option_type'] == 'put'

                # Option-specific calculations
                if (is_put and is_bearish) or (not is_put and not is_bearish):
                    # Favorable setup
                    if volatility_state == 'HIGH':
                        sl_percentage = 0.40  # 40% for high volatility
                    elif volatility_state == 'LOW':
                        sl_percentage = 0.25  # 25% for low volatility
                    else:
                        sl_percentage = 0.30  # 30% for normal volatility

                    stop_loss = round(entry_price * (1 - sl_percentage), 2)
                    target = round(entry_price * (1 + (sl_percentage * self.risk_reward_ratio)), 2)

                    return {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'action': 'BUY',
                        'risk_reward_ratio': self.risk_reward_ratio,
                        'risk_per_trade': round(entry_price - stop_loss, 2),
                        'potential_reward': round(target - entry_price, 2),
                        'price_source': 'option',
                        'volatility_state': volatility_state,
                        'option_details': {
                            'strike': instrument_info['strike'],
                            'days_to_expiry': instrument_info['days_to_expiry'],
                            'type': instrument_info['option_type'],
                            'spot_price': context.index_price
                        }
                    }
                else:
                    # Unfavorable setup
                    return {
                        'entry_price': entry_price,
                        'action': 'DO_NOT_TRADE',
                        'reason': f"{'Bullish' if is_bearish else 'Bearish'} signal mismatched with {instrument_info['option_type'].upper()} option",
                        'price_source': 'option',
                        'volatility_state': volatility_state,
                        'option_details': {
                            'strike': instrument_info['strike'],
                            'days_to_expiry': instrument_info['days_to_expiry'],
                            'type': instrument_info['option_type'],
                            'spot_price': context.index_price
                        }
                    }
            else:
                # For index/stock analysis
                sl_percentage = 0.03  # 3% for stocks
                target_percentage = sl_percentage * 1.5  # 1.5 RR ratio

                stop_loss = round(entry_price * (1 + sl_percentage), 2)
                target = round(entry_price * (1 - target_percentage), 2)

                return {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'action': 'SELL' if combined_score < 0 else 'BUY',
                    'risk_reward_ratio': 1.5,
                    'risk_per_trade': round(abs(stop_loss - entry_price), 2),
                    'potential_reward': round(abs(target - entry_price), 2),
                    'price_source': 'index/stock',
                    'volatility_state': volatility_state
                }

        except Exception as e:
            logger.error(f"Error calculating risk parameters: {str(e)}")
            return {
                'entry_price': entry_price if 'entry_price' in locals() else 0.0,
                'action': 'ERROR',
                'error': str(e),
                'price_source': 'option' if instrument_info['type'] == 'option' else 'index/stock'
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
        """Generate recommendations with option-specific focus"""
        try:
            # Extract risk parameters
            action = risk_params.get('action', 'NEUTRAL')
            entry_price = risk_params.get('entry_price', 0.0)
            stop_loss = risk_params.get('stop_loss', 0.0)
            target = risk_params.get('target', 0.0)
            option_details = risk_params.get('option_details', {})

            # Calculate percentages safely
            sl_pct = round(abs(stop_loss - entry_price) / entry_price * 100, 2) if entry_price != 0 else 0.0
            target_pct = round(abs(target - entry_price) / entry_price * 100, 2) if entry_price != 0 else 0.0

            # Get signal data
            signal_strength = abs(signals.get('combined_score', 0)) if signals else 0

            conditions = []

            # Add option-specific conditions
            if option_details:
                strike = option_details.get('strike', 0)
                spot = option_details.get('spot_price', 0)
                days = option_details.get('days_to_expiry', 0)
                opt_type = option_details.get('type', '')

                if strike and spot:
                    moneyness = abs(1 - (strike/spot)) * 100
                    if moneyness < 1:
                        conditions.append("ATM Option")
                    elif moneyness < 3:
                        conditions.append("Near-the-money Option")
                    else:
                        conditions.append(f"{'OTM' if strike > spot else 'ITM'} Option ({moneyness:.1f}% away)")

                if days:
                    if days < 7:
                        conditions.append("Very Short Expiry")
                    elif days < 15:
                        conditions.append("Short Expiry")
                    elif days > 60:
                        conditions.append("Long Dated Option")

            # Add general market conditions
            if signals and signals.get('price_action'):
                price_action = signals['price_action']
                if price_action.get('rsi', {}).get('signal') == 'OVERSOLD':
                    conditions.append("Market Oversold")
                elif price_action.get('rsi', {}).get('signal') == 'OVERBOUGHT':
                    conditions.append("Market Overbought")

            if action == 'DO_NOT_TRADE':
                conditions.append(risk_params.get('reason', 'Unfavorable Setup'))

            return {
                'signal': 'BUY' if action == 'BUY' else 'NEUTRAL',
                'confidence': signal_strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'stop_loss_percentage': sl_pct,
                'target': target,
                'target_percentage': target_pct,
                'risk_reward': risk_params.get('risk_reward_ratio', 1.5),
                'action': action,
                'conditions': conditions
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'entry_price': entry_price,
                'stop_loss': 0.0,
                'stop_loss_percentage': 0.0,
                'target': 0.0,
                'target_percentage': 0.0,
                'risk_reward': 1.5,
                'conditions': ["Error generating recommendations"]
            }

class OptionsRiskCalculator:
    """Calculate risk parameters specifically for options"""
    
    def __init__(self):
        self.min_premium = 5.0  # Minimum premium to consider
        self.max_risk_percent = 0.30  # Maximum risk per trade
        
    def calculate_option_risk(self, context: MarketContext, signal: Dict) -> Dict:
        """Calculate option-specific risk parameters"""
        try:
            option_price = context.option_price
            
            # Skip if premium is too low
            if option_price < self.min_premium:
                return {
                    'action': 'DO_NOT_TRADE',
                    'reason': 'Premium too low',
                    'premium': option_price
                }
                
            # Calculate base risk parameters
            if signal['action'] == 'BUY':
                max_loss = option_price  # Full premium at risk
                stop_loss = option_price * (1 - self.max_risk_percent)
                target = option_price * (1 + self.max_risk_percent * 1.5)  # 1.5 RR ratio
            else:  # SELL
                max_loss = option_price * 2  # 2x premium as max loss
                stop_loss = option_price * (1 + self.max_risk_percent)
                target = option_price * (1 - self.max_risk_percent)
                
            # Calculate Greeks exposure
            greeks = self.calculate_position_greeks(context)
            
            return {
                'action': signal['action'],
                'entry': option_price,
                'stop_loss': round(stop_loss, 2),
                'target': round(target, 2),
                'max_loss': round(max_loss, 2),
                'risk_reward_ratio': 1.5,
                'position_sizing': self.calculate_position_size(option_price, max_loss),
                'greeks_exposure': greeks,
                'underlying_ref': {
                    'index_price': context.index_price,
                    'strike': context.strike_price,
                    'moneyness': context.moneyness
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating option risk: {str(e)}")
            return {
                'action': 'ERROR',
                'error': str(e)
            }
            
    def calculate_position_size(self, premium: float, max_loss: float) -> Dict:
        """Calculate suggested position size"""
        # Assume account size of 100,000 for example
        account_size = 100000
        risk_per_trade = account_size * 0.02  # 2% risk per trade
        
        max_contracts = int(risk_per_trade / max_loss)
        suggested_contracts = max(1, int(max_contracts * 0.7))  # 70% of max
        
        return {
            'max_contracts': max_contracts,
            'suggested_contracts': suggested_contracts,
            'premium_per_contract': premium,
            'total_premium': premium * suggested_contracts
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

    def analyze_oi_patterns(self, oi_df: pd.DataFrame) -> Dict:
        """Analyze Open Interest patterns"""
        try:
            if len(oi_df) < 2:
                return {'trend': 'INSUFFICIENT_DATA'}

            # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
            df = oi_df.copy()
            
            # Calculate OI changes using loc
            df.loc[:, 'oi_change'] = df['oi'].pct_change()
            df.loc[:, 'oi_ma'] = TechnicalIndicators.calculate_sma(df['oi'], 5)

            latest = df.iloc[-1]

            # Detect OI trend
            oi_trend = 'INCREASING' if latest['oi'] > latest['oi_ma'] else 'DECREASING'

            # Calculate buildup
            price_up = latest['close'] > df['close'].iloc[-2] if 'close' in df.columns else None
            oi_up = latest['oi'] > df['oi'].iloc[-2]

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


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_instrument():
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        # Log the start of request processing
        logger.info("Starting analysis request")
        
        data = request.get_json()
        logger.info(f"Received request data: {data.keys() if data else 'No data'}")

        if not data:
            error_response = {
                'status': 'error',
                'message': 'No data provided'
            }
            logger.error("No data provided in request")
            return jsonify(error_response), 400

        # Extract data
        index_data = data.get('index_data')
        option_data = data.get('option_data')
        symbol = data.get('option_symbol') or data.get('symbol')

        # Detailed logging
        logger.info(f"Processing request for symbol: {symbol}")
        logger.info(f"Index data length: {len(index_data) if index_data else 0}")
        logger.info(f"Option data length: {len(option_data) if option_data else 0}")

        # Validate minimum required fields
        if not index_data or not symbol:
            error_response = {
                'status': 'error',
                'message': f'Missing required data fields. Need index_data and symbol. Got: {list(data.keys())}'
            }
            logger.error(f"Missing required fields: {error_response}")
            return jsonify(error_response), 400

        # If option_data is not provided, use index_data
        if not option_data:
            option_data = index_data
            logger.info("Using index_data as option_data")

        # Initialize analyzer and process data
        analyzer = EnhancedOptionsAnalyzer()
        result = analyzer.analyze_instrument(
            index_data=index_data,
            option_data=option_data,
            symbol=symbol
        )

        # Log the result before sending
        logger.info(f"Analysis completed. Result status: {result.get('status')}")
        
        # Convert result to JSON-serializable format
        serialized_result = convert_to_serializable(result)
        
        # Log the serialized result
        logger.info(f"Sending response: {serialized_result}")
        
        # Create response with CORS headers
        response = make_response(jsonify(serialized_result))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Content-Type'] = 'application/json'
        return response

    except Exception as e:
        error_msg = f"Error in analysis endpoint: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        error_response = {
            'status': 'error',
            'message': error_msg,
            'type': str(type(e).__name__)
        }
        response = make_response(jsonify(error_response))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Content-Type'] = 'application/json'
        return response, 500
        
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

        # Setup ngrok tunnel
        public_url = setup_server()

        if public_url:
            print(f"\nServer running at: {public_url}")
            print("Use this URL to connect to the Options Analysis API")
            print("\nAvailable endpoints:")
            print("  POST /analyze - Analyze options data")
            print("  GET / - Check API status")

            # Start Flask app with specific host and port
            app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
        else:
            print("Failed to establish ngrok tunnel")
            print("Please check your ngrok authentication token and internet connection")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        try:
            ngrok.kill()
        except:
            pass

if __name__ == '__main__':
    try:
        # Get port from environment variable (Heroku will set this)
        port = int(os.environ.get('PORT', 5000))
        
        # Log startup
        logger.info(f"Starting server on port {port}")
        
        # Start Flask app
        app.run(host='0.0.0.0', port=port)
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")