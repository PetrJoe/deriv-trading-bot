import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress
import ta  # Replace talib with ta library
from typing import Tuple, List, Dict, Optional, Union


class PatternRecognition:
    def __init__(self, df: pd.DataFrame, price_col: str = 'close', time_col: str = 'epoch'):
        """
        Initialize the pattern recognition class with price data.
        
        Args:
            df: DataFrame containing price data
            price_col: Column name for price data
            time_col: Column name for time data
        """
        self.df = df.copy()
        if time_col in self.df.columns and not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.set_index(time_col, inplace=True)
        self.price_col = price_col
        
        # Ensure we have high, low, open, close columns for pattern detection
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain {col} column")
        
        # Add columns for local extrema
        self._add_extrema()
        
    def _add_extrema(self, window: int = 5):
        """
        Add columns for local maxima and minima.
        
        Args:
            window: Window size for detecting local extrema
        """
        # Find local maxima and minima
        self.df['local_max'] = self.df.iloc[argrelextrema(self.df[self.price_col].values, np.greater_equal, 
                                                         order=window)[0]][self.price_col]
        self.df['local_min'] = self.df.iloc[argrelextrema(self.df[self.price_col].values, np.less_equal, 
                                                         order=window)[0]][self.price_col]
    
    def detect_all_patterns(self) -> Dict[str, Dict[str, Union[bool, float]]]:
        """
        Detect all implemented chart patterns and return results.
        
        Returns:
            Dictionary with pattern names as keys and detection results as values
        """
        results = {}
        
        # Reversal patterns
        results['head_and_shoulders'] = self.detect_head_and_shoulders()
        results['inverse_head_and_shoulders'] = self.detect_inverse_head_and_shoulders()
        results['double_top'] = self.detect_double_top()
        results['double_bottom'] = self.detect_double_bottom()
        results['triple_top'] = self.detect_triple_top()
        results['triple_bottom'] = self.detect_triple_bottom()
        results['rounding_top'] = self.detect_rounding_top()
        results['rounding_bottom'] = self.detect_rounding_bottom()
        
        # Continuation patterns
        results['rising_wedge'] = self.detect_rising_wedge()
        results['falling_wedge'] = self.detect_falling_wedge()
        results['symmetrical_triangle'] = self.detect_symmetrical_triangle()
        results['ascending_triangle'] = self.detect_ascending_triangle()
        results['descending_triangle'] = self.detect_descending_triangle()
        results['bullish_flag'] = self.detect_bullish_flag()
        results['bearish_flag'] = self.detect_bearish_flag()
        results['bullish_pennant'] = self.detect_bullish_pennant()
        results['bearish_pennant'] = self.detect_bearish_pennant()
        results['rectangle'] = self.detect_rectangle()
        results['cup_and_handle'] = self.detect_cup_and_handle()
        results['inverted_cup_and_handle'] = self.detect_inverted_cup_and_handle()
        results['diamond_top'] = self.detect_diamond_top()
        results['diamond_bottom'] = self.detect_diamond_bottom()
        results['megaphone'] = self.detect_megaphone()
        
        # Add pattern signals from TALib
        results.update(self.detect_candlestick_patterns())
        
        return results
    
    def get_trading_signal(self) -> Tuple[str, Dict[str, Dict]]:
        """
        Generate a trading signal based on detected patterns.
        
        Returns:
            Tuple containing the recommendation (Buy/Sell/Hold) and detailed pattern results
        """
        patterns = self.detect_all_patterns()
        
        # Count bullish and bearish signals
        bullish_count = 0
        bearish_count = 0
        
        pattern_signals = {}
        
        for pattern, result in patterns.items():
            if isinstance(result, dict) and 'detected' in result and result['detected']:
                if 'signal' in result:
                    if result['signal'] == 'bullish':
                        bullish_count += 1
                        pattern_signals[pattern] = result
                    elif result['signal'] == 'bearish':
                        bearish_count += 1
                        pattern_signals[pattern] = result
        
        # Generate recommendation based on pattern counts
        if bullish_count > bearish_count:
            return "Buy", pattern_signals
        elif bearish_count > bullish_count:
            return "Sell", pattern_signals
        else:
            return "Hold", pattern_signals
    
    def detect_head_and_shoulders(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Head and Shoulders pattern (bearish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Get local maxima
        maxima = self.df[self.df['local_max'].notna()].copy()
        
        if len(maxima) < 3:
            return result
        
        # Need at least 5 points for a proper H&S pattern (left shoulder, head, right shoulder)
        for i in range(len(maxima) - 4):
            # Get 5 consecutive peaks
            peaks = maxima.iloc[i:i+5]
            
            # Check if we have the pattern: shoulder (lower), head (higher), shoulder (lower)
            # with approximately equal shoulder heights
            if (peaks.iloc[0][self.price_col] < peaks.iloc[2][self.price_col] and 
                peaks.iloc[4][self.price_col] < peaks.iloc[2][self.price_col] and
                abs(peaks.iloc[0][self.price_col] - peaks.iloc[4][self.price_col]) / peaks.iloc[0][self.price_col] < 0.1):
                
                # Check for neckline (connecting the troughs between shoulders and head)
                trough1 = self.df[self.df.index > peaks.iloc[0].name][self.df.index < peaks.iloc[2].name]['local_min'].first_valid_index()
                trough2 = self.df[self.df.index > peaks.iloc[2].name][self.df.index < peaks.iloc[4].name]['local_min'].first_valid_index()
                
                if trough1 is not None and trough2 is not None:
                    trough1_val = self.df.loc[trough1, self.price_col]
                    trough2_val = self.df.loc[trough2, self.price_col]
                    
                    # Check if neckline is roughly horizontal (less than 5% difference)
                    if abs(trough1_val - trough2_val) / trough1_val < 0.05:
                        # Check if the current price is below the neckline (confirmation)
                        current_price = self.df[self.price_col].iloc[-1]
                        neckline = (trough1_val + trough2_val) / 2
                        
                        if current_price < neckline:
                            result['detected'] = True
                            result['confidence'] = min(1.0, (neckline - current_price) / neckline * 5)
                            result['neckline'] = neckline
                            result['target'] = neckline - (peaks.iloc[2][self.price_col] - neckline)
                            break
        
        return result
    
    def detect_inverse_head_and_shoulders(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Inverse Head and Shoulders pattern (bullish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Get local minima
        minima = self.df[self.df['local_min'].notna()].copy()
        
        if len(minima) < 3:
            return result
        
        # Need at least 5 points for a proper IH&S pattern (left shoulder, head, right shoulder)
        for i in range(len(minima) - 4):
            # Get 5 consecutive troughs
            troughs = minima.iloc[i:i+5]
            
            # Check if we have the pattern: shoulder (higher), head (lower), shoulder (higher)
            # with approximately equal shoulder heights
            if (troughs.iloc[0][self.price_col] > troughs.iloc[2][self.price_col] and 
                troughs.iloc[4][self.price_col] > troughs.iloc[2][self.price_col] and
                abs(troughs.iloc[0][self.price_col] - troughs.iloc[4][self.price_col]) / troughs.iloc[0][self.price_col] < 0.1):
                
                # Check for neckline (connecting the peaks between shoulders and head)
                peak1 = self.df[self.df.index > troughs.iloc[0].name][self.df.index < troughs.iloc[2].name]['local_max'].first_valid_index()
                peak2 = self.df[self.df.index > troughs.iloc[2].name][self.df.index < troughs.iloc[4].name]['local_max'].first_valid_index()
                
                if peak1 is not None and peak2 is not None:
                    peak1_val = self.df.loc[peak1, self.price_col]
                    peak2_val = self.df.loc[peak2, self.price_col]
                    
                    # Check if neckline is roughly horizontal (less than 5% difference)
                    if abs(peak1_val - peak2_val) / peak1_val < 0.05:
                        # Check if the current price is above the neckline (confirmation)
                        current_price = self.df[self.price_col].iloc[-1]
                        neckline = (peak1_val + peak2_val) / 2
                        
                        if current_price > neckline:
                            result['detected'] = True
                            result['confidence'] = min(1.0, (current_price - neckline) / neckline * 5)
                            result['neckline'] = neckline
                            result['target'] = neckline + (neckline - troughs.iloc[2][self.price_col])
                            break
        
        return result
    
    def detect_double_top(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Double Top pattern (bearish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Get local maxima
        maxima = self.df[self.df['local_max'].notna()].copy()
        
        if len(maxima) < 2:
            return result
        
        # Look at the last few peaks
        for i in range(max(0, len(maxima) - 5), len(maxima) - 1):
            peak1 = maxima.iloc[i]
            peak2 = maxima.iloc[i+1]
            
            # Check if peaks are roughly equal (within 2%)
            if abs(peak1[self.price_col] - peak2[self.price_col]) / peak1[self.price_col] < 0.02:
                # Find the trough between the peaks
                trough_idx = self.df[self.df.index > peak1.name][self.df.index < peak2.name]['local_min'].first_valid_index()
                
                if trough_idx is not None:
                    trough_val = self.df.loc[trough_idx, self.price_col]
                    
                    # Check if the current price is below the trough (confirmation)
                    current_price = self.df[self.price_col].iloc[-1]
                    
                    if current_price < trough_val:
                        result['detected'] = True
                        result['confidence'] = min(1.0, (trough_val - current_price) / trough_val * 5)
                        result['neckline'] = trough_val
                        result['target'] = trough_val - (peak1[self.price_col] - trough_val)
                        break
        
        return result
    
    def detect_double_bottom(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Double Bottom pattern (bullish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Get local minima
        minima = self.df[self.df['local_min'].notna()].copy()
        
        if len(minima) < 2:
            return result
        
        # Look at the last few troughs
        for i in range(max(0, len(minima) - 5), len(minima) - 1):
            trough1 = minima.iloc[i]
            trough2 = minima.iloc[i+1]
            
            # Check if troughs are roughly equal (within 2%)
            if abs(trough1[self.price_col] - trough2[self.price_col]) / trough1[self.price_col] < 0.02:
                # Find the peak between the troughs
                peak_idx = self.df[self.df.index > trough1.name][self.df.index < trough2.name]['local_max'].first_valid_index()
                
                if peak_idx is not None:
                    peak_val = self.df.loc[peak_idx, self.price_col]
                    
                    # Check if the current price is above the peak (confirmation)
                    current_price = self.df[self.price_col].iloc[-1]
                    
                    if current_price > peak_val:
                        result['detected'] = True
                        result['confidence'] = min(1.0, (current_price - peak_val) / peak_val * 5)
                        result['neckline'] = peak_val
                        result['target'] = peak_val + (peak_val - trough1[self.price_col])
                        break
        
        return result
    
    def detect_triple_top(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Triple Top pattern (bearish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Get local maxima
        maxima = self.df[self.df['local_max'].notna()].copy()
        
        if len(maxima) < 3:
            return result
        
        # Look at the last few peaks
        for i in range(max(0, len(maxima) - 7), len(maxima) - 2):
            peak1 = maxima.iloc[i]
            peak2 = maxima.iloc[i+1]
            peak3 = maxima.iloc[i+2]
            
            # Check if all three peaks are roughly equal (within 3%)
            if (abs(peak1[self.price_col] - peak2[self.price_col]) / peak1[self.price_col] < 0.03 and
                abs(peak1[self.price_col] - peak3[self.price_col]) / peak1[self.price_col] < 0.03 and
                abs(peak2[self.price_col] - peak3[self.price_col]) / peak2[self.price_col] < 0.03):
                
                # Find the troughs between the peaks
                trough1_idx = self.df[self.df.index > peak1.name][self.df.index < peak2.name]['local_min'].first_valid_index()
                trough2_idx = self.df[self.df.index > peak2.name][self.df.index < peak3.name]['local_min'].first_valid_index()
                
                if trough1_idx is not None and trough2_idx is not None:
                    trough1_val = self.df.loc[trough1_idx, self.price_col]
                    trough2_val = self.df.loc[trough2_idx, self.price_col]
                    
                    # Check if troughs are roughly equal (neckline)
                    if abs(trough1_val - trough2_val) / trough1_val < 0.05:
                        neckline = (trough1_val + trough2_val) / 2
                        
                        # Check if the current price is below the neckline (confirmation)
                        current_price = self.df[self.price_col].iloc[-1]
                        
                        if current_price < neckline:
                            result['detected'] = True
                            result['confidence'] = min(1.0, (neckline - current_price) / neckline * 5)
                            result['neckline'] = neckline
                            result['target'] = neckline - (peak1[self.price_col] - neckline)
                            break
        
        return result
    
    def detect_triple_bottom(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Triple Bottom pattern (bullish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Get local minima
        minima = self.df[self.df['local_min'].notna()].copy()
        
        if len(minima) < 3:
            return result
        
        # Look at the last few troughs
        for i in range(max(0, len(minima) - 7), len(minima) - 2):
            trough1 = minima.iloc[i]
            trough2 = minima.iloc[i+1]
            trough3 = minima.iloc[i+2]
            
            # Check if all three troughs are roughly equal (within 3%)
            if (abs(trough1[self.price_col] - trough2[self.price_col]) / trough1[self.price_col] < 0.03 and
                abs(trough1[self.price_col] - trough3[self.price_col]) / trough1[self.price_col] < 0.03 and
                abs(trough2[self.price_col] - trough3[self.price_col]) / trough2[self.price_col] < 0.03):
                
                # Find the peaks between the troughs
                peak1_idx = self.df[self.df.index > trough1.name][self.df.index < trough2.name]['local_max'].first_valid_index()
                peak2_idx = self.df[self.df.index > trough2.name][self.df.index < trough3.name]['local_max'].first_valid_index()
                
                if peak1_idx is not None and peak2_idx is not None:
                    peak1_val = self.df.loc[peak1_idx, self.price_col]
                    peak2_val = self.df.loc[peak2_idx, self.price_col]
                    
                    # Check if peaks are roughly equal (neckline)
                    if abs(peak1_val - peak2_val) / peak1_val < 0.05:
                        neckline = (peak1_val + peak2_val) / 2
                        
                        # Check if the current price is above the neckline (confirmation)
                        current_price = self.df[self.price_col].iloc[-1]
                        
                        if current_price > neckline:
                            result['detected'] = True
                            result['confidence'] = min(1.0, (current_price - neckline) / neckline * 5)
                            result['neckline'] = neckline
                            result['target'] = neckline + (neckline - trough1[self.price_col])
                            break
        
        return result
    
    def detect_rounding_bottom(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Rounding Bottom pattern (bullish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 20 periods for a proper rounding bottom
        if len(self.df) < 20:
            return result
        
        # Get local minima
        minima = self.df[self.df['local_min'].notna()].copy()
        
        if len(minima) < 3:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Find the lowest point
        lowest_idx = recent_data[self.price_col].idxmin()
        lowest_val = recent_data.loc[lowest_idx, self.price_col]
        
        # Check if the lowest point is roughly in the middle
        middle_idx = recent_data.index[len(recent_data) // 2]
        if abs((recent_data.index.get_loc(lowest_idx) - len(recent_data) // 2)) > len(recent_data) // 4:
            return result
        
        # Fit a quadratic curve to the data
        x = np.arange(len(recent_data))
        y = recent_data[self.price_col].values
        
        try:
            # Fit a quadratic polynomial (degree 2)
            coeffs = np.polyfit(x, y, 2)
            
            # Check if the curve is concave up (a > 0 for ax^2 + bx + c)
            if coeffs[0] > 0:
                # Calculate R-squared to measure fit quality
                p = np.poly1d(coeffs)
                y_fit = p(x)
                ss_tot = np.sum((y - np.mean(y))**2)
                ss_res = np.sum((y - y_fit)**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # If R-squared is high enough and current price is higher than the lowest point
                current_price = self.df[self.price_col].iloc[-1]
                if r_squared > 0.7 and current_price > lowest_val:
                    result['detected'] = True
                    result['confidence'] = min(1.0, r_squared)
                    result['lowest_point'] = lowest_val
                    result['target'] = current_price + (current_price - lowest_val)
        except:
            pass
        
        return result
    
    def detect_rounding_top(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Rounding Top pattern (bearish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 20 periods for a proper rounding top
        if len(self.df) < 20:
            return result
        
        # Get local maxima
        maxima = self.df[self.df['local_max'].notna()].copy()
        
        if len(maxima) < 3:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Find the highest point
        highest_idx = recent_data[self.price_col].idxmax()
        highest_val = recent_data.loc[highest_idx, self.price_col]
        
        # Check if the highest point is roughly in the middle
        middle_idx = recent_data.index[len(recent_data) // 2]
        if abs((recent_data.index.get_loc(highest_idx) - len(recent_data) // 2)) > len(recent_data) // 4:
            return result
        
        # Fit a quadratic curve to the data
        x = np.arange(len(recent_data))
        y = recent_data[self.price_col].values
        
        try:
            # Fit a quadratic polynomial (degree 2)
            coeffs = np.polyfit(x, y, 2)
            
            # Check if the curve is concave down (a < 0 for ax^2 + bx + c)
            if coeffs[0] < 0:
                # Calculate R-squared to measure fit quality
                p = np.poly1d(coeffs)
                y_fit = p(x)
                ss_tot = np.sum((y - np.mean(y))**2)
                ss_res = np.sum((y - y_fit)**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # If R-squared is high enough and current price is lower than the highest point
                current_price = self.df[self.price_col].iloc[-1]
                if r_squared > 0.7 and current_price < highest_val:
                    result['detected'] = True
                    result['confidence'] = min(1.0, r_squared)
                    result['highest_point'] = highest_val
                    result['target'] = current_price - (highest_val - current_price)
        except:
            pass
        
        return result
    
    def _fit_trendline(self, points: pd.DataFrame, col: str = 'close') -> Tuple[float, float, float]:
        """
        Fit a trendline to a set of points.
        
        Args:
            points: DataFrame containing the points
            col: Column name for the y-values
            
        Returns:
            Tuple containing (slope, intercept, r_value)
        """
        x = np.arange(len(points))
        y = points[col].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return slope, intercept, r_value
    
    def detect_rising_wedge(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Rising Wedge pattern (bearish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 10 periods for a proper wedge
        if len(self.df) < 10:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # For a rising wedge:
        # 1. Both trendlines should be rising (positive slope)
        # 2. Lower trendline should be steeper than upper trendline
        # 3. Trendlines should be converging
        if (high_slope > 0 and low_slope > 0 and 
            low_slope > high_slope and 
            high_r > 0.7 and low_r > 0.7):
            
            # Check if price is near the upper trendline (potential breakdown)
            current_price = self.df[self.price_col].iloc[-1]
            last_x = len(recent_data) - 1
            upper_trendline = high_slope * last_x + high_intercept
            lower_trendline = low_slope * last_x + low_intercept
            
            # If price is within 2% of the upper trendline
            if abs(current_price - upper_trendline) / upper_trendline < 0.02:
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
                result['upper_trendline'] = upper_trendline
                result['lower_trendline'] = lower_trendline
                result['target'] = lower_trendline - (upper_trendline - lower_trendline)
        
        return result
    
    def detect_falling_wedge(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Falling Wedge pattern (bullish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 10 periods for a proper wedge
        if len(self.df) < 10:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # For a falling wedge:
        # 1. Both trendlines should be falling (negative slope)
        # 2. Upper trendline should be steeper than lower trendline
        # 3. Trendlines should be converging
        if (high_slope < 0 and low_slope < 0 and 
            high_slope < low_slope and 
            high_r > 0.7 and low_r > 0.7):
            
            # Check if price is near the lower trendline (potential breakout)
            current_price = self.df[self.price_col].iloc[-1]
            last_x = len(recent_data) - 1
            upper_trendline = high_slope * last_x + high_intercept
            lower_trendline = low_slope * last_x + low_intercept
            
            # If price is within 2% of the lower trendline
            if abs(current_price - lower_trendline) / lower_trendline < 0.02:
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
                result['upper_trendline'] = upper_trendline
                result['lower_trendline'] = lower_trendline
                result['target'] = upper_trendline + (upper_trendline - lower_trendline)
        
        return result
    
    def detect_symmetrical_triangle(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Symmetrical Triangle pattern (continuation or reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'neutral', 'confidence': 0.0}
        
        # Need at least 10 periods for a proper triangle
        if len(self.df) < 10:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # For a symmetrical triangle:
        # 1. Upper trendline should be falling (negative slope)
        # 2. Lower trendline should be rising (positive slope)
        # 3. Slopes should be roughly equal in magnitude
        if (high_slope < 0 and low_slope > 0 and 
            abs(high_slope) / abs(low_slope) > 0.7 and abs(high_slope) / abs(low_slope) < 1.3 and
            high_r > 0.7 and low_r > 0.7):
            
            # Calculate where trendlines converge
            # Solve: high_slope * x + high_intercept = low_slope * x + low_intercept
            convergence_x = (low_intercept - high_intercept) / (high_slope - low_slope)
            
            # Check if convergence point is within reasonable future range (not too far)
            if convergence_x > len(recent_data) and convergence_x < len(recent_data) * 1.5:
                # Determine signal based on recent price movement
                current_price = self.df[self.price_col].iloc[-1]
                prev_price = self.df[self.price_col].iloc[-6]  # Look back 5 periods
                
                if current_price > prev_price:
                    result['signal'] = 'bullish'
                    result['target'] = current_price + (high_intercept - low_intercept)
                else:
                    result['signal'] = 'bearish'
                    result['target'] = current_price - (high_intercept - low_intercept)
                
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
                result['upper_trendline'] = high_slope * len(recent_data) + high_intercept
                result['lower_trendline'] = low_slope * len(recent_data) + low_intercept
        
        return result
    
    def detect_ascending_triangle(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Ascending Triangle pattern (bullish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 10 periods for a proper triangle
        if len(self.df) < 10:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # For an ascending triangle:
        # 1. Upper trendline should be roughly horizontal (slope near 0)
        # 2. Lower trendline should be rising (positive slope)
        if (abs(high_slope) < 0.001 and low_slope > 0 and 
            high_r > 0.7 and low_r > 0.7):
            
            # Check if price is near the upper trendline (potential breakout)
            current_price = self.df[self.price_col].iloc[-1]
            last_x = len(recent_data) - 1
            upper_trendline = high_slope * last_x + high_intercept
            
            # If price is within 2% of the upper trendline
            if abs(current_price - upper_trendline) / upper_trendline < 0.02:
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
                result['resistance'] = upper_trendline
                result['target'] = upper_trendline + (upper_trendline - (low_slope * last_x + low_intercept))
        
        return result
    
    def detect_descending_triangle(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Descending Triangle pattern (bearish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 10 periods for a proper triangle
        if len(self.df) < 10:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # For a descending triangle:
        # 1. Lower trendline should be roughly horizontal (slope near 0)
        # 2. Upper trendline should be falling (negative slope)
        if (abs(low_slope) < 0.001 and high_slope < 0 and 
            high_r > 0.7 and low_r > 0.7):
            
            # Check if price is near the lower trendline (potential breakdown)
            current_price = self.df[self.price_col].iloc[-1]
            last_x = len(recent_data) - 1
            lower_trendline = low_slope * last_x + low_intercept
            
            # If price is within 2% of the lower trendline
            if abs(current_price - lower_trendline) / lower_trendline < 0.02:
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
                result['support'] = lower_trendline
                result['target'] = lower_trendline - ((high_slope * last_x + high_intercept) - lower_trendline)
        
        return result
    
    def detect_bullish_flag(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Bullish Flag pattern (bullish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 15 periods for a proper flag
        if len(self.df) < 15:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # A flag consists of a sharp upward move (pole) followed by a consolidation (flag)
        # First, identify if there was a strong upward move
        first_third = recent_data.iloc[:len(recent_data)//3]
        first_third_change = (first_third[self.price_col].iloc[-1] - first_third[self.price_col].iloc[0]) / first_third[self.price_col].iloc[0]
        
        # Check for a strong upward move (>5% in the first third)
        if first_third_change > 0.05:
            # Now check for a consolidation pattern in the latter part
            latter_part = recent_data.iloc[len(recent_data)//3:]
            
            # Fit a trendline to the latter part
            slope, intercept, r_value = self._fit_trendline(latter_part)
            
            # For a bullish flag, the consolidation should be slightly downward or sideways
            if -0.005 < slope < 0.001 and r_value > 0.7:
                # Check if price is near the upper boundary of the flag (potential breakout)
                current_price = self.df[self.price_col].iloc[-1]
                
                # Calculate the height of the pole
                pole_height = first_third[self.price_col].iloc[-1] - first_third[self.price_col].iloc[0]
                
                result['detected'] = True
                result['confidence'] = min(1.0, r_value)
                result['pole_height'] = pole_height
                result['target'] = current_price + pole_height
        
        return result
    
    def detect_bearish_flag(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Bearish Flag pattern (bearish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 15 periods for a proper flag
        if len(self.df) < 15:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # A bearish flag consists of a sharp downward move (pole) followed by a consolidation (flag)
        # First, identify if there was a strong downward move
        first_third = recent_data.iloc[:len(recent_data)//3]
        first_third_change = (first_third[self.price_col].iloc[-1] - first_third[self.price_col].iloc[0]) / first_third[self.price_col].iloc[0]
        
        # Check for a strong downward move (<-5% in the first third)
        if first_third_change < -0.05:
            # Now check for a consolidation pattern in the latter part
            latter_part = recent_data.iloc[len(recent_data)//3:]
            
            # Fit a trendline to the latter part
            slope, intercept, r_value = self._fit_trendline(latter_part)
            
            # For a bearish flag, the consolidation should be slightly upward or sideways
            if 0 < slope < 0.005 and r_value > 0.7:
                # Check if price is near the lower boundary of the flag (potential breakdown)
                current_price = self.df[self.price_col].iloc[-1]
                
                # Calculate the height of the pole
                pole_height = abs(first_third[self.price_col].iloc[0] - first_third[self.price_col].iloc[-1])
                
                result['detected'] = True
                result['confidence'] = min(1.0, r_value)
                result['pole_height'] = pole_height
                result['target'] = current_price - pole_height
        
        return result
    

    def detect_bullish_pennant(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Bullish Pennant pattern (bullish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 15 periods for a proper pennant
        if len(self.df) < 15:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # A pennant consists of a sharp upward move (pole) followed by a symmetrical triangle (pennant)
        # First, identify if there was a strong upward move
        first_third = recent_data.iloc[:len(recent_data)//3]
        first_third_change = (first_third[self.price_col].iloc[-1] - first_third[self.price_col].iloc[0]) / first_third[self.price_col].iloc[0]
        
        # Check for a strong upward move (>5% in the first third)
        if first_third_change > 0.05:
            # Now check for a symmetrical triangle in the latter part
            latter_part = recent_data.iloc[len(recent_data)//3:]
            
            # Get local maxima and minima in the latter part
            highs = latter_part[latter_part['local_max'].notna()].copy()
            lows = latter_part[latter_part['local_min'].notna()].copy()
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Fit trendlines to highs and lows
                high_slope, high_intercept, high_r = self._fit_trendline(highs)
                low_slope, low_intercept, low_r = self._fit_trendline(lows)
                
                # For a pennant, the triangle should be converging
                if (high_slope < 0 and low_slope > 0 and 
                    high_r > 0.7 and low_r > 0.7):
                    
                    # Calculate the height of the pole
                    pole_height = first_third[self.price_col].iloc[-1] - first_third[self.price_col].iloc[0]
                    
                    result['detected'] = True
                    result['confidence'] = min(1.0, (high_r + low_r) / 2)
                    result['pole_height'] = pole_height
                    result['target'] = self.df[self.price_col].iloc[-1] + pole_height
        
        return result
    
    def detect_bearish_pennant(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Bearish Pennant pattern (bearish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 15 periods for a proper pennant
        if len(self.df) < 15:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # A bearish pennant consists of a sharp downward move (pole) followed by a symmetrical triangle (pennant)
        # First, identify if there was a strong downward move
        first_third = recent_data.iloc[:len(recent_data)//3]
        first_third_change = (first_third[self.price_col].iloc[-1] - first_third[self.price_col].iloc[0]) / first_third[self.price_col].iloc[0]
        
        # Check for a strong downward move (<-5% in the first third)
        if first_third_change < -0.05:
            # Now check for a symmetrical triangle in the latter part
            latter_part = recent_data.iloc[len(recent_data)//3:]
            
            # Get local maxima and minima in the latter part
            highs = latter_part[latter_part['local_max'].notna()].copy()
            lows = latter_part[latter_part['local_min'].notna()].copy()
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Fit trendlines to highs and lows
                high_slope, high_intercept, high_r = self._fit_trendline(highs)
                low_slope, low_intercept, low_r = self._fit_trendline(lows)
                
                # For a pennant, the triangle should be converging
                if (high_slope < 0 and low_slope > 0 and 
                    high_r > 0.7 and low_r > 0.7):
                    
                    # Calculate the height of the pole
                    pole_height = abs(first_third[self.price_col].iloc[0] - first_third[self.price_col].iloc[-1])
                    
                    result['detected'] = True
                    result['confidence'] = min(1.0, (high_r + low_r) / 2)
                    result['pole_height'] = pole_height
                    result['target'] = self.df[self.price_col].iloc[-1] - pole_height
        
        return result
    
    def detect_rectangle(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Rectangle pattern (continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'neutral', 'confidence': 0.0}
        
        # Need at least 15 periods for a proper rectangle
        if len(self.df) < 15:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # For a rectangle, we need roughly horizontal support and resistance levels
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # Check if both trendlines are roughly horizontal (slope near 0)
        if abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
            # Calculate the average levels
            resistance = highs[self.price_col].mean()
            support = lows[self.price_col].mean()
            
            # Check if the range is significant (at least 2%)
            if (resistance - support) / support > 0.02:
                # Determine the signal based on the current price position and trend
                current_price = self.df[self.price_col].iloc[-1]
                prev_price = self.df[self.price_col].iloc[-6]  # Look back 5 periods
                
                # If price is near resistance and moving up
                if abs(current_price - resistance) / resistance < 0.01 and current_price > prev_price:
                    result['signal'] = 'bullish'
                    result['target'] = resistance + (resistance - support)
                # If price is near support and moving down
                elif abs(current_price - support) / support < 0.01 and current_price < prev_price:
                    result['signal'] = 'bearish'
                    result['target'] = support - (resistance - support)
                
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
                result['resistance'] = resistance
                result['support'] = support
        
        return result
    
    def detect_cup_and_handle(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Cup and Handle pattern (bullish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 30 periods for a proper cup and handle
        if len(self.df) < 30:
            return result
        
        # Look at a longer period for cup and handle
        lookback = min(60, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Divide the data into potential cup (first 70%) and handle (last 30%)
        cup_end_idx = int(len(recent_data) * 0.7)
        cup_data = recent_data.iloc[:cup_end_idx]
        handle_data = recent_data.iloc[cup_end_idx:]
        
        # For a cup, we need a U-shaped pattern
        # First, check if the first and last points of the cup are at similar levels
        cup_start_price = cup_data[self.price_col].iloc[0]
        cup_end_price = cup_data[self.price_col].iloc[-1]
        
        if abs(cup_start_price - cup_end_price) / cup_start_price < 0.05:
            # Find the lowest point in the cup
            cup_low = cup_data[self.price_col].min()
            cup_low_idx = cup_data[self.price_col].idxmin()
            
            # Check if the low point is roughly in the middle of the cup
            cup_middle_idx = cup_data.index[len(cup_data) // 2]
            if abs((cup_data.index.get_loc(cup_low_idx) - len(cup_data) // 2)) < len(cup_data) // 4:
                # Now check for a small pullback in the handle
                if len(handle_data) >= 5:
                    handle_high = handle_data[self.price_col].max()
                    handle_low = handle_data[self.price_col].min()
                    
                    # Handle should be a small pullback (less than 50% of cup depth)
                    cup_depth = cup_start_price - cup_low
                    handle_depth = handle_high - handle_low
                    
                    if handle_depth < cup_depth * 0.5 and handle_high < cup_start_price * 1.05:
                        # Check if the current price is breaking above the handle
                        current_price = self.df[self.price_col].iloc[-1]
                        
                        if current_price > handle_high:
                            result['detected'] = True
                            result['confidence'] = 0.8
                            result['cup_depth'] = cup_depth
                            result['target'] = current_price + cup_depth
        
        return result
    
    def detect_inverted_cup_and_handle(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Inverted Cup and Handle pattern (bearish continuation).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 30 periods for a proper inverted cup and handle
        if len(self.df) < 30:
            return result
        
        # Look at a longer period for inverted cup and handle
        lookback = min(60, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Divide the data into potential cup (first 70%) and handle (last 30%)
        cup_end_idx = int(len(recent_data) * 0.7)
        cup_data = recent_data.iloc[:cup_end_idx]
        handle_data = recent_data.iloc[cup_end_idx:]
        
        # For an inverted cup, we need an inverted U-shaped pattern
        # First, check if the first and last points of the cup are at similar levels
        cup_start_price = cup_data[self.price_col].iloc[0]
        cup_end_price = cup_data[self.price_col].iloc[-1]
        
        if abs(cup_start_price - cup_end_price) / cup_start_price < 0.05:
            # Find the highest point in the cup
            cup_high = cup_data[self.price_col].max()
            cup_high_idx = cup_data[self.price_col].idxmax()
            
            # Check if the high point is roughly in the middle of the cup
            cup_middle_idx = cup_data.index[len(cup_data) // 2]
            if abs((cup_data.index.get_loc(cup_high_idx) - len(cup_data) // 2)) < len(cup_data) // 4:
                # Now check for a small pullback in the handle
                if len(handle_data) >= 5:
                    handle_low = handle_data[self.price_col].min()
                    handle_high = handle_data[self.price_col].max()
                    
                    # Handle should be a small pullback (less than 50% of cup height)
                    cup_height = cup_high - cup_start_price
                    handle_height = handle_high - handle_low
                    
                    if handle_height < cup_height * 0.5 and handle_low > cup_start_price * 0.95:
                        # Check if the current price is breaking below the handle
                        current_price = self.df[self.price_col].iloc[-1]
                        
                        if current_price < handle_low:
                            result['detected'] = True
                            result['confidence'] = 0.8
                            result['cup_height'] = cup_height
                            result['target'] = current_price - cup_height
        
        return result
    
    def detect_diamond_top(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Diamond Top pattern (bearish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 20 periods for a proper diamond
        if len(self.df) < 20:
            return result
        
        # Look at a longer period for diamond pattern
        lookback = min(40, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Divide the data into first half (broadening) and second half (narrowing)
        mid_idx = len(recent_data) // 2
        first_half = recent_data.iloc[:mid_idx]
        second_half = recent_data.iloc[mid_idx:]
        
        # Get local maxima and minima for both halves
        first_highs = first_half[first_half['local_max'].notna()].copy()
        first_lows = first_half[first_half['local_min'].notna()].copy()
        second_highs = second_half[second_half['local_max'].notna()].copy()
        second_lows = second_half[second_half['local_min'].notna()].copy()
        
        if (len(first_highs) >= 2 and len(first_lows) >= 2 and 
            len(second_highs) >= 2 and len(second_lows) >= 2):
            
            # Fit trendlines to all segments
            first_high_slope, _, first_high_r = self._fit_trendline(first_highs)
            first_low_slope, _, first_low_r = self._fit_trendline(first_lows)
            second_high_slope, _, second_high_r = self._fit_trendline(second_highs)
            second_low_slope, _, second_low_r = self._fit_trendline(second_lows)
            
            # For a diamond top:
            # 1. First half should be broadening (high slope up, low slope down)
            # 2. Second half should be narrowing (high slope down, low slope up)
            if (first_high_slope > 0 and first_low_slope < 0 and 
                second_high_slope < 0 and second_low_slope > 0 and
                first_high_r > 0.6 and first_low_r > 0.6 and 
                second_high_r > 0.6 and second_low_r > 0.6):
                
                # Check if the current price is breaking below the lower trendline
                current_price = self.df[self.price_col].iloc[-1]
                
                # Calculate the height of the diamond
                diamond_height = first_highs[self.price_col].max() - first_lows[self.price_col].min()
                
                # Calculate the lower trendline at the current point
                x_pos = len(recent_data) - 1
                lower_trendline = second_low_slope * (x_pos - mid_idx) + second_lows[self.price_col].iloc[0]
                
                if current_price < lower_trendline:
                    result['detected'] = True
                    result['confidence'] = min(1.0, (first_high_r + first_low_r + second_high_r + second_low_r) / 4)
                    result['diamond_height'] = diamond_height
                    result['target'] = current_price - diamond_height
        
        return result
    
    def detect_diamond_bottom(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Diamond Bottom pattern (bullish reversal).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bullish', 'confidence': 0.0}
        
        # Need at least 20 periods for a proper diamond
        if len(self.df) < 20:
            return result
        
        # Look at a longer period for diamond pattern
        lookback = min(40, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Divide the data into first half (broadening) and second half (narrowing)
        mid_idx = len(recent_data) // 2
        first_half = recent_data.iloc[:mid_idx]
        second_half = recent_data.iloc[mid_idx:]
        
        # Get local maxima and minima for both halves
        first_highs = first_half[first_half['local_max'].notna()].copy()
        first_lows = first_half[first_half['local_min'].notna()].copy()
        second_highs = second_half[second_half['local_max'].notna()].copy()
        second_lows = second_half[second_half['local_min'].notna()].copy()
        
        if (len(first_highs) >= 2 and len(first_lows) >= 2 and 
            len(second_highs) >= 2 and len(second_lows) >= 2):
            
            # Fit trendlines to all segments
            first_high_slope, _, first_high_r = self._fit_trendline(first_highs)
            first_low_slope, _, first_low_r = self._fit_trendline(first_lows)
            second_high_slope, _, second_high_r = self._fit_trendline(second_highs)
            second_low_slope, _, second_low_r = self._fit_trendline(second_lows)
            
            # For a diamond bottom:
            # 1. First half should be broadening (high slope down, low slope up)
            # 2. Second half should be narrowing (high slope up, low slope down)
            if (first_high_slope < 0 and first_low_slope > 0 and 
                second_high_slope > 0 and second_low_slope < 0 and
                first_high_r > 0.6 and first_low_r > 0.6 and 
                second_high_r > 0.6 and second_low_r > 0.6):
                
                # Check if the current price is breaking above the upper trendline
                current_price = self.df[self.price_col].iloc[-1]
                
                # Calculate the height of the diamond
                diamond_height = first_highs[self.price_col].max() - first_lows[self.price_col].min()
                
                # Calculate the upper trendline at the current point
                x_pos = len(recent_data) - 1
                upper_trendline = second_high_slope * (x_pos - mid_idx) + second_highs[self.price_col].iloc[0]
                
                if current_price > upper_trendline:
                    result['detected'] = True
                    result['confidence'] = min(1.0, (first_high_r + first_low_r + second_high_r + second_low_r) / 4)
                    result['diamond_height'] = diamond_height
                    result['target'] = current_price + diamond_height
        
        return result
    
    def detect_megaphone(self) -> Dict[str, Union[bool, float, str]]:
        """
        Detect Megaphone pattern (broadening formation, volatile and potentially bearish).
        
        Returns:
            Dictionary with detection results
        """
        result = {'detected': False, 'signal': 'bearish', 'confidence': 0.0}
        
        # Need at least 15 periods for a proper megaphone
        if len(self.df) < 15:
            return result
        
        # Look at the last 20-30 periods
        lookback = min(30, len(self.df) - 1)
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Get local maxima and minima
        highs = recent_data[recent_data['local_max'].notna()].copy()
        lows = recent_data[recent_data['local_min'].notna()].copy()
        
        if len(highs) < 3 or len(lows) < 3:
            return result
        
        # Fit trendlines to highs and lows
        high_slope, high_intercept, high_r = self._fit_trendline(highs)
        low_slope, low_intercept, low_r = self._fit_trendline(lows)
        
        # For a megaphone:
        # 1. Upper trendline should be rising (positive slope)
        # 2. Lower trendline should be falling (negative slope)
        # 3. Both trendlines should have good fit
        if (high_slope > 0 and low_slope < 0 and 
            high_r > 0.7 and low_r > 0.7):
            
            # Calculate the current values of the trendlines
            last_x = len(recent_data) - 1
            upper_trendline = high_slope * last_x + high_intercept
            lower_trendline = low_slope * last_x + low_intercept
            
            # Check if the current price is near either trendline
            current_price = self.df[self.price_col].iloc[-1]
            
            # If price is near the upper trendline (potential reversal down)
            if abs(current_price - upper_trendline) / upper_trendline < 0.02:
                result['signal'] = 'bearish'
                result['target'] = lower_trendline
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
            
            # If price is near the lower trendline (potential reversal up)
            elif abs(current_price - lower_trendline) / lower_trendline < 0.02:
                result['signal'] = 'bullish'
                result['target'] = upper_trendline
                result['detected'] = True
                result['confidence'] = min(1.0, (high_r + low_r) / 2)
            
            # Store trendline values
            result['upper_trendline'] = upper_trendline
            result['lower_trendline'] = lower_trendline
        
        return result
    
    def detect_candlestick_patterns(self) -> Dict[str, Dict[str, Union[bool, float, str]]]:
        """
        Detect candlestick patterns using ta library instead of TA-Lib.
        
        Returns:
            Dictionary with pattern names as keys and detection results as values
        """
        results = {}
        
        try:
            # Make sure we have the required OHLC data
            if len(self.df) < 2:
                return results
                
            # Create a copy of the dataframe to avoid modifying the original
            df = self.df.copy()
            
            # Calculate some pattern indicators using ta library
            # Note: ta doesn't have direct candlestick pattern recognition like TA-Lib
            # We'll use momentum indicators and other signals as proxies
            
            # Add momentum indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
            df['stoch_signal'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            
            # Add volatility indicators
            df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            
            # Current candle and previous candle
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            
            if previous is not None:
                # Bullish engulfing
                if (previous['open'] > previous['close'] and  # Previous candle is bearish
                    current['open'] < current['close'] and    # Current candle is bullish
                    current['open'] <= previous['close'] and  # Current open is below or equal to previous close
                    current['close'] > previous['open']):     # Current close is above previous open
                    results['engulfing_bullish'] = {
                        'detected': True,
                        'signal': 'bullish',
                        'confidence': 0.8,
                        'candle_idx': len(df) - 1
                    }
                
                # Bearish engulfing
                if (previous['open'] < previous['close'] and  # Previous candle is bullish
                    current['open'] > current['close'] and    # Current candle is bearish
                    current['open'] >= previous['close'] and  # Current open is above or equal to previous close
                    current['close'] < previous['open']):     # Current close is below previous open
                    results['engulfing_bearish'] = {
                        'detected': True,
                        'signal': 'bearish',
                        'confidence': 0.8,
                        'candle_idx': len(df) - 1
                    }
                
                # Hammer (bullish)
                body_size = abs(current['open'] - current['close'])
                lower_wick = min(current['open'], current['close']) - current['low']
                upper_wick = current['high'] - max(current['open'], current['close'])
                
                if (lower_wick > body_size * 2 and  # Lower wick is at least 2x the body
                    upper_wick < body_size * 0.5 and  # Upper wick is small
                    current['close'] > current['open']):  # Bullish candle
                    results['hammer'] = {
                        'detected': True,
                        'signal': 'bullish',
                        'confidence': 0.7,
                        'candle_idx': len(df) - 1
                    }
                
                # Shooting star (bearish)
                if (upper_wick > body_size * 2 and  # Upper wick is at least 2x the body
                    lower_wick < body_size * 0.5 and  # Lower wick is small
                    current['close'] < current['open']):  # Bearish candle
                    results['shooting_star'] = {
                        'detected': True,
                        'signal': 'bearish',
                        'confidence': 0.7,
                        'candle_idx': len(df) - 1
                    }
                
                # Doji
                if abs(current['open'] - current['close']) < (current['high'] - current['low']) * 0.1:
                    if current['high'] - max(current['open'], current['close']) > 2 * body_size:
                        # Gravestone doji (bearish)
                        results['gravestone_doji'] = {
                            'detected': True,
                            'signal': 'bearish',
                            'confidence': 0.6,
                            'candle_idx': len(df) - 1
                        }
                    elif min(current['open'], current['close']) - current['low'] > 2 * body_size:
                        # Dragonfly doji (bullish)
                        results['dragonfly_doji'] = {
                            'detected': True,
                            'signal': 'bullish',
                            'confidence': 0.6,
                            'candle_idx': len(df) - 1
                        }
            
            # Check for momentum-based signals
            if current['rsi'] < 30:
                results['oversold'] = {
                    'detected': True,
                    'signal': 'bullish',
                    'confidence': 0.6,
                    'candle_idx': len(df) - 1
                }
            
            if current['rsi'] > 70:
                results['overbought'] = {
                    'detected': True,
                    'signal': 'bearish',
                    'confidence': 0.6,
                    'candle_idx': len(df) - 1
                }
            
            # MACD crossover (bullish)
            if (previous is not None and
                previous['macd'] < previous['macd_signal'] and
                current['macd'] > current['macd_signal']):
                results['macd_bullish_crossover'] = {
                    'detected': True,
                    'signal': 'bullish',
                    'confidence': 0.7,
                    'candle_idx': len(df) - 1
                }
            
            # MACD crossover (bearish)
            if (previous is not None and
                previous['macd'] > previous['macd_signal'] and
                current['macd'] < current['macd_signal']):
                results['macd_bearish_crossover'] = {
                    'detected': True,
                    'signal': 'bearish',
                    'confidence': 0.7,
                    'candle_idx': len(df) - 1
                }
            
            # Stochastic crossover (bullish)
            if (previous is not None and
                previous['stoch'] < previous['stoch_signal'] and
                current['stoch'] > current['stoch_signal'] and
                current['stoch'] < 20):
                results['stoch_bullish_crossover'] = {
                    'detected': True,
                    'signal': 'bullish',
                    'confidence': 0.7,
                    'candle_idx': len(df) - 1
                }
            
            # Stochastic crossover (bearish)
            if (previous is not None and
                previous['stoch'] > previous['stoch_signal'] and
                current['stoch'] < current['stoch_signal'] and
                current['stoch'] > 80):
                results['stoch_bearish_crossover'] = {
                    'detected': True,
                    'signal': 'bearish',
                    'confidence': 0.7,
                    'candle_idx': len(df) - 1
                }
            
            # Bollinger Band signals
            if current['close'] < current['bb_low']:
                results['bb_oversold'] = {
                    'detected': True,
                    'signal': 'bullish',
                    'confidence': 0.6,
                    'candle_idx': len(df) - 1
                }
            
            if current['close'] > current['bb_high']:
                results['bb_overbought'] = {
                    'detected': True,
                    'signal': 'bearish',
                    'confidence': 0.6,
                    'candle_idx': len(df) - 1
                }
        
        except Exception as e:
            # If there's an error, just return empty results
            print(f"Error detecting candlestick patterns: {e}")
        
        return results


