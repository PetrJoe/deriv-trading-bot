import pandas as pd
import numpy as np
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple
from dotenv import load_dotenv

from pattern_recognition import PatternRecognizer
from chart_utils import plot_enhanced_chart
from bot import fetch_deriv_candles

# Load environment variables
load_dotenv()

# Configuration
APP_ID = os.getenv('APP_ID', 'YOUR_APP_ID')
DEFAULT_SYMBOL = os.getenv('SYMBOL', 'R_75')
DEFAULT_GRANULARITY = int(os.getenv('GRANULARITY', 60))
DEFAULT_CANDLE_COUNT = int(os.getenv('CANDLE_COUNT', 100))

def analyze_deriv_asset(symbol: str = DEFAULT_SYMBOL, 
                        granularity: int = DEFAULT_GRANULARITY, 
                        candle_count: int = DEFAULT_CANDLE_COUNT) -> Dict:
    """
    Analyze a Deriv asset for technical patterns.
    
    Args:
        symbol: Deriv asset symbol (e.g., 'R_75')
        granularity: Candle granularity in seconds
        candle_count: Number of candles to fetch
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Fetch data from Deriv API
        df = asyncio.run(fetch_deriv_candles(symbol, granularity, candle_count))
        
        if df.empty:
            return {
                'symbol': symbol,
                'success': False,
                'error': 'Failed to fetch data'
            }
        
        # Calculate some basic indicators
        df.set_index('epoch', inplace=True)
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Initialize pattern recognizer
        recognizer = PatternRecognizer(df)
        
        # Detect patterns
        patterns = recognizer.detect_all_patterns()
        
        # Generate chart
        chart_path = f"static/{symbol}_chart.png"
        plot_enhanced_chart(df, patterns, 
                           title=f"{symbol} Technical Analysis", 
                           save_path=chart_path)
        
        # Prepare results
        detected_patterns = {}
        for pattern_name, pattern_data in patterns.items():
            if pattern_data.get('detected', False):
                detected_patterns[pattern_name] = pattern_data
        
        # Calculate overall signal
        bullish_score = 0
        bearish_score = 0
        
        for pattern_name, pattern_data in detected_patterns.items():
            signal = pattern_data.get('signal', 'neutral')
            confidence = pattern_data.get('confidence', 0.5)
            
            if signal == 'bullish':
                bullish_score += confidence
            elif signal == 'bearish':
                bearish_score += confidence
        
        # Determine overall recommendation
        if bullish_score > bearish_score:
            recommendation = 'BUY'
            signal_strength = min(1.0, bullish_score / 3)  # Normalize to 0-1
        elif bearish_score > bullish_score:
            recommendation = 'SELL'
            signal_strength = min(1.0, bearish_score / 3)  # Normalize to 0-1
        else:
            recommendation = 'HOLD'
            signal_strength = 0.5
        
        # Get current price and basic stats
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else None
        change = ((current_price - prev_close) / prev_close * 100) if prev_close else None
        
        # Prepare the final result
        result = {
            'symbol': symbol,
            'success': True,
            'current_price': current_price,
            'change_percent': change,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'granularity': granularity,
            'candle_count': candle_count,
            'recommendation': recommendation,
            'signal_strength': signal_strength,
            'detected_patterns': detected_patterns,
            'chart_path': chart_path
        }
        
        return result
    
    except Exception as e:
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }

def analyze_multiple_assets(symbols: List[str], granularity: int = DEFAULT_GRANULARITY, 
                           candle_count: int = DEFAULT_CANDLE_COUNT) -> Dict[str, Dict]:
    """
    Analyze multiple Deriv assets for technical patterns.
    
    Args:
        symbols: List of Deriv asset symbols
        granularity: Candle granularity in seconds
        candle_count: Number of candles to fetch
    
    Returns:
        Dictionary with analysis results for each symbol
    """
    results = {}
    
    for symbol in symbols:
        print(f"Analyzing {symbol}...")
        result = analyze_deriv_asset(symbol, granularity, candle_count)
        results[symbol] = result
        
        # Print a summary
        if result['success']:
            detected_count = len(result.get('detected_patterns', {}))
            print(f"  Current price: {result['current_price']}")
            if result.get('change_percent'):
                print(f"  Change: {result['change_percent']:.2f}%")
            print(f"  Detected {detected_count} patterns")
            print(f"  Recommendation: {result['recommendation']} (Strength: {result['signal_strength']:.2f})")
            print(f"  Chart saved to: {result['chart_path']}")
        else:
            print(f"  Analysis failed: {result.get('error', 'Unknown error')}")
        
        print()
    
    # Save results to JSON
    os.makedirs('static', exist_ok=True)
    with open('static/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to static/analysis_results.json")
    
    return results

def main():
    """
    Main function to run the Deriv pattern analyzer.
    """
    # Get symbols from environment or use default
    symbols_str = os.getenv('SYMBOLS', DEFAULT_SYMBOL)
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    granularity = int(os.getenv('GRANULARITY', DEFAULT_GRANULARITY))
    candle_count = int(os.getenv('CANDLE_COUNT', DEFAULT_CANDLE_COUNT))
    
    print(f"Analyzing {len(symbols)} Deriv assets: {', '.join(symbols)}")
    print(f"Granularity: {granularity} seconds, Candle count: {candle_count}")
    
    results = analyze_multiple_assets(symbols, granularity, candle_count)
    
    # Print overall summary
    successful = sum(1 for r in results.values() if r.get('success', False))
    print(f"\nSummary: Successfully analyzed {successful}/{len(symbols)} assets")
    
    # Print buy/sell recommendations
    buys = [s for s, r in results.items() if r.get('success') and r.get('recommendation') == 'BUY']
    sells = [s for s, r in results.items() if r.get('success') and r.get('recommendation') == 'SELL']
    holds = [s for s, r in results.items() if r.get('success') and r.get('recommendation') == 'HOLD']
    
    if buys:
        print(f"BUY recommendations: {', '.join(buys)}")
    if sells:
        print(f"SELL recommendations: {', '.join(sells)}")
    if holds:
        print(f"HOLD recommendations: {', '.join(holds)}")

if __name__ == "__main__":
    main()
