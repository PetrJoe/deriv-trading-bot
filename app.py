from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import asyncio
from datetime import datetime
from deriv_api import DerivAPI
from dotenv import load_dotenv
from pattern_recognition import PatternRecognizer
from chart_utils import plot_enhanced_chart
from bot import fetch_deriv_candles, analyze_data

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form.get('symbol', os.getenv('SYMBOL', 'R_75'))
    granularity = int(request.form.get('granularity', os.getenv('GRANULARITY', 60)))
    candle_count = int(request.form.get('candle_count', os.getenv('CANDLE_COUNT', 100)))
    
    # Run analysis
    result = analyze_deriv_asset(symbol, granularity, candle_count)
    
    # Save result to a file
    os.makedirs('static', exist_ok=True)
    with open(f'static/{symbol}_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return redirect(url_for('results', symbol=symbol))

@app.route('/results/<symbol>')
def results(symbol):
    # Load analysis results
    try:
        with open(f'static/{symbol}_analysis.json', 'r') as f:
            result = json.load(f)
    except FileNotFoundError:
        return render_template('error.html', message=f"No analysis found for {symbol}")
    
    return render_template('results.html', result=result, symbol=symbol)

@app.route('/api/analyze/<symbol>')
def api_analyze(symbol):
    granularity = int(request.args.get('granularity', os.getenv('GRANULARITY', 60)))
    candle_count = int(request.args.get('candle_count', os.getenv('CANDLE_COUNT', 100)))
    
    # Run analysis
    result = analyze_deriv_asset(symbol, granularity, candle_count)
    
    return jsonify(result)

def analyze_deriv_asset(symbol, granularity, candle_count):
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

if __name__ == '__main__':
    app.run(debug=True)
