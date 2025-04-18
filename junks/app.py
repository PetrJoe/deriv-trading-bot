from flask import Flask, render_template, request, jsonify, redirect, url_for  #type:ignore
import os
import json
import asyncio
from datetime import datetime
from deriv_api import DerivAPI #type:ignore
from dotenv import load_dotenv
from pattern_recognition import PatternRecognition  # Update class name to match
from chart_utils import plot_enhanced_chart, plot_deriv_chart  # Add plot_deriv_chart import
from bot import fetch_deriv_candles, analyze_data



load_dotenv()

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

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
        
        # Ensure required fields exist to prevent template errors
        if 'current_price' not in result:
            result['current_price'] = 0.0
        if 'change_percent' not in result:
            result['change_percent'] = 0.0
        if 'analysis_date' not in result:
            result['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if 'granularity' not in result:
            result['granularity'] = 60
        if 'candle_count' not in result:
            result['candle_count'] = 100
        if 'recommendation' not in result:
            result['recommendation'] = 'HOLD'
        if 'signal_strength' not in result:
            result['signal_strength'] = 0.5
        if 'detected_patterns' not in result:
            result['detected_patterns'] = {}
        if 'chart_path' not in result:
            result['chart_path'] = f"static/{symbol}_chart.png"
            
        return render_template('results.html', result=result, symbol=symbol)
    except FileNotFoundError:
        return render_template('error.html', message=f"No analysis found for {symbol}")


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
        recognizer = PatternRecognition(df)
        
        # Detect patterns
        patterns = recognizer.detect_all_patterns()
        
        # Generate chart - FIXED PATH
        os.makedirs('static', exist_ok=True)  # Ensure directory exists
        chart_filename = f"{symbol}_chart.png"
        chart_path = os.path.join("static", chart_filename)
        
        # Plot chart with absolute path for saving
        plot_enhanced_chart(df, patterns, 
                           title=f"{symbol} Technical Analysis", 
                           save_path=chart_path)
        
        # Store relative path for display in HTML
        display_chart_path = chart_path
        
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
            'chart_path': display_chart_path
        }
        
        return result
    
    except Exception as e:
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }

@app.route('/test_chart/<symbol>')

@app.route('/test_chart/<symbol>')
def test_chart(symbol):
    """Test route to verify chart generation and display"""
    try:
        # Create a simple test chart
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample data
        dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        data = {
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Calculate some basic indicators for plot_enhanced_chart
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        
        # Add local extrema for pattern detection
        from scipy.signal import argrelextrema
        df['local_max'] = df.iloc[argrelextrema(df['close'].values, np.greater_equal, order=5)[0]]['close']
        df['local_min'] = df.iloc[argrelextrema(df['close'].values, np.less_equal, order=5)[0]]['close']
        
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        # Save a test chart using plot_enhanced_chart instead
        chart_path = f"static/{symbol}_test_chart.png"
        plot_enhanced_chart(df, {}, title=f"{symbol} Test Chart", save_path=chart_path)
        
        return f"""
        <html>
            <head><title>Chart Test</title></head>
            <body>
                <h1>Test Chart for {symbol}</h1>
                <img src="/{chart_path}" alt="Test Chart">
                <p>Chart path: {chart_path}</p>
                <p>Current time: {datetime.now()}</p>
            </body>
        </html>
        """
    except Exception as e:
        return f"Error generating test chart: {str(e)}"



if __name__ == '__main__':
    app.run(debug=True)
