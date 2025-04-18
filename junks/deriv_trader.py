import os
import logging
import asyncio
import pandas as pd
import json
from datetime import datetime
import time
from typing import Dict, List, Union, Optional, Tuple
from dotenv import load_dotenv

from bot import fetch_deriv_candles, analyze_data, plot_chart
from deriv.pattern_recognition import PatternRecognizer
from chart_utils import plot_enhanced_chart, plot_deriv_chart

# Load environment variables
load_dotenv()

# --- CONFIGURATION AND LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("deriv_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- APPLICATION CONFIGURATION VARIABLES ---
APP_ID = os.getenv('APP_ID', 'YOUR_APP_ID')
DEFAULT_SYMBOL = os.getenv('SYMBOL', 'R_75')
DEFAULT_GRANULARITY = int(os.getenv('GRANULARITY', 60))
DEFAULT_CANDLE_COUNT = int(os.getenv('CANDLE_COUNT', 100))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 300))

class DerivTrader:
    def __init__(self, 
                 symbols: Union[str, List[str]] = DEFAULT_SYMBOL,
                 granularity: int = DEFAULT_GRANULARITY,
                 candle_count: int = DEFAULT_CANDLE_COUNT,
                 update_interval: int = UPDATE_INTERVAL):
        """
        Initialize the Deriv Trader.
        
        Args:
            symbols: Symbol or list of symbols to analyze
            granularity: Candle granularity in seconds
            candle_count: Number of candles to fetch
            update_interval: Update interval in seconds
        """
        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols
            
        self.granularity = granularity
        self.candle_count = candle_count
        self.update_interval = update_interval
        self.results = {}
        
        logger.info(f"DerivTrader initialized with symbols: {', '.join(self.symbols)}")
        logger.info(f"Granularity: {granularity}s, Candle count: {candle_count}, Update interval: {update_interval}s")
    
    async def fetch_and_analyze_symbol(self, symbol: str) -> Dict:
        """
        Fetch and analyze data for a single symbol.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Fetch data
            df = await fetch_deriv_candles(symbol, self.granularity, self.candle_count)
            
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': 'No data received'
                }
            
            # Set index to epoch for analysis
            if 'epoch' in df.columns:
                df.set_index('epoch', inplace=True)
            
            # Calculate indicators
            df['SMA_5'] = df['close'].rolling(window=5).mean()
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # Simple analysis
            analyzed_df, simple_recommendation = analyze_data(df.copy())
            
            # Advanced pattern recognition
            recognizer = PatternRecognizer(df)
            patterns = recognizer.detect_all_patterns()
            recommendation, pattern_signals = recognizer.get_trading_signal()
            
            # Generate charts
            simple_chart_path = f"static/{symbol}_simple_chart.png"
            advanced_chart_path = f"static/{symbol}_advanced_chart.png"
            
            # Simple chart
            plot_deriv_chart(df, symbol, simple_chart_path)
            
            # Advanced chart with patterns
            plot_enhanced_chart(df, patterns, 
                               title=f"{symbol} Technical Analysis", 
                               save_path=advanced_chart_path)
            
            # Prepare detected patterns
            detected_patterns = {}
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.get('detected', False):
                    detected_patterns[pattern_name] = pattern_data
            
            # Calculate signal strength
            bullish_score = 0
            bearish_score = 0
            
            for pattern_name, pattern_data in detected_patterns.items():
                signal = pattern_data.get('signal', 'neutral')
                confidence = pattern_data.get('confidence', 0.5)
                
                if signal == 'bullish':
                    bullish_score += confidence
                elif signal == 'bearish':
                    bearish_score += confidence
            
            # Determine final recommendation
            if bullish_score > bearish_score:
                final_recommendation = 'BUY'
                signal_strength = min(1.0, bullish_score / 3)
            elif bearish_score > bullish_score:
                final_recommendation = 'SELL'
                signal_strength = min(1.0, bearish_score / 3)
            else:
                final_recommendation = 'HOLD'
                signal_strength = 0.5
            
            # Get current price and stats
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else None
            change = ((current_price - prev_close) / prev_close * 100) if prev_close else None
            
            # Prepare result
            result = {
                'symbol': symbol,
                'success': True,
                'current_price': current_price,
                'change_percent': change,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'granularity': self.granularity,
                'candle_count': self.candle_count,
                'simple_recommendation': simple_recommendation,
                'advanced_recommendation': final_recommendation,
                'signal_strength': signal_strength,
                'detected_patterns': detected_patterns,
                'simple_chart_path': simple_chart_path,
                'advanced_chart_path': advanced_chart_path
            }
            
            logger.info(f"Analysis complete for {symbol}. Recommendation: {final_recommendation}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }
    
    async def analyze_all_symbols(self) -> Dict[str, Dict]:
        """
        Analyze all configured symbols.
        
        Returns:
            Dictionary with analysis results for each symbol
        """
        results = {}
        
        for symbol in self.symbols:
            logger.info(f"Analyzing {symbol}...")
            result = await self.fetch_and_analyze_symbol(symbol)
            results[symbol] = result
            
            # Print a summary
            if result['success']:
                detected_count = len(result.get('detected_patterns', {}))
                logger.info(f"  Current price: {result['current_price']}")
                if result.get('change_percent'):
                    logger.info(f"  Change: {result['change_percent']:.2f}%")
                logger.info(f"  Detected {detected_count} patterns")
                logger.info(f"  Recommendation: {result['advanced_recommendation']} (Strength: {result['signal_strength']:.2f})")
            else:
                logger.warning(f"  Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Save results to JSON
        os.makedirs('static', exist_ok=True)
        with open('static/deriv_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to static/deriv_analysis_results.json")
        
        self.results = results
        return results
    
    async def run_once(self) -> Dict[str, Dict]:
        """
        Run analysis once for all symbols.
        
        Returns:
            Analysis results
        """
        return await self.analyze_all_symbols()
    
    async def run_continuously(self):
        """
        Run analysis continuously for all symbols at the specified interval.
        """
        logger.info(f"Starting continuous analysis for {len(self.symbols)} symbols...")
        
        try:
            while True:
                start_time = time.time()
                
                # Run analysis
                await self.analyze_all_symbols()
                
                # Calculate time taken and sleep for the remainder of the interval
                elapsed = time.time() - start_time
                sleep_time = max(1, self.update_interval - elapsed)
                
                logger.info(f"Analysis took {elapsed:.2f} seconds. Waiting {sleep_time:.2f} seconds until next update...")
                
                # Sleep in small chunks to allow for clean interruption
                for _ in range(int(sleep_time)):
                    await asyncio.sleep(1)
                
                # Sleep any remaining fractional second
                await asyncio.sleep(sleep_time - int(sleep_time))
                
        except asyncio.CancelledError:
            logger.info("Analysis loop cancelled.")
        except Exception as e:
            logger.error(f"Error in continuous analysis: {str(e)}", exc_info=True)
    
    def get_latest_results(self) -> Dict[str, Dict]:
        """
        Get the latest analysis results.
        
        Returns:
            Latest analysis results
        """
        return self.results
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the latest analysis results.
        
        Returns:
            Summary of analysis results
        """
        if not self.results:
            return {
                'status': 'No analysis results available',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        buys = [s for s, r in self.results.items() if r.get('success') and r.get('advanced_recommendation') == 'BUY']
        sells = [s for s, r in self.results.items() if r.get('success') and r.get('advanced_recommendation') == 'SELL']
        holds = [s for s, r in self.results.items() if r.get('success') and r.get('advanced_recommendation') == 'HOLD']
        failed = [s for s, r in self.results.items() if not r.get('success')]
        
        return {
            'status': 'Analysis complete',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_symbols': len(self.symbols),
            'successful_analyses': len(self.symbols) - len(failed),
            'failed_analyses': len(failed),
            'buy_recommendations': buys,
            'sell_recommendations': sells,
            'hold_recommendations': holds,
            'failed_symbols': failed
        }

def run_trader():
    """
    Run the Deriv Trader from the command line.
    """
    # Get symbols from environment or use default
    symbols_str = os.getenv('SYMBOLS', DEFAULT_SYMBOL)
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    granularity = int(os.getenv('GRANULARITY', DEFAULT_GRANULARITY))
    candle_count = int(os.getenv('CANDLE_COUNT', DEFAULT_CANDLE_COUNT))
    update_interval = int(os.getenv('UPDATE_INTERVAL', UPDATE_INTERVAL))
    
    # Create trader
    trader = DerivTrader(symbols, granularity, candle_count, update_interval)
    
    # Run continuously
    try:
        asyncio.run(trader.run_continuously())
    except KeyboardInterrupt:
        logger.info("Trader stopped by user.")
    except Exception as e:
        logger.error(f"Trader stopped due to error: {str(e)}", exc_info=True)
    finally:
        logger.info("Trader shutdown complete.")

if __name__ == "__main__":
    run_trader()
