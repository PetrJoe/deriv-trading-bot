import os
import logging
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from deriv_api import DerivAPI  # type: ignore  # Ensure your DerivAPI package is installed and up 
from dotenv import load_dotenv
import time

# Load environment variables from a .env file (if available)
load_dotenv()

# --- CONFIGURATION AND LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("trading_bot_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- APPLICATION CONFIGURATION VARIABLES ---
APP_ID = os.getenv('APP_ID', 'YOUR_APP_ID')
SYMBOL = os.getenv('SYMBOL', 'R_75')
GRANULARITY = int(os.getenv('GRANULARITY', 60))
CANDLE_COUNT = int(os.getenv('CANDLE_COUNT', 100))
CSV_FILE = os.getenv('CSV_FILE', 'data.csv')
CHART_FILE = os.path.join('static', os.getenv('CHART_FILE', 'chart.png'))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 300))

# --- UTILITY FUNCTIONS ---

async def fetch_deriv_candles(symbol=SYMBOL, granularity=GRANULARITY, count=CANDLE_COUNT, csv_file=CSV_FILE):
    """Asynchronously fetch candlestick data from Deriv API, save to CSV, and return DataFrame."""
    logger.info(f"Starting data fetch from Deriv API for {symbol}.")
    api = DerivAPI(app_id=APP_ID)
    candles = []
    try:
        # Wait until API is connected
        while not api.connected:
            await asyncio.sleep(0.1)
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles"
        }
        response = await api.send(request)
        candles = response.get("candles", [])
        logger.info(f"Fetched {len(candles)} candles for {symbol}.")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}", exc_info=True)
    
    if candles:
        try:
            df = pd.DataFrame(candles)
            df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
            df.to_csv(csv_file, index=False)
            logger.info(f"Data saved to {csv_file}.")
            return df
        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning(f"No candle data received for {symbol}.")
        return pd.DataFrame()

def analyze_data(df):
    """Analyze DataFrame: compute SMAs, generate a simple trading signal, and return recommendation."""
    try:
        df = df.copy()
        df.set_index('epoch', inplace=True)
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['signal'] = 0
        df.loc[df['SMA_5'] > df['SMA_10'], 'signal'] = 1
        df.loc[df['SMA_5'] < df['SMA_10'], 'signal'] = -1
        df['position'] = df['signal'].diff()
        latest_signal = df['signal'].iloc[-1]
        recommendation = "Buy" if latest_signal == 1 else "Sell" if latest_signal == -1 else "Hold"
        logger.info(f"Analysis complete. Latest recommendation: {recommendation}.")
        return df, recommendation
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
        return df, "Hold"

def plot_chart(df, symbol=SYMBOL, chart_file=CHART_FILE):
    """Plot candlestick chart with SMAs and save to static folder."""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['close'], label='Close Price', color='blue')
        plt.plot(df.index, df['SMA_5'], label='SMA 5', color='orange')
        plt.plot(df.index, df['SMA_10'], label='SMA 10', color='green')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{symbol} - Chart ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
        plt.legend()
        os.makedirs('static', exist_ok=True)
        plt.savefig(chart_file)
        plt.close()
        logger.info(f"Chart saved to {chart_file}.")
    except Exception as e:
        logger.error(f"Error plotting chart: {str(e)}", exc_info=True)


# --- MAIN FUNCTION FOR DATA FETCH AND ANALYSIS ---
def fetch_and_analyze(symbol=SYMBOL, granularity=GRANULARITY, count=CANDLE_COUNT):
    """Function to fetch data, analyze it, and log the trading signal."""
    try:
        # Use asyncio.run to handle the async function call
        df = asyncio.run(fetch_deriv_candles(symbol, granularity, count))
        if df.empty:
            logger.warning(f"No data to process for {symbol}.")
            return
        analyzed_df, recommendation = analyze_data(df)
        plot_chart(analyzed_df, symbol)
        
        # Log the trading signal with additional information
        current_price = analyzed_df['close'].iloc[-1] if not analyzed_df.empty else "Unknown"
        logger.info(f"TRADING SIGNAL: {recommendation} | Symbol: {symbol} | Current Price: {current_price} | Time: {datetime.now().isoformat()}")
        
        return analyzed_df, recommendation
        
    except Exception as e:
        logger.error(f"Error in fetch_and_analyze for {symbol}: {str(e)}", exc_info=True)
        return None, None


def run_bot(symbols=None, granularity=GRANULARITY, count=CANDLE_COUNT, update_interval=UPDATE_INTERVAL):
    """Main function to run the trading bot continuously for multiple symbols."""
    if symbols is None:
        symbols = [SYMBOL]
    elif isinstance(symbols, str):
        symbols = [symbols]
        
    logger.info(f"Starting Trading Bot for symbols: {', '.join(symbols)}...")
    
    running = True
    while running:
        try:
            for symbol in symbols:
                logger.info(f"Analyzing {symbol}...")
                df, recommendation = fetch_and_analyze(symbol, granularity, count)
                
                if df is not None:
                    logger.info(f"Analysis complete for {symbol}. Recommendation: {recommendation}")
            
            logger.info(f"Waiting {update_interval} seconds until next update...")
            
            # Break the sleep into smaller chunks to handle interrupts more gracefully
            for _ in range(update_interval):
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down gracefully...")
            running = False
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
            # Sleep a bit before retrying to avoid tight error loops
            time.sleep(5)
    
    logger.info("Trading bot has been stopped.")


# --- ENTRY POINT ---
if __name__ == '__main__':
    try:
        # You can specify multiple symbols to monitor
        symbols_to_monitor = os.getenv('SYMBOLS', SYMBOL).split(',')
        run_bot(symbols=symbols_to_monitor)
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading bot shutdown complete.")
