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

async def fetch_deriv_candles():
    """Asynchronously fetch 1-min candlestick data from Deriv API, save to CSV, and return DataFrame."""
    logger.info("Starting data fetch from Deriv API.")
    api = DerivAPI(app_id=APP_ID)
    candles = []
    try:
        # Remove explicit connect call. Optionally, if you want to be sure the API is ready,
        # you could wait until it reports connected:
        while not api.connected:
            await asyncio.sleep(0.1)
        
        request = {
            "ticks_history": SYMBOL,
            "adjust_start_time": 1,
            "count": CANDLE_COUNT,
            "end": "latest",
            "granularity": GRANULARITY,
            "style": "candles"
        }
        response = await api.send(request)
        candles = response.get("candles", [])
        logger.info(f"Fetched {len(candles)} candles.")
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}", exc_info=True)
    # No need to disconnect if the API manages its connection automatically.
    if candles:
        try:
            df = pd.DataFrame(candles)
            df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
            df.to_csv(CSV_FILE, index=False)
            logger.info(f"Data saved to {CSV_FILE}.")
            return df
        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning("No candle data received.")
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

def plot_chart(df):
    """Plot candlestick chart with SMAs and save to static folder."""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['close'], label='Close Price', color='blue')
        plt.plot(df.index, df['SMA_5'], label='SMA 5', color='orange')
        plt.plot(df.index, df['SMA_10'], label='SMA 10', color='green')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{SYMBOL} - 1 Minute Chart ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
        plt.legend()
        os.makedirs('static', exist_ok=True)
        plt.savefig(CHART_FILE)
        plt.close()
        logger.info(f"Chart saved to {CHART_FILE}.")
    except Exception as e:
        logger.error(f"Error plotting chart: {str(e)}", exc_info=True)


# --- MAIN FUNCTION FOR DATA FETCH AND ANALYSIS ---
def fetch_and_analyze():
    """Function to fetch data, analyze it, and log the trading signal."""
    try:
        # Use asyncio.run to handle the async function call
        df = asyncio.run(fetch_deriv_candles())
        if df.empty:
            logger.warning("No data to process.")
            return
        analyzed_df, recommendation = analyze_data(df)
        plot_chart(analyzed_df)
        
        # Log the trading signal with additional information
        current_price = analyzed_df['close'].iloc[-1] if not analyzed_df.empty else "Unknown"
        logger.info(f"TRADING SIGNAL: {recommendation} | Symbol: {SYMBOL} | Current Price: {current_price} | Time: {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.error(f"Error in fetch_and_analyze: {str(e)}", exc_info=True)


def run_bot():
    """Main function to run the trading bot continuously."""
    logger.info("Starting Trading Bot...")
    
    running = True
    while running:
        try:
            fetch_and_analyze()
            logger.info(f"Waiting {UPDATE_INTERVAL} seconds until next update...")
            
            # Break the sleep into smaller chunks to handle interrupts more gracefully
            for _ in range(UPDATE_INTERVAL):
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
        run_bot()
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading bot shutdown complete.")
