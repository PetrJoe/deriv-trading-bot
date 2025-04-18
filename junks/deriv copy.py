import os
import logging
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from deriv_api import DerivAPI
from dotenv import load_dotenv
import time
from telegram import Bot  # For sending signals
import ta  # Technical analysis package

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("trading_bot_app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Config
APP_ID = os.getenv('APP_ID', 'YOUR_APP_ID')
SYMBOL = os.getenv('SYMBOL', 'R_75')
GRANULARITY = int(os.getenv('GRANULARITY', 60))
CANDLE_COUNT = int(os.getenv('CANDLE_COUNT', 100))
CSV_FILE = os.getenv('CSV_FILE', 'data.csv')
CHART_FILE = os.path.join('static', os.getenv('CHART_FILE', 'chart.png'))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 300))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'your_bot_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')


async def fetch_deriv_candles(symbol=SYMBOL, granularity=GRANULARITY, count=CANDLE_COUNT, csv_file=CSV_FILE):
    logger.info(f"Fetching data for {symbol}")
    api = DerivAPI(app_id=APP_ID)

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

    if candles:
        df = pd.DataFrame(candles)
        df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
        df.to_csv(csv_file, index=False)
        return df
    return pd.DataFrame()


def detect_chart_patterns(df):
    """Detect simple chart patterns."""
    pattern = "None"

    # Check for double top
    recent = df['close'].tail(20).values
    if len(recent) >= 10:
        max1 = recent[:10].max()
        max2 = recent[10:].max()
        if abs(max1 - max2) / max1 < 0.01:
            pattern = "Double Top"

    # Check for double bottom
    min1 = recent[:10].min()
    min2 = recent[10:].min()
    if abs(min1 - min2) / min1 < 0.01:
        pattern = "Double Bottom"

    return pattern


def analyze_data(df):
    try:
        df = df.copy()
        df.set_index('epoch', inplace=True)

        # Calculate moving averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        # Signal logic
        df['signal'] = 0
        if df['SMA_5'].iloc[-1] > df['SMA_10'].iloc[-1] and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] and df['RSI'].iloc[-1] < 70:
            signal = "Buy"
        elif df['SMA_5'].iloc[-1] < df['SMA_10'].iloc[-1] and df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1] and df['RSI'].iloc[-1] > 30:
            signal = "Sell"
        else:
            signal = "Hold"

        pattern = detect_chart_patterns(df)
        return df, signal, pattern

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return df, "Hold", "None"



def plot_chart(df, symbol=SYMBOL, chart_file=CHART_FILE):
    try:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Price + SMA
        axs[0].plot(df.index, df['close'], label='Close', color='blue')
        axs[0].plot(df.index, df['SMA_5'], label='SMA 5', color='orange')
        axs[0].plot(df.index, df['SMA_10'], label='SMA 10', color='green')
        axs[0].set_title(f'{symbol} Price & SMA')
        axs[0].legend()

        # RSI
        axs[1].plot(df.index, df['RSI'], label='RSI', color='purple')
        axs[1].axhline(70, color='red', linestyle='--', linewidth=1)
        axs[1].axhline(30, color='green', linestyle='--', linewidth=1)
        axs[1].set_title('RSI')
        axs[1].legend()

        # MACD
        axs[2].plot(df.index, df['MACD'], label='MACD', color='black')
        axs[2].plot(df.index, df['MACD_signal'], label='Signal Line', color='magenta')
        axs[2].set_title('MACD')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(chart_file)
        plt.close()
    except Exception as e:
        logger.error(f"Error in chart plotting: {str(e)}")


def send_telegram_signal(message, image_path=None):
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        # bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        if image_path and os.path.exists(image_path):
            bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open(image_path, 'rb'))
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")


def fetch_and_analyze(symbol=SYMBOL, granularity=GRANULARITY, count=CANDLE_COUNT):
    try:
        df = asyncio.run(fetch_deriv_candles(symbol, granularity, count))
        if df.empty:
            return

        analyzed_df, recommendation, pattern = analyze_data(df)
        plot_chart(analyzed_df, symbol)

        current_price = analyzed_df['close'].iloc[-1]
        rsi = analyzed_df['RSI'].iloc[-1]
        macd_val = analyzed_df['MACD'].iloc[-1]
        macd_signal = analyzed_df['MACD_signal'].iloc[-1]
        
        signal_message = (
            f"📊 *{symbol}* Signal: *{recommendation}*\n"
            f"🧠 Pattern: *{pattern}*\n"
            f"💰 Price: {current_price:.2f}\n"
            f"📈 RSI: {rsi:.2f}\n"
            f"📉 MACD: {macd_val:.2f} | Signal: {macd_signal:.2f}\n"
            f"🕒 {datetime.now().strftime('%H:%M:%S')}"
        )
        
        logger.info(signal_message)
        send_telegram_signal(signal_message, image_path=CHART_FILE)

        return analyzed_df, recommendation

    except Exception as e:
        logger.error(f"Error in fetch_and_analyze: {str(e)}")
        return None, None


def run_bot(symbols=None, granularity=GRANULARITY, count=CANDLE_COUNT, update_interval=UPDATE_INTERVAL):
    if symbols is None:
        symbols = [SYMBOL]
    elif isinstance(symbols, str):
        symbols = [symbols]

    logger.info(f"Bot started. Monitoring: {symbols}")

    while True:
        try:
            for symbol in symbols:
                logger.info(f"Processing {symbol}")
                fetch_and_analyze(symbol, granularity, count)

            time.sleep(update_interval)
        except KeyboardInterrupt:
            logger.info("Bot stopped manually.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {str(e)}")
            time.sleep(10)


# Entry point
if __name__ == "__main__":
    try:
        symbols_to_monitor = os.getenv('SYMBOLS', SYMBOL).split(',')
        run_bot(symbols=symbols_to_monitor)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")


