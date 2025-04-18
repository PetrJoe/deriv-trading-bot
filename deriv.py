import os
import logging
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from deriv_api import DerivAPI
from dotenv import load_dotenv
import time
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import ta  # Technical analysis package
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pattern_recognition import PatternRecognition



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

# Available symbols for analysis
AVAILABLE_SYMBOLS = [
    'R_10', 'R_25', 'R_50', 'R_75', 'R_100',
    'BOOM500', 'BOOM1000', 'CRASH500', 'CRASH1000',
    'RDBEAR', 'RDBULL'
]

# Available timeframes
AVAILABLE_TIMEFRAMES = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}

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


async def analyze_data(df):
    try:
        df = df.copy()
        df.set_index('epoch', inplace=True)

        # Calculate moving averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()  # Add longer SMA for trend direction

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Add ADX for trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['ADX'] = adx.adx()
        df['DI+'] = adx.adx_pos()
        df['DI-'] = adx.adx_neg()

        # Use PatternRecognition for advanced pattern detection
        pattern_analyzer = PatternRecognition(df)
        pattern_recommendation, pattern_signals = pattern_analyzer.get_trading_signal()
        
        # Determine market direction
        market_direction = "Sideways"
        direction_strength = "Weak"
        
        # Check trend direction using SMAs and ADX
        if df['SMA_5'].iloc[-1] > df['SMA_20'].iloc[-1] and df['close'].iloc[-1] > df['SMA_5'].iloc[-1]:
            market_direction = "Uptrend"
            if df['ADX'].iloc[-1] > 25:
                direction_strength = "Strong"
            elif df['ADX'].iloc[-1] > 20:
                direction_strength = "Moderate"
        elif df['SMA_5'].iloc[-1] < df['SMA_20'].iloc[-1] and df['close'].iloc[-1] < df['SMA_5'].iloc[-1]:
            market_direction = "Downtrend"
            if df['ADX'].iloc[-1] > 25:
                direction_strength = "Strong"
            elif df['ADX'].iloc[-1] > 20:
                direction_strength = "Moderate"
        
        # Generate more decisive signal with relaxed conditions
        traditional_signal = "Hold"
        
        # Buy conditions - more aggressive
        if ((df['SMA_5'].iloc[-1] > df['SMA_10'].iloc[-1]) or 
            (df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]) or 
            (df['RSI'].iloc[-1] < 45) or
            (df['DI+'].iloc[-1] > df['DI-'].iloc[-1] and df['ADX'].iloc[-1] > 20)):
            traditional_signal = "Buy"
        # Sell conditions - more aggressive
        elif ((df['SMA_5'].iloc[-1] < df['SMA_10'].iloc[-1]) or 
              (df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]) or 
              (df['RSI'].iloc[-1] > 55) or
              (df['DI-'].iloc[-1] > df['DI+'].iloc[-1] and df['ADX'].iloc[-1] > 20)):
            traditional_signal = "Sell"
        
        # Prioritize pattern recognition signals, but use traditional if pattern is "Hold"
        final_signal = pattern_recommendation if pattern_recommendation != "Hold" else traditional_signal
        
        # Get the most significant detected patterns
        detected_patterns = [pattern for pattern, details in pattern_signals.items() 
                           if details.get('detected', False)]
        pattern_str = ", ".join(detected_patterns) if detected_patterns else "None"

        return df, final_signal, pattern_str, market_direction, direction_strength

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return df, "Hold", "None", "Unknown", "Unknown"


async def r75_analysis_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed analysis of R_75 with pattern recognition."""
    await update.message.reply_text("Analyzing R_75 with pattern recognition... Please wait.")
    
    # Analyze with different timeframes
    timeframes = ['5m', '15m', '1h', '4h']
    results = []
    
    for timeframe in timeframes:
        granularity = AVAILABLE_TIMEFRAMES[timeframe]
        df = await fetch_deriv_candles(symbol='R_75', granularity=granularity, count=100)
        
        if df.empty:
            results.append(f"❌ Failed to fetch data for {timeframe} timeframe")
            continue
            
        # Update to include market direction
        analyzed_df, recommendation, patterns, market_direction, direction_strength = await analyze_data(df)
        
        # Get key indicators
        current_price = analyzed_df['close'].iloc[-1]
        rsi = analyzed_df['RSI'].iloc[-1]
        macd_val = analyzed_df['MACD'].iloc[-1]
        macd_signal = analyzed_df['MACD_signal'].iloc[-1]
        
        # Create a message for this timeframe
        signal_emoji = "🟢" if recommendation == "Buy" else "🔴" if recommendation == "Sell" else "⚪"
        direction_emoji = "↔️" if market_direction == "Sideways" else "🔼" if market_direction == "Uptrend" else "🔽"
        
        timeframe_result = (
            f"{signal_emoji} <b>{timeframe}</b>: {recommendation} {direction_emoji}\n"
            f"  • Direction: {market_direction} ({direction_strength})\n"
            f"  • Patterns: {patterns}\n"
            f"  • RSI: {rsi:.2f}\n"
            f"  • MACD: {macd_val:.2f} | Signal: {macd_signal:.2f}\n"
        )
        results.append(timeframe_result)
    
    # Generate chart for 1h timeframe
    df = await fetch_deriv_candles(symbol='R_75', granularity=AVAILABLE_TIMEFRAMES['1h'], count=100)
    if not df.empty:
        # Update to include market direction
        analyzed_df, _, _, market_direction, direction_strength = await analyze_data(df)
        chart_path = plot_chart(analyzed_df, symbol='R_75')
    else:
        chart_path = None
    
    # Determine overall recommendation based on multiple timeframes
    buy_count = sum(1 for result in results if "🟢" in result)
    sell_count = sum(1 for result in results if "🔴" in result)
    
    if buy_count > sell_count:
        overall = "🟢 <b>OVERALL: BUY (LONG)</b>"
    elif sell_count > buy_count:
        overall = "🔴 <b>OVERALL: SELL (SHORT)</b>"
    else:
        overall = "⚪ <b>OVERALL: NEUTRAL</b>"
    
    # Add overall market direction
    direction_emoji = "↔️" if market_direction == "Sideways" else "🔼" if market_direction == "Uptrend" else "🔽"
    overall_direction = f"{direction_emoji} <b>MARKET DIRECTION: {market_direction} ({direction_strength})</b>"
    
    # Combine all results
    message = (
        f"📊 <b>R_75 ANALYSIS</b> 📊\n\n"
        f"{overall}\n"
        f"{overall_direction}\n\n"
        f"<b>TIMEFRAME ANALYSIS:</b>\n"
        f"{chr(10).join(results)}\n"
        f"🕒 {datetime.now().strftime('%H:%M:%S')}"
    )
    
    if chart_path and os.path.exists(chart_path):
        await update.message.reply_photo(
            photo=open(chart_path, 'rb'),
            caption=message,
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text(message, parse_mode='HTML')


async def force_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force a signal analysis with lower thresholds for testing purposes."""
    if not context.args or len(context.args) < 1:
        await update.message.reply_text(
            "Please specify a symbol. Example: /force_signal R_75"
        )
        return

    symbol = context.args[0].upper()
    if symbol not in AVAILABLE_SYMBOLS:
        await update.message.reply_text(
            f"Symbol {symbol} not found. Use /symbols to see available options."
        )
        return

    await update.message.reply_text(f"Forcing signal analysis for {symbol}... Please wait.")
    
    df = await fetch_deriv_candles(symbol=symbol)
    if df.empty:
        await update.message.reply_text("Failed to fetch data.")
        return
    
    # Analyze with lower thresholds
    df = df.copy()
    df.set_index('epoch', inplace=True)
    
    # Calculate indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Add ADX for trend strength
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['ADX'] = adx.adx()
    df['DI+'] = adx.adx_pos()
    df['DI-'] = adx.adx_neg()
    
    # Determine market direction
    market_direction = "Sideways"
    direction_strength = "Weak"
    
    # Check trend direction using SMAs and ADX
    if df['SMA_5'].iloc[-1] > df['SMA_20'].iloc[-1] and df['close'].iloc[-1] > df['SMA_5'].iloc[-1]:
        market_direction = "Uptrend"
        if df['ADX'].iloc[-1] > 25:
            direction_strength = "Strong"
        elif df['ADX'].iloc[-1] > 20:
            direction_strength = "Moderate"
    elif df['SMA_5'].iloc[-1] < df['SMA_20'].iloc[-1] and df['close'].iloc[-1] < df['SMA_5'].iloc[-1]:
        market_direction = "Downtrend"
        if df['ADX'].iloc[-1] > 25:
            direction_strength = "Strong"
        elif df['ADX'].iloc[-1] > 20:
            direction_strength = "Moderate"
    
    # Get pattern signals
    pattern_analyzer = PatternRecognition(df)
    pattern_signals = pattern_analyzer.detect_all_patterns()
    
    # Count signals
    bullish_patterns = [p for p, r in pattern_signals.items() 
                      if isinstance(r, dict) and r.get('detected', False) and r.get('signal') == 'bullish']
    bearish_patterns = [p for p, r in pattern_signals.items() 
                      if isinstance(r, dict) and r.get('detected', False) and r.get('signal') == 'bearish']
    
    # Technical indicators
    current_price = df['close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd_val = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    
    # Generate signal with very low threshold - more decisive
    if len(bullish_patterns) > len(bearish_patterns) or macd_val > macd_signal or rsi < 45 or df['DI+'].iloc[-1] > df['DI-'].iloc[-1]:
        recommendation = "Buy"
    elif len(bearish_patterns) > len(bullish_patterns) or macd_val < macd_signal or rsi > 55 or df['DI-'].iloc[-1] > df['DI+'].iloc[-1]:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    
    # Create detailed message
    signal_emoji = "🟢" if recommendation == "Buy" else "🔴" if recommendation == "Sell" else "⚪"
    direction_emoji = "↔️" if market_direction == "Sideways" else "🔼" if market_direction == "Uptrend" else "🔽"
    
    message = (
        f"{signal_emoji} <b>{symbol} FORCED SIGNAL: {recommendation}</b>\n"
        f"{direction_emoji} <b>Market Direction: {market_direction} ({direction_strength})</b>\n\n"
        f"<b>Technical Indicators:</b>\n"
        f"• Price: {current_price:.2f}\n"
        f"• RSI: {rsi:.2f}\n"
        f"• MACD: {macd_val:.2f} | Signal: {macd_signal:.2f}\n"
        f"• SMA 5: {df['SMA_5'].iloc[-1]:.2f}\n"
        f"• SMA 10: {df['SMA_10'].iloc[-1]:.2f}\n"
        f"• ADX: {df['ADX'].iloc[-1]:.2f} (DI+: {df['DI+'].iloc[-1]:.2f}, DI-: {df['DI-'].iloc[-1]:.2f})\n\n"
        f"<b>Bullish Patterns ({len(bullish_patterns)}):</b>\n"
        f"{', '.join(bullish_patterns) if bullish_patterns else 'None'}\n\n"
        f"<b>Bearish Patterns ({len(bearish_patterns)}):</b>\n"
        f"{', '.join(bearish_patterns) if bearish_patterns else 'None'}\n\n"
        f"🕒 {datetime.now().strftime('%H:%M:%S')}"
    )
    
    # Plot chart
    chart_path = plot_chart(df, symbol)
    
    if chart_path and os.path.exists(chart_path):
        await update.message.reply_photo(
            photo=open(chart_path, 'rb'),
            caption=message,
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text(message, parse_mode='HTML')


def plot_chart(df, symbol=SYMBOL, chart_file=CHART_FILE):
    try:
        os.makedirs(os.path.dirname(chart_file), exist_ok=True)
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

        # Price + SMA
        axs[0].plot(df.index, df['close'], label='Close', color='blue')
        axs[0].plot(df.index, df['SMA_5'], label='SMA 5', color='orange')
        axs[0].plot(df.index, df['SMA_10'], label='SMA 10', color='green')
        if 'SMA_20' in df.columns:
            axs[0].plot(df.index, df['SMA_20'], label='SMA 20', color='red')
        
        # Add market direction arrow and text
        last_idx = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        # Determine trend direction for visual indicator
        if 'SMA_5' in df.columns and 'SMA_20' in df.columns:
            if df['SMA_5'].iloc[-1] > df['SMA_20'].iloc[-1]:
                axs[0].annotate('↑', xy=(last_idx, last_price*1.01), 
                             xytext=(last_idx, last_price*1.03),
                             arrowprops=dict(facecolor='green', shrink=0.05),
                             fontsize=16, color='green')
            elif df['SMA_5'].iloc[-1] < df['SMA_20'].iloc[-1]:
                axs[0].annotate('↓', xy=(last_idx, last_price*0.99), 
                             xytext=(last_idx, last_price*0.97),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             fontsize=16, color='red')
        
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
        
        # ADX if available
        if 'ADX' in df.columns and 'DI+' in df.columns and 'DI-' in df.columns:
            axs[3].plot(df.index, df['ADX'], label='ADX', color='blue')
            axs[3].plot(df.index, df['DI+'], label='DI+', color='green')
            axs[3].plot(df.index, df['DI-'], label='DI-', color='red')
            axs[3].axhline(25, color='gray', linestyle='--', linewidth=1)
            axs[3].set_title('ADX')
            axs[3].legend()

        plt.tight_layout()
        plt.savefig(chart_file)
        plt.close()
        return chart_file
    except Exception as e:
        logger.error(f"Error in chart plotting: {str(e)}")
        return None



async def fetch_and_analyze(symbol=SYMBOL, granularity=GRANULARITY, count=CANDLE_COUNT):
    try:
        df = await fetch_deriv_candles(symbol, granularity, count)
        if df.empty:
            return None, None, None, "Failed to fetch data", None, None

        # Update to include market direction
        analyzed_df, recommendation, pattern, market_direction, direction_strength = await analyze_data(df)
        chart_path = plot_chart(analyzed_df, symbol)

        current_price = analyzed_df['close'].iloc[-1]
        rsi = analyzed_df['RSI'].iloc[-1]
        macd_val = analyzed_df['MACD'].iloc[-1]
        macd_signal = analyzed_df['MACD_signal'].iloc[-1]
        
        # Add direction emoji
        direction_emoji = "↔️" if market_direction == "Sideways" else "🔼" if market_direction == "Uptrend" else "🔽"
        signal_emoji = "🟢" if recommendation == "Buy" else "🔴" if recommendation == "Sell" else "⚪"
        
        # Using HTML formatting instead of Markdown
        signal_message = (
            f"{signal_emoji} <b>{symbol}</b> Signal: <b>{recommendation}</b>\n"
            f"{direction_emoji} Market Direction: <b>{market_direction} ({direction_strength})</b>\n"
            f"🧠 Pattern: <b>{pattern}</b>\n"
            f"💰 Price: {current_price:.2f}\n"
            f"📈 RSI: {rsi:.2f}\n"
            f"📉 MACD: {macd_val:.2f} | Signal: {macd_signal:.2f}\n"
            f"🕒 {datetime.now().strftime('%H:%M:%S')}"
        )
        
        logger.info(signal_message)
        return analyzed_df, recommendation, chart_path, signal_message, market_direction, direction_strength

    except Exception as e:
        logger.error(f"Error in fetch_and_analyze: {str(e)}")
        return None, None, None, f"Error: {str(e)}", None, None


async def periodic_r75_analysis(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodically analyze R_75 and send updates to a specified chat."""
    chat_id = context.job.data.get('chat_id')
    if not chat_id:
        logger.error("No chat_id provided for periodic analysis")
        return
    
    logger.info(f"Running periodic R_75 analysis for chat {chat_id}")
    
    # Analyze with 1h timeframe
    granularity = AVAILABLE_TIMEFRAMES['1h']
    df = await fetch_deriv_candles(symbol='R_75', granularity=granularity, count=100)
    
    if df.empty:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Failed to fetch data for R_75 analysis",
            parse_mode='HTML'
        )
        return
        
    # Update to include market direction
    analyzed_df, recommendation, patterns, market_direction, direction_strength = await analyze_data(df)
    chart_path = plot_chart(analyzed_df, symbol='R_75')
    
    # Get key indicators
    current_price = analyzed_df['close'].iloc[-1]
    rsi = analyzed_df['RSI'].iloc[-1]
    macd_val = analyzed_df['MACD'].iloc[-1]
    macd_signal = analyzed_df['MACD_signal'].iloc[-1]
    
    # Create signal message
    signal_emoji = "🟢" if recommendation == "Buy" else "🔴" if recommendation == "Sell" else "⚪"
    direction_emoji = "↔️" if market_direction == "Sideways" else "🔼" if market_direction == "Uptrend" else "🔽"
    
    message = (
        f"🔄 <b>R_75 AUTOMATIC UPDATE</b> 🔄\n\n"
        f"{signal_emoji} <b>Signal: {recommendation}</b>\n"
        f"{direction_emoji} <b>Market Direction: {market_direction} ({direction_strength})</b>\n"
        f"🧠 Patterns: {patterns}\n"
        f"💰 Price: {current_price:.2f}\n"
        f"📈 RSI: {rsi:.2f}\n"
        f"📉 MACD: {macd_val:.2f} | Signal: {macd_signal:.2f}\n"
        f"🕒 {datetime.now().strftime('%H:%M:%S')}"
    )
    
    if chart_path and os.path.exists(chart_path):
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=open(chart_path, 'rb'),
            caption=message,
            parse_mode='HTML'
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='HTML'
        )


async def start_auto_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start automatic R_75 analysis."""
    chat_id = update.effective_chat.id
    
    # Remove existing job if any
    current_jobs = context.job_queue.get_jobs_by_name(f"r75_analysis_{chat_id}")
    for job in current_jobs:
        job.schedule_removal()
    
    # Get interval from arguments or use default
    interval = 60  # Default: 60 minutes
    if context.args and len(context.args) > 0:
        try:
            interval = int(context.args[0])
            if interval < 5:
                interval = 5  # Minimum 5 minutes
        except ValueError:
            pass
    
    # Schedule new job
    context.job_queue.run_repeating(
        periodic_r75_analysis, 
        interval=interval*60,  # Convert to seconds
        first=10,  # Start first analysis after 10 seconds
        data={'chat_id': chat_id},
        name=f"r75_analysis_{chat_id}"
    )
    
    await update.message.reply_text(
        f"✅ Automatic R_75 analysis started. You will receive updates every {interval} minutes."
    )


async def stop_auto_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop automatic R_75 analysis."""
    chat_id = update.effective_chat.id
    
    # Remove existing job if any
    current_jobs = context.job_queue.get_jobs_by_name(f"r75_analysis_{chat_id}")
    for job in current_jobs:
        job.schedule_removal()
    
    await update.message.reply_text("✅ Automatic R_75 analysis stopped.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 Hello {user.first_name}!\n\n"
        f"Welcome to the Deriv Trading Signal Bot. I can help you analyze trading signals for various assets.\n\n"
        f"Available commands:\n"
        f"/signal <symbol> - Get trading signals for a specific symbol\n"
        f"/symbols - List all available symbols\n"
        f"/timeframes - List available timeframes\n"
        f"/help - Show this help message"
    )
    await update.message.reply_text(welcome_message)



async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    help_message = (
        "🤖 <b>Deriv Trading Signal Bot Help</b>\n\n"
        "<b>Available Commands:</b>\n"
        "/signal &lt;symbol&gt; - Get trading signals (e.g., /signal R_75)\n"
        "/symbols - List all available symbols\n"
        "/timeframes - List available timeframes\n"
        "/analyze &lt;symbol&gt; &lt;timeframe&gt; - Detailed analysis (e.g., /analyze R_75 1h)\n"
        "/r75 - Comprehensive R_75 analysis across multiple timeframes\n"
        "/force_signal &lt;symbol&gt; - Force signal generation with lower thresholds\n"
        "/auto_start [interval] - Start automatic R_75 analysis (interval in minutes, default: 60)\n"
        "/auto_stop - Stop automatic R_75 analysis\n"
        "/help - Show this help message\n\n"
        "For any issues or feedback, please contact the administrator @petrjoe."
    )
    await update.message.reply_text(help_message, parse_mode='HTML')


async def list_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all available symbols."""
    symbols_message = "📈 <b>Available Symbols:</b>\n\n"
    for symbol in AVAILABLE_SYMBOLS:
        symbols_message += f"• {symbol}\n"
    symbols_message += "\nUse /signal &lt;symbol&gt; to get trading signals."
    await update.message.reply_text(symbols_message, parse_mode='HTML')


async def list_timeframes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all available timeframes."""
    timeframes_message = "⏱️ <b>Available Timeframes:</b>\n\n"
    for tf, seconds in AVAILABLE_TIMEFRAMES.items():
        timeframes_message += f"• {tf}\n"
    timeframes_message += "\nUse /analyze &lt;symbol&gt; &lt;timeframe&gt; for detailed analysis."
    await update.message.reply_text(timeframes_message, parse_mode='HTML')


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get trading signal for a specific symbol."""
    if not context.args:
        await update.message.reply_text("Please specify a symbol. Example: /signal R_75")
        return

    symbol = context.args[0].upper()
    if symbol not in AVAILABLE_SYMBOLS:
        await update.message.reply_text(
            f"Symbol {symbol} not found. Use /symbols to see available options."
        )
        return

    await update.message.reply_text(f"Analyzing {symbol}... Please wait.")
    
    df = await fetch_deriv_candles(symbol=symbol)
    if df.empty:
        await update.message.reply_text("Failed to fetch data.")
        return
    
    # Update to include market direction
    analyzed_df, recommendation, patterns, market_direction, direction_strength = await analyze_data(df)
    chart_path = plot_chart(analyzed_df, symbol)
    
    current_price = analyzed_df['close'].iloc[-1]
    rsi = analyzed_df['RSI'].iloc[-1]
    macd_val = analyzed_df['MACD'].iloc[-1]
    macd_signal = analyzed_df['MACD_signal'].iloc[-1]
    
    # Add direction emoji
    direction_emoji = "↔️" if market_direction == "Sideways" else "🔼" if market_direction == "Uptrend" else "🔽"
    signal_emoji = "🟢" if recommendation == "Buy" else "🔴" if recommendation == "Sell" else "⚪"
    
    signal_message = (
        f"{signal_emoji} <b>{symbol}</b> Signal: <b>{recommendation}</b>\n"
        f"{direction_emoji} Market Direction: <b>{market_direction} ({direction_strength})</b>\n"
        f"🧠 Pattern: <b>{patterns}</b>\n"
        f"💰 Price: {current_price:.2f}\n"
        f"📈 RSI: {rsi:.2f}\n"
        f"📉 MACD: {macd_val:.2f} | Signal: {macd_signal:.2f}\n"
        f"🕒 {datetime.now().strftime('%H:%M:%S')}"
    )
    
    if chart_path and os.path.exists(chart_path):
        await update.message.reply_photo(
            photo=open(chart_path, 'rb'),
            caption=signal_message,
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text(signal_message, parse_mode='HTML')


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed analysis with custom timeframe."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Please specify both symbol and timeframe. Example: /analyze R_75 1h"
        )
        return

    symbol = context.args[0].upper()
    timeframe = context.args[1].lower()

    if symbol not in AVAILABLE_SYMBOLS:
        await update.message.reply_text(
            f"Symbol {symbol} not found. Use /symbols to see available options."
        )
        return

    if timeframe not in AVAILABLE_TIMEFRAMES:
        await update.message.reply_text(
            f"Timeframe {timeframe} not found. Use /timeframes to see available options."
        )
        return

    granularity = AVAILABLE_TIMEFRAMES[timeframe]
    
    await update.message.reply_text(f"Analyzing {symbol} on {timeframe} timeframe... Please wait.")
    
    _, _, chart_path, signal_message = await fetch_and_analyze(
        symbol=symbol, 
        granularity=granularity
    )
    
    if chart_path and os.path.exists(chart_path):
        await update.message.reply_photo(
            photo=open(chart_path, 'rb'),
            caption=signal_message,
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text(signal_message, parse_mode='HTML')


# Add error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a message to the user."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # Send message to the user
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "Sorry, an error occurred while processing your request. Please try again later."
        )


def run_telegram_bot():
    """Run the Telegram bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("symbols", list_symbols))
    application.add_handler(CommandHandler("timeframes", list_timeframes))
    application.add_handler(CommandHandler("signal", signal_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("force_signal", force_signal_command))  # Add this line
    
    # Add R_75 specific commands
    application.add_handler(CommandHandler("r75", r75_analysis_command))
    application.add_handler(CommandHandler("auto_start", start_auto_analysis))
    application.add_handler(CommandHandler("auto_stop", stop_auto_analysis))
    
    # Add error handler
    application.add_error_handler(error_handler)

    # Start the Bot
    application.run_polling()
    

# Entry point
if __name__ == "__main__":
    try:
        logger.info("Starting Deriv Trading Signal Bot...")
        run_telegram_bot()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        
