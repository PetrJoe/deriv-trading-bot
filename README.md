# Deriv Trading Pattern Analyzer

A comprehensive trading bot and pattern recognition system for Deriv assets. This tool analyzes price patterns, identifies technical chart patterns, and provides trading signals for various Deriv trading pairs.

## Features
- Real-time Data Fetching: Connect to Deriv API to fetch candlestick data
- Technical Analysis: Calculate moving averages and detect chart patterns
- Pattern Recognition: Identify over 20 chart patterns including Head & Shoulders, Double Tops/Bottoms, Triangles, etc.
- Candlestick Patterns: Detect common candlestick patterns using TA-Lib
- Trading Signals: Generate Buy/Sell/Hold recommendations based on detected patterns
- Visualization: Create detailed charts with highlighted patterns and signals
- Web Interface: View analysis results through a Flask web application
- Continuous Monitoring: Run in continuous mode to track multiple trading pairs
- Command-line Interface: Easy-to-use CLI for quick analysis

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- TA-Lib (Technical Analysis Library)

### Step 1: Clone the Repository
```bash
git clone [(https://github.com/PetrJoe/deriv-trading-bot.git)]
cd deriv-trading-bot
```

### Step 2: Install TA-Lib
TA-Lib is required for candlestick pattern recognition. Installation varies by platform:

On Ubuntu/Debian:
```bash
sudo apt-get install build-essential
sudo apt-get install python3-dev
sudo apt-get install libta-lib0 libta-lib-dev
pip install TA-Lib
```

On macOS (using Homebrew):
```bash
brew install ta-lib
pip install TA-Lib
```

On Windows:
Download the appropriate wheel file from here and install it:
```bash
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```
(Replace the filename with the one matching your Python version and architecture)

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root directory with the following content:

```env
# Deriv secret Key
SECRET_KEY=deriv secret key

# Deriv API Configuration
APP_ID=YOUR_APP_ID_HERE

# Default Trading Pair
SYMBOL=R_75

# Multiple symbols can be specified as comma-separated values
# SYMBOLS=R_75,R_100,RDBEAR

# Candle Configuration
GRANULARITY=60
CANDLE_COUNT=100

# Update Interval (seconds)
UPDATE_INTERVAL=300

# Output Files
CSV_FILE=data.csv
CHART_FILE=chart.png
```

Replace `YOUR_APP_ID_HERE` with your Deriv API App ID. You can get one by creating an app at Deriv API.

## Usage

### Command-line Interface
The CLI provides a simple way to run analysis on Deriv assets:

```bash
python cli.py --symbols R_75,R_100 --granularity 60 --count 100 --output results.json
```

Options:
- `--symbols`, `-s`: Comma-separated list of symbols to analyze (default: from .env)
- `--granularity`, `-g`: Candle granularity in seconds (default: 60)
- `--count`, `-c`: Number of candles to fetch (default: 100)
- `--output`, `-o`: Output file for analysis results (default: static/analysis_output.json)
- `--continuous`: Run analysis continuously
- `--interval`, `-i`: Update interval in seconds for continuous mode (default: 300)

### Running the Trading Bot
To run the trading bot in continuous mode:

```bash
python deriv_trader.py
```

This will start the bot with the settings from your .env file.

### Web Interface
To start the web interface:

```bash
python app.py
```

Then open your browser and navigate to http://localhost:5000 to view the analysis dashboard.

## Example Analysis Workflow

### Quick Single Analysis:
```bash
python cli.py --symbols R_75
```

### Continuous Monitoring of Multiple Pairs:
```bash
python cli.py --symbols R_75,R_100,RDBEAR --continuous --interval 300
```

### Detailed Analysis with Web Interface:
```bash
python app.py
```
Then open your browser to http://localhost:5000 and use the interface to analyze specific symbols.

## Understanding the Results
The analysis results include:
- Current Price: Latest price of the asset
- Change Percent: Price change since previous candle
- Recommendation: BUY, SELL, or HOLD based on detected patterns
- Signal Strength: Confidence level of the recommendation (0-1)
- Detected Patterns: List of technical patterns found in the price data
- Charts: Visual representation of the price with highlighted patterns

## Customizing the Analysis
You can customize the analysis by:
- Modifying the `.env` file to change default settings
- Editing `pattern_recognition.py` to adjust pattern detection parameters
- Updating `chart_utils.py` to change chart visualization options

## Troubleshooting

### Common Issues:
- Connection Error: Make sure your APP_ID is valid and you have internet access
- No Data Received: Verify the symbol name is correct and available on Deriv
- TA-Lib Import Error: Ensure TA-Lib is properly installed for your platform

### Logs:
Check the log files for detailed error information:
- `deriv_trader.log`: Contains logs from the trading bot
- `trading_bot_app.log`: Contains logs from the bot module

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Deriv API for providing access to market data
- TA-Lib for technical analysis functions
- Matplotlib for chart visualization



## Entry Points

### bot.py
This is the main trading bot entry point. You can run it directly with:

```bash
python bot.py
```

This will start the trading bot that continuously fetches data from Deriv API, analyzes it, and logs trading signals.

### app.py
This is the Flask web application entry point. You can run it with:

```bash
python app.py
```

This will start a web server that provides a user interface for analyzing stocks.

### deriv_trader.py
This is the entry point for the enhanced Deriv trader. You can run it with:

```bash
python deriv_trader.py
```

This will start the enhanced trading analysis system that continuously monitors multiple Deriv assets.

### cli.py
This is the command-line interface entry point. You can run it with:

```bash
python cli.py
```

This provides a flexible command-line tool for running analyses with various options.

### main.py
This contains the main function for stock pattern analysis. You can run it with:

```bash
python main.py
```

This will analyze a predefined list of stock symbols.

The primary entry point for the Deriv trading bot functionality is bot.py, while deriv_trader.py and cli.py provide enhanced functionality. The app.py file is for the web interface.

For more information or to report issues, please open an issue on the GitHub repository.