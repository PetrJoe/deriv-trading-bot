import argparse
import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from deriv_trader import DerivTrader

# Load environment variables
load_dotenv()

# Default values
DEFAULT_SYMBOL = os.getenv('SYMBOL', 'R_75')
DEFAULT_GRANULARITY = int(os.getenv('GRANULARITY', 60))
DEFAULT_CANDLE_COUNT = int(os.getenv('CANDLE_COUNT', 100))
DEFAULT_OUTPUT = 'static/analysis_output.json'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deriv Trading Pattern Analyzer')
    
    parser.add_argument('--symbols', '-s', type=str, default=DEFAULT_SYMBOL,
                        help=f'Comma-separated list of symbols to analyze (default: {DEFAULT_SYMBOL})')
    
    parser.add_argument('--granularity', '-g', type=int, default=DEFAULT_GRANULARITY,
                        help=f'Candle granularity in seconds (default: {DEFAULT_GRANULARITY})')
    
    parser.add_argument('--count', '-c', type=int, default=DEFAULT_CANDLE_COUNT,
                        help=f'Number of candles to fetch (default: {DEFAULT_CANDLE_COUNT})')
    
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output file for analysis results (default: {DEFAULT_OUTPUT})')
    
    parser.add_argument('--continuous', action='store_true',
                        help='Run analysis continuously')
    
    parser.add_argument('--interval', '-i', type=int, default=300,
                        help='Update interval in seconds for continuous mode (default: 300)')
    
    return parser.parse_args()

async def main():
    """Main function to run the analyzer."""
    args = parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    print(f"Deriv Trading Pattern Analyzer")
    print(f"==============================")
    print(f"Analyzing symbols: {', '.join(symbols)}")
    print(f"Granularity: {args.granularity} seconds")
    print(f"Candle count: {args.count}")
    print(f"Output file: {args.output}")
    
    # Create trader
    trader = DerivTrader(symbols, args.granularity, args.count, args.interval)
    
    if args.continuous:
        print(f"Running in continuous mode with {args.interval} second updates.")
        print("Press Ctrl+C to stop.")
        try:
            await trader.run_continuously()
        except KeyboardInterrupt:
            print("\nAnalysis stopped by user.")
    else:
        print("Running single analysis...")
        results = await trader.run_once()
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        summary = trader.get_summary()
        print("\nAnalysis Summary:")
        print(f"- Timestamp: {summary['timestamp']}")
        print(f"- Total symbols: {summary['total_symbols']}")
        print(f"- Successful analyses: {summary['successful_analyses']}")
        print(f"- Failed analyses: {summary['failed_analyses']}")
        
        if summary['buy_recommendations']:
            print(f"- BUY recommendations: {', '.join(summary['buy_recommendations'])}")
        if summary['sell_recommendations']:
            print(f"- SELL recommendations: {', '.join(summary['sell_recommendations'])}")
        if summary['hold_recommendations']:
            print(f"- HOLD recommendations: {', '.join(summary['hold_recommendations'])}")
        
        print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
