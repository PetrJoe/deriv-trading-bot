import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import os

def plot_enhanced_chart(df: pd.DataFrame, patterns: Dict, 
                        title: str = 'Chart Analysis with Patterns',
                        save_path: str = 'static/chart.png',
                        show_patterns: bool = True,
                        show_signals: bool = True,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot an enhanced chart with detected patterns and signals.
    
    Args:
        df: DataFrame with price data
        patterns: Dictionary of detected patterns
        title: Chart title
        save_path: Path to save the chart image
        show_patterns: Whether to highlight detected patterns
        show_signals: Whether to show buy/sell signals
        figsize: Figure size (width, height) in inches
    """
    # Create figure and primary axis for price
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot candlestick chart
    plot_candlestick_chart(ax1, df)
    
    # Plot moving averages
    if 'SMA_5' in df.columns and 'SMA_10' in df.columns:
        ax1.plot(df.index, df['SMA_5'], color='blue', linewidth=1, label='SMA 5')
        ax1.plot(df.index, df['SMA_10'], color='red', linewidth=1, label='SMA 10')
    
    if 'SMA_20' in df.columns:
        ax1.plot(df.index, df['SMA_20'], color='green', linewidth=1, label='SMA 20')
    
    if 'SMA_50' in df.columns:
        ax1.plot(df.index, df['SMA_50'], color='purple', linewidth=1, label='SMA 50')
    
    # Plot volume as a secondary axis if available
    if 'volume' in df.columns:
        ax2 = ax1.twinx()
        ax2.bar(df.index, df['volume'], color='gray', alpha=0.3, width=0.8)
        ax2.set_ylabel('Volume')
        ax2.grid(False)
    
    # Highlight patterns if requested
    if show_patterns:
        highlight_patterns(ax1, df, patterns)
    
    # Show buy/sell signals if requested
    if show_signals:
        plot_signals(ax1, df, patterns)
    
    # Format the x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Set the number of x-ticks based on the data size
    if len(df) > 50:
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    else:
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.xticks(rotation=45)
    
    # Add grid, legend, and labels
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_candlestick_chart(ax, df: pd.DataFrame) -> None:
    """
    Plot a candlestick chart on the given axis.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with OHLC data
    """
    # Make sure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Calculate width of candlestick body
    width = 0.6
    
    # Create candlesticks
    for i, (idx, row) in enumerate(df.iterrows()):
        # Determine if it's an up or down day
        is_up = row['close'] >= row['open']
        
        # Set colors based on up/down
        color = 'green' if is_up else 'red'
        
        # Plot the wick (high to low)
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
        
        # Plot the body (open to close)
        bottom = row['open'] if is_up else row['close']
        height = row['close'] - row['open'] if is_up else row['open'] - row['close']
        rect = patches.Rectangle((idx - width/2, bottom), width, height, 
                                 edgecolor=color, facecolor=color, alpha=0.8)
        ax.add_patch(rect)
    
    # Set x-axis ticks and labels to use the datetime index
    ax.set_xticks(range(0, len(df), max(1, len(df) // 10)))
    ax.set_xticklabels([d.strftime('%Y-%m-%d %H:%M') for d in df.index[::max(1, len(df) // 10)]], rotation=45)

def highlight_patterns(ax, df: pd.DataFrame, patterns: Dict) -> None:
    """
    Highlight detected patterns on the chart.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with price data
        patterns: Dictionary of detected patterns
    """
    # Define colors for different pattern types
    pattern_colors = {
        'bullish': 'green',
        'bearish': 'red',
        'neutral': 'blue'
    }
    
    # Track vertical position for text annotations to avoid overlap
    text_positions = {}
    
    # Iterate through patterns
    for pattern_name, pattern_data in patterns.items():
        if pattern_data.get('detected', False):
            # Get pattern signal and color
            signal = pattern_data.get('signal', 'neutral')
            color = pattern_colors.get(signal, 'blue')
            
            # Get the last few candles where the pattern might be visible
            last_idx = len(df) - 1
            start_idx = max(0, last_idx - 10)  # Look at last 10 candles by default
            
            # Adjust start index based on pattern type
            if 'head_and_shoulders' in pattern_name or 'triple' in pattern_name:
                start_idx = max(0, last_idx - 20)
            elif 'cup_and_handle' in pattern_name:
                start_idx = max(0, last_idx - 30)
            
            # Create a rectangle to highlight the pattern area
            rect = patches.Rectangle((start_idx, df['low'].iloc[start_idx:].min() * 0.99), 
                                    last_idx - start_idx, 
                                    df['high'].iloc[start_idx:].max() * 1.01 - df['low'].iloc[start_idx:].min() * 0.99,
                                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.1)
            ax.add_patch(rect)
            
            # Add text annotation for the pattern
            confidence = pattern_data.get('confidence', 0.5)
            pattern_display_name = pattern_name.replace('_', ' ').title()
            
            # Determine vertical position to avoid overlap
            if last_idx in text_positions:
                text_positions[last_idx] += 0.03  # Increment position
            else:
                text_positions[last_idx] = 0.03  # Initial position
            
            y_pos = df['high'].iloc[start_idx:].max() * (1.01 + text_positions[last_idx])
            
            ax.annotate(f"{pattern_display_name} ({confidence:.2f})", 
                       xy=(last_idx, df['high'].iloc[last_idx]),
                       xytext=(last_idx, y_pos),
                       color=color, fontweight='bold',
                       arrowprops=dict(facecolor=color, shrink=0.05, alpha=0.7))
            
            # Draw additional pattern-specific elements
            if 'neckline' in pattern_data:
                ax.axhline(y=pattern_data['neckline'], color=color, linestyle='--', alpha=0.7)
            
            if 'support' in pattern_data and 'resistance' in pattern_data:
                ax.axhline(y=pattern_data['support'], color='green', linestyle='--', alpha=0.7)
                ax.axhline(y=pattern_data['resistance'], color='red', linestyle='--', alpha=0.7)
            
            if 'upper_trendline' in pattern_data and 'lower_trendline' in pattern_data:
                # Draw trendlines
                x = np.arange(start_idx, last_idx + 1)
                
                # For wedges and triangles, we need to calculate the trendline values
                if 'wedge' in pattern_name or 'triangle' in pattern_name:
                    # Simple linear interpolation between start and end points
                    upper_start = pattern_data.get('upper_trendline_start', pattern_data['upper_trendline'])
                    upper_end = pattern_data['upper_trendline']
                    lower_start = pattern_data.get('lower_trendline_start', pattern_data['lower_trendline'])
                    lower_end = pattern_data['lower_trendline']
                    
                    upper_line = np.linspace(upper_start, upper_end, len(x))
                    lower_line = np.linspace(lower_start, lower_end, len(x))
                else:
                    # Horizontal lines
                    upper_line = np.ones(len(x)) * pattern_data['upper_trendline']
                    lower_line = np.ones(len(x)) * pattern_data['lower_trendline']
                
                ax.plot(x, upper_line, color=color, linestyle='--', alpha=0.7)
                ax.plot(x, lower_line, color=color, linestyle='--', alpha=0.7)

def plot_signals(ax, df: pd.DataFrame, patterns: Dict) -> None:
    """
    Plot buy/sell signals on the chart based on detected patterns.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with price data
        patterns: Dictionary of detected patterns
    """
    # Determine overall signal based on patterns
    bullish_count = 0
    bearish_count = 0
    
    for pattern_name, pattern_data in patterns.items():
        if pattern_data.get('detected', False):
            signal = pattern_data.get('signal', 'neutral')
            confidence = pattern_data.get('confidence', 0.5)
            
            if signal == 'bullish':
                bullish_count += confidence
            elif signal == 'bearish':
                bearish_count += confidence
    
    # Plot the signal
    last_idx = len(df) - 1
    last_price = df['close'].iloc[-1]
    
    if bullish_count > bearish_count and bullish_count > 0.5:
        # Buy signal
        ax.scatter(last_idx, last_price * 0.98, marker='^', color='green', s=200, alpha=0.8)
        ax.annotate('BUY', xy=(last_idx, last_price * 0.98), xytext=(last_idx, last_price * 0.95),
                   color='green', fontweight='bold', ha='center',
                   arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7))
    
    elif bearish_count > bullish_count and bearish_count > 0.5:
        # Sell signal
        ax.scatter(last_idx, last_price * 1.02, marker='v', color='red', s=200, alpha=0.8)
        ax.annotate('SELL', xy=(last_idx, last_price * 1.02), xytext=(last_idx, last_price * 1.05),
                   color='red', fontweight='bold', ha='center',
                   arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7))
    else:
        # Neutral signal
        ax.scatter(last_idx, last_price, marker='o', color='gray', s=200, alpha=0.8)
        ax.annotate('HOLD', xy=(last_idx, last_price), xytext=(last_idx, last_price * 1.03),
                   color='gray', fontweight='bold', ha='center',
                   arrowprops=dict(facecolor='gray', shrink=0.05, alpha=0.7))

def plot_deriv_chart(df: pd.DataFrame, symbol: str = 'R_75', save_path: str = 'static/simple_chart.png'):
    """
    Plot a simple chart for Deriv data with basic indicators.
    
    Args:
        df: DataFrame with price data
        symbol: Symbol name for the title
        save_path: Path to save the chart image
    """
    plt.figure(figsize=(12, 6))
    
    # Plot price
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    
    # Plot SMAs if available
    if 'SMA_5' in df.columns:
        plt.plot(df.index, df['SMA_5'], label='SMA 5', color='orange')
    if 'SMA_10' in df.columns:
        plt.plot(df.index, df['SMA_10'], label='SMA 10', color='red')
    if 'SMA_20' in df.columns:
        plt.plot(df.index, df['SMA_20'], label='SMA 20', color='green')
    if 'SMA_50' in df.columns:
        plt.plot(df.index, df['SMA_50'], label='SMA 50', color='purple')
    
    # Format the chart
    plt.title(f'{symbol} - Price Chart')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
