import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import numpy as np

def run_csv_backtest(trade_csv_path, symbol, start_date, end_date, initial_cash, output_dir):
    try:
        # Download price data
        print(f"[INFO] Downloading price data for {symbol}...")
        raw_data = yf.download(symbol, start=start_date, end=end_date)
        if raw_data.empty:
            print("[ERROR] Unable to download price data")
            return
        
        print(f"[DEBUG] Raw data shape: {raw_data.shape}")
        print(f"[DEBUG] Raw data columns: {raw_data.columns}")
        
        # Handle different yfinance return formats
        if isinstance(raw_data.columns, pd.MultiIndex):
            print("[DEBUG] Multi-level columns detected")
            print(f"[DEBUG] Column level names: {raw_data.columns.names}")
            
            # Try to find Close in different levels
            if 'Close' in raw_data.columns.get_level_values(0):
                # Close is in level 0 (Price level)
                close_data = raw_data.xs('Close', axis=1, level=0)
                print("[DEBUG] Found Close in level 0")
            elif 'Close' in raw_data.columns.get_level_values(1):
                # Close is in level 1 (Ticker level)
                close_data = raw_data.xs('Close', axis=1, level=1)
                print("[DEBUG] Found Close in level 1")
            else:
                # Fallback: look for Close column by flattening
                print("[DEBUG] Close not found in either level, trying flattened approach")
                close_cols = [col for col in raw_data.columns if 'Close' in str(col)]
                if close_cols:
                    close_data = raw_data[close_cols[0]]
                    print(f"[DEBUG] Using column: {close_cols[0]}")
                else:
                    print(f"[ERROR] No Close column found in: {raw_data.columns.tolist()}")
                    return
            
            if isinstance(close_data, pd.Series):
                price_df = close_data.to_frame(name='Close')
            else:
                # Multiple tickers case - take first column
                price_df = close_data.iloc[:, 0].to_frame(name='Close')
                
            print(f"[DEBUG] Extracted Close data shape: {price_df.shape}")
        else:
            # Single ticker case: columns are just field names
            print("[DEBUG] Single-level columns detected")
            if 'Close' in raw_data.columns:
                price_df = raw_data[['Close']].copy()
            else:
                print(f"[ERROR] 'Close' column not found. Available: {raw_data.columns.tolist()}")
                return
        
        # Clean up price data
        price_df.index = pd.to_datetime(price_df.index)
        price_df.index.name = "date"
        price_df = price_df.dropna()  # Remove any NaN values
        
        print(f"[INFO] Downloaded {len(price_df)} valid price records")
        print(f"[DEBUG] Price data date range: {price_df.index.min()} to {price_df.index.max()}")

        # Load trade instructions
        print(f"[INFO] Loading trades from {trade_csv_path}...")
        if not os.path.exists(trade_csv_path):
            print(f"[ERROR] Trade CSV file not found: {trade_csv_path}")
            return
            
        # Load CSV without parsing dates first to handle different formats
        trades = pd.read_csv(trade_csv_path)
        print(f"[DEBUG] Raw trades data preview:\n{trades.head()}")
        
        # Handle different date formats including Chinese dates
        def parse_chinese_date(date_str):
            """Parse Chinese date format like '2024年6月15日'"""
            try:
                # Try standard parsing first
                return pd.to_datetime(date_str)
            except:
                try:
                    # Handle Chinese format: 2024年6月15日
                    import re
                    if '年' in str(date_str) and '月' in str(date_str) and '日' in str(date_str):
                        # Extract numbers using regex
                        match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', str(date_str))
                        if match:
                            year, month, day = match.groups()
                            return pd.to_datetime(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
                    
                    # Try other common formats
                    return pd.to_datetime(date_str, infer_datetime_format=True)
                except:
                    print(f"[WARNING] Could not parse date: {date_str}")
                    return pd.NaT
        
        # Apply custom date parsing
        print("[DEBUG] Parsing dates...")
        trades['date'] = trades['date'].apply(parse_chinese_date)
        
        # Remove any rows with unparseable dates
        original_len = len(trades)
        trades = trades.dropna(subset=['date'])
        if len(trades) < original_len:
            print(f"[WARNING] Removed {original_len - len(trades)} rows with unparseable dates")
        
        trades = trades.set_index('date')
        
        print(f"[INFO] Loaded {len(trades)} trade records")
        print(f"[DEBUG] Trades date range: {trades.index.min()} to {trades.index.max()}")
        print(f"[DEBUG] Trade columns: {trades.columns.tolist()}")
        
        # Standardize column names - look for action column (case insensitive)
        action_col = None
        for col in trades.columns:
            if col.lower() == 'action':
                action_col = col
                break
        
        if action_col is None:
            print("[ERROR] No 'action' column found in trades CSV")
            print(f"[DEBUG] Available columns: {trades.columns.tolist()}")
            return
        
        # Rename to standard 'Action' if needed
        if action_col != 'Action':
            trades = trades.rename(columns={action_col: 'Action'})
            print(f"[DEBUG] Renamed column '{action_col}' to 'Action'")
            
        # Validate trade data
        valid_actions = ['BUY', 'SELL', 'HOLD']
        invalid_actions = trades[~trades['Action'].isin(valid_actions)]
        if not invalid_actions.empty:
            print(f"[WARNING] Found {len(invalid_actions)} invalid actions:")
            print(invalid_actions['Action'].value_counts())

        # Create a complete date range and merge data step by step
        print("[DEBUG] Merging price and trade data...")
        
        # Start with price data as base
        combined_df = price_df.copy()
        
        # Add trade actions, filling missing with 'HOLD'
        combined_df = combined_df.join(trades[['Action']], how='left')
        combined_df['Action'] = combined_df['Action'].fillna('HOLD')
        
        print(f"[INFO] Combined dataset has {len(combined_df)} records")
        print(f"[DEBUG] Action distribution: {combined_df['Action'].value_counts().to_dict()}")

        # Initialize portfolio
        cash = float(initial_cash)
        position = 0.0
        portfolio_values = []

        print(f"[INFO] Starting backtest with ${initial_cash:,.2f}")
        print("-" * 80)

        trade_count = 0
        for date, row in combined_df.iterrows():
            action = row['Action']
            price = float(row['Close'])
            
            # Execute trades
            if action == 'BUY' and cash > 0:
                shares_to_buy = cash / price
                position += shares_to_buy
                cash = 0.0
                trade_count += 1
                print(f"[TRADE {trade_count}] {date.date()} | BUY | Price: ${price:.2f} | Shares: {shares_to_buy:.6f}")
            elif action == 'SELL' and position > 0:
                cash_from_sale = position * price
                cash += cash_from_sale
                position = 0.0
                trade_count += 1
                print(f"[TRADE {trade_count}] {date.date()} | SELL | Price: ${price:.2f} | Cash: ${cash_from_sale:.2f}")
            
            # Calculate portfolio value
            portfolio_value = cash + position * price
            
            portfolio_values.append({
                'date': date,
                'price': price,
                'action': action,
                'cash': cash,
                'position': position,
                'portfolio_value': portfolio_value
            })

        if not portfolio_values:
            print("[ERROR] No portfolio values calculated")
            return

        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_values)
        results_df.set_index('date', inplace=True)

        # Calculate performance metrics
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        # Calculate buy and hold return for comparison
        first_price = results_df['price'].iloc[0]
        last_price = results_df['price'].iloc[-1]
        buy_hold_return = (last_price - first_price) / first_price * 100
        
        # Calculate additional metrics
        returns = results_df['portfolio_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        max_drawdown = ((results_df['portfolio_value'] / results_df['portfolio_value'].cummax()) - 1).min() * 100

        print("-" * 80)
        print(f"[RESULTS] Initial Value: ${initial_cash:,.2f}")
        print(f"[RESULTS] Final Value: ${final_value:,.2f}")
        print(f"[RESULTS] Total Return: {total_return:.2f}%")
        print(f"[RESULTS] Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"[RESULTS] Annualized Volatility: {volatility:.2f}%")
        print(f"[RESULTS] Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"[RESULTS] Total Trades: {trade_count}")

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, 'backtest_result.csv')
        results_df.to_csv(result_path)
        print(f"[INFO] Backtest results saved to: {result_path}")

        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Portfolio value comparison
        plt.subplot(3, 1, 1)
        plt.plot(results_df.index, results_df['portfolio_value'], label='Strategy', linewidth=2, color='blue')
        
        # Calculate buy & hold portfolio value
        buy_hold_value = initial_cash * (results_df['price'] / first_price)
        plt.plot(results_df.index, buy_hold_value, label='Buy & Hold', linestyle='--', alpha=0.7, color='orange')
        
        plt.title(f"Portfolio Performance Comparison: {symbol}")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Price and trading signals
        plt.subplot(3, 1, 2)
        plt.plot(results_df.index, results_df['price'], label='Price', color='black', alpha=0.7)
        
        # Mark buy/sell points
        buy_points = results_df[results_df['action'] == 'BUY']
        sell_points = results_df[results_df['action'] == 'SELL']
        
        if not buy_points.empty:
            plt.scatter(buy_points.index, buy_points['price'], 
                       color='green', marker='^', s=100, label=f'Buy ({len(buy_points)})', zorder=5)
        if not sell_points.empty:
            plt.scatter(sell_points.index, sell_points['price'], 
                       color='red', marker='v', s=100, label=f'Sell ({len(sell_points)})', zorder=5)
        
        plt.title(f"{symbol} Price and Trading Signals")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cash and position over time
        plt.subplot(3, 1, 3)
        plt.plot(results_df.index, results_df['cash'], label='Cash', color='green', alpha=0.7)
        plt.plot(results_df.index, results_df['position'] * results_df['price'], 
                label='Stock Value', color='red', alpha=0.7)
        
        plt.title("Portfolio Composition Over Time")
        plt.ylabel("Value ($)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'backtest_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Backtest chart saved to: {plot_path}")
        plt.close()

        # Save comprehensive summary statistics
        summary_stats = {
            'Symbol': symbol,
            'Start Date': start_date,
            'End Date': end_date,
            'Initial Cash': f"${initial_cash:,.2f}",
            'Final Value': f"${final_value:,.2f}",
            'Total Return': f"{total_return:.2f}%",
            'Buy & Hold Return': f"{buy_hold_return:.2f}%",
            'Excess Return': f"{total_return - buy_hold_return:.2f}%",
            'Annualized Volatility': f"{volatility:.2f}%",
            'Maximum Drawdown': f"{max_drawdown:.2f}%",
            'Total Trades': trade_count,
            'Number of Days': len(results_df)
        }
        
        summary_path = os.path.join(output_dir, 'summary_stats.txt')
        with open(summary_path, 'w') as f:
            f.write("=== BACKTEST SUMMARY ===\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        print(f"[INFO] Summary statistics saved to: {summary_path}")
        
        return results_df

    except Exception as e:
        print(f"[ERROR] An error occurred during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a backtest from CSV trade signals")
    parser.add_argument("--trade-csv-path", required=True, 
                       help="Path to CSV file containing trade signals")
    parser.add_argument("--symbol", required=True,
                       help="Stock symbol to backtest (e.g., AAPL)")
    parser.add_argument("--start-date", required=True,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=10000,
                       help="Initial cash amount (default: 10000)")
    parser.add_argument("--output-dir", default="backtest_result",
                       help="Output directory for results (default: backtest_result)")

    args = parser.parse_args()

    result = run_csv_backtest(
        trade_csv_path=args.trade_csv_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        output_dir=args.output_dir
    )
    
    if result is not None:
        print(f"\n[SUCCESS] Backtest completed successfully!")
    else:
        print(f"\n[FAILURE] Backtest failed. Check error messages above.")

#运行：python cli/backtest_from_csv.py --trade-csv-path "C:\Users\lauri\GitHub\TradingAgents\backtest_decision_data\btc_july_back.csv" --symbol BTC-USD --start-date 2024-06-15 --end-date 2024-07-15 --initial-cash 10000

