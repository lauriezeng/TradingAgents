# cli/backtest_tradingagent.py

import typer
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

app = typer.Typer()

@app.command()
def run_backtest(
    symbol: str = typer.Option("ETH-USD"),
    start_date: str = typer.Option("2023-01-01"),
    end_date: str = typer.Option("2024-01-01"),
    initial_cash: float = typer.Option(10000),
    actions_file: str = typer.Option("outputs/trader_actions.csv")
):
    df = yf.download(symbol, start=start_date, end=end_date)[['Close']]
    df.index = df.index.date

    actions = pd.read_csv(actions_file, names=["Date", "Action"], parse_dates=["Date"])
    actions["Date"] = actions["Date"].dt.date
    actions = actions.set_index("Date")

    cash = initial_cash
    position = 0
    portfolio = []
    trades = []

    for date in df.index:
        if date not in actions.index:
            continue

        price = df.loc[date, 'Close']
        action = actions.loc[date, "Action"]

        if action == "BUY" and cash > 0:
            position = cash / price
            cash = 0
            trades.append((date, "BUY", price))
        elif action == "SELL" and position > 0:
            cash = position * price
            position = 0
            trades.append((date, "SELL", price))

        value = cash + position * price
        portfolio.append((date, value))

    pf_df = pd.DataFrame(portfolio, columns=["Date", "PortfolioValue"]).set_index("Date")
    os.makedirs("outputs", exist_ok=True)
    pf_df.to_csv("outputs/portfolio_value_tradingagent.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(pf_df.index, pf_df["PortfolioValue"], label="Portfolio Value")
    plt.title(f"Backtest using TradingAgents' Advice")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/tradingagent_backtest.png")
    plt.show()

    print("✅ 回测完成，已保存结果")

if __name__ == "__main__":
    app()
