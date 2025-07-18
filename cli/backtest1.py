
import typer
from agents.sma_agent import SMAAgent
from core.backtester import run_backtest
import matplotlib.pyplot as plt

app = typer.Typer()

@app.command()
def simulate(symbol: str = "ETH-USD", start: str = "2023-01-01", end: str = "2024-01-01"):
    agent = SMAAgent()
    pf_df = run_backtest(symbol, start, end, agent)
    
    pf_df.plot(title=f"Backtest for {symbol}")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()
