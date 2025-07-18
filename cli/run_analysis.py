from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import datetime
import csv
import os
import sys

# ✅ 加载 .env 文件
load_dotenv()

# ✅ 获取命令行参数
# 示例：python run_analysis.py NVDA,AAPL 2024-07-01 2024-07-04
symbols_arg = sys.argv[1] if len(sys.argv) > 1 else "BTC,ETH"
start_date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.date.today().isoformat()
end_date_str = sys.argv[3] if len(sys.argv) > 3 else start_date_str  # 默认只分析一天

MY_SYMBOLS = [sym.strip().upper() for sym in symbols_arg.split(",")]
start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

# ✅ 配置模型
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1-nano"
config["quick_think_llm"] = "gpt-4.1-nano"
config["max_debate_rounds"] = 1
config["online_tools"] = True

ta = TradingAgentsGraph(debug=True, config=config)
results = []

# ✅ 遍历日期和股票
current_date = start_date
while current_date <= end_date:
    analysis_date = current_date.isoformat()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for symbol in MY_SYMBOLS:
        try:
            print(f"🧠 正在分析 {symbol} 的 {analysis_date} ...")
            _, decision = ta.propagate(symbol, analysis_date)
            print(f"✅ {symbol} 决策: {decision}")
            results.append({
                "symbol": symbol,
                "analysis_date": analysis_date,
                "timestamp": timestamp,
                "decision": decision
            })
        except Exception as e:
            print(f"❌ 分析 {symbol} 失败: {e}")
            results.append({
                "symbol": symbol,
                "analysis_date": analysis_date,
                "timestamp": timestamp,
                "decision": f"ERROR: {e}"
            })
    current_date += datetime.timedelta(days=1)

# ✅ 保存 CSV（追加）
csv_path = os.path.join(os.path.dirname(__file__), "quickresults.csv")
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["symbol", "analysis_date", "timestamp", "decision"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n📁 分析完成，结果已保存至: {csv_path}")



# 默认今天： python cli/run_analysis.py
# 示例：python cli/run_analysis.py BTC,ETH, NVDA 2024-07-01 2024-07-04
