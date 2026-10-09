"""Run daily (GitHub Actions). Alerts only when a signal ENTERS or LEAVES 'BUY'."""
import os

from stock_core import scan, send_email, send_telegram

DEFAULT = "AAPL,MSFT,NVDA,AMZN,GOOGL,TSLA,SPY"
tickers = [t.strip().upper() for t in (os.getenv("WATCHLIST") or DEFAULT).split(",") if t.strip()]
rows = scan(tickers, float(os.getenv("ACCOUNT") or 10000), float(os.getenv("RISK") or 0.01))

lines = []
for r in rows:
    if r["Signal"] == "BUY" and r["Prev"] != "BUY":
        lines.append(f"🟢 {r['Ticker']} new BUY @ {r['Price']} | stop {r['Stop']} | size {r['Shares']} sh")
    elif r["Prev"] == "BUY" and r["Signal"] != "BUY":
        lines.append(f"🔴 {r['Ticker']} dropped out of BUY @ {r['Price']}")

for r in rows:
    print(r)
if lines:
    msg = "Stock Edge alerts\n" + "\n".join(lines) + "\n\nNot financial advice."
    sent = [send_telegram(msg), send_email("Stock Edge alerts", msg)]
    print("Sent." if any(sent) else "Alerts found but no channel configured.")
else:
    print("No signal changes today.")
