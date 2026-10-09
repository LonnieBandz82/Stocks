"""Core logic: indicators, scoring, risk sizing, backtests, walk-forward, alerts.
Not financial advice. Past performance does not predict future results."""
import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

import numpy as np
import pandas as pd

COST = 0.001  # 0.1% per position change


def load(ticker, period="5y"):
    import yfinance as yf
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


# ---------- indicators ----------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return line, line.ewm(span=9, adjust=False).mean()


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_SIG"] = macd(df["Close"])
    df["ATR"] = atr(df)
    df["VOL_RATIO"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["RET"] = df["Close"].pct_change()
    return df


# ---------- scoring & risk ----------
def score(df):
    s = pd.Series(0, index=df.index, dtype=float)
    s += ((df["Close"] > df["SMA50"]) & (df["SMA50"] > df["SMA200"])).astype(int)
    s += (df["MACD"] > df["MACD_SIG"]).astype(int)
    s += df["RSI"].between(50, 70).astype(int)
    s += ((df["VOL_RATIO"] > 1.5) & (df["RET"] > 0)).astype(int)
    s -= (df["RSI"] > 75).astype(int)
    s -= (df["Close"] < df["SMA200"]).astype(int)
    return s


def label(sc, thr=3):
    return "BUY" if sc >= thr else "WATCH" if sc == thr - 1 else "AVOID"


def position_size(price, atr_val, account, risk_pct, atr_mult=2.0):
    stop_dist = atr_mult * atr_val
    shares = int((account * risk_pct) // stop_dist) if stop_dist > 0 else 0
    return min(shares, int(account // price)), price - stop_dist


# ---------- backtests ----------
def sharpe(r):
    sd = r.std()
    return r.mean() / sd * np.sqrt(252) if sd and sd > 0 else -np.inf


def run_strategy(d, sc, thr, cost=COST):
    pos = (sc >= thr).astype(int).shift(1).fillna(0)  # trade next day: no lookahead
    trades = pos.diff().abs().fillna(0)
    return pos * d["RET"] - trades * cost, pos


def backtest(df, thr=3):
    d = df.dropna()
    strat, pos = run_strategy(d, score(d), thr)
    eq, bh = (1 + strat).cumprod(), (1 + d["RET"]).cumprod()
    return {"equity": eq, "buy_hold": bh, "return": eq.iloc[-1] - 1,
            "bh_return": bh.iloc[-1] - 1, "max_dd": (eq / eq.cummax() - 1).min(),
            "sharpe": sharpe(strat), "bh_sharpe": sharpe(d["RET"]),
            "time_in": pos.mean()}


def walk_forward(df, train=504, test=126, thresholds=(1, 2, 3, 4)):
    """Pick the best threshold on past data only, then trade it on the NEXT unseen
    window. Stitch the unseen windows together: a far more honest result."""
    d = df.dropna()
    if len(d) < train + test + 1:
        return None
    sc = score(d)
    rets = {t: run_strategy(d, sc, t)[0] for t in thresholds}
    oos, chosen = [], []
    for s in range(0, len(d) - train - test + 1, test):
        tr, te = slice(s, s + train), slice(s + train, s + train + test)
        best = max(thresholds, key=lambda t: sharpe(rets[t].iloc[tr]))
        oos.append(rets[best].iloc[te])
        chosen.append(best)
    oos = pd.concat(oos)
    bh = d["RET"].loc[oos.index]
    return {"equity": (1 + oos).cumprod(), "buy_hold": (1 + bh).cumprod(),
            "return": (1 + oos).prod() - 1, "bh_return": (1 + bh).prod() - 1,
            "sharpe": sharpe(oos), "bh_sharpe": sharpe(bh), "chosen": chosen}


# ---------- scanning ----------
def scan(tickers, account=10000, risk=0.01, period="5y", loader=load):
    rows = []
    for t in tickers:
        try:
            df = loader(t, period)
            if len(df) < 250:
                continue
            df = add_indicators(df)
            sc = score(df)
            last = df.iloc[-1]
            shares, stop = position_size(last["Close"], last["ATR"], account, risk)
            rows.append({"Ticker": t, "Price": round(last["Close"], 2),
                         "RSI": round(last["RSI"], 1), "Score": int(sc.iloc[-1]),
                         "Signal": label(int(sc.iloc[-1])), "Prev": label(int(sc.iloc[-2])),
                         "Shares": shares, "Stop": round(stop, 2)})
        except Exception as e:
            print(f"{t}: failed ({e})")
    return sorted(rows, key=lambda r: -r["Score"])


# ---------- alerts ----------
def send_telegram(text):
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20)
    return True


def send_email(subject, body):
    host, user, pw, to = (os.getenv(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"))
    if not all((host, user, pw, to)):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    with smtplib.SMTP_SSL(host, int(os.getenv("SMTP_PORT") or 465)) as s:
        s.login(user, pw)
        s.send_message(msg)
    return True
