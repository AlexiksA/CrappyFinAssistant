import yfinance as yf
from db_man import sql_to_df
import pandas as pd
import sys
import os
import sqlite3 as sql
from datetime import datetime

DEF_MAP = {"TOTB": "TOTB.F", "AEU": "AEU.F", "VUAA": "VUAA.DE"}


def get_history(table_name: str, stock_map=None):
    if stock_map is None:
        print("No map supplied... Using default map")
        stock_map = DEF_MAP
    df = sql_to_df(table_name=table_name)
    if df is None:
        print("No df supplied")
        return None, None
    if os.environ.get("DEBUG"):
        for idx, row in df.iterrows():
            # print(f"Index: {idx}")
            print(
                f"Name: {row['name']}, avg buy price: {'€' if row['currency'] == 'EUR' else 'USD$'}{row['avg_buy']}"
            )

    company_list = df["ticker"].replace(stock_map).tolist()

    if os.environ.get("DEBUG"):
        print(company_list)

    all_hist = []
    for tick in company_list:
        try:
            print(f"Getting history for {tick}..")
            ticker_df = yf.Ticker(tick).history(period="1y", interval="1d")

            if not ticker_df.empty:
                ticker_df["ticker"] = tick
                ticker_df.reset_index(inplace=True)
                if os.environ.get("DEBUG"):
                    print(ticker_df["Date"])
                all_hist.append(ticker_df)
            else:
                print(f"No data found on {tick}")
        except Exception as e:
            print(f"Random error while getting ticker -> {e}")

    if all_hist:
        master_df = pd.concat(all_hist)
        print("Succesfully combined all data frames bruv")
        if master_df is None or company_list is None:
            return None, None
        else:
            return [master_df, company_list]
    else:
        print("muie")
        return None, None


# A simple moving average of the stock
def compute_sma(days: int, ticker: str, df=None):
    if df is None:
        print("No df supplied or df is None")
        return

    filtered_df = df.loc[df["ticker"] == ticker]
    if filtered_df is None:
        print("Ticker not found")
        return
    sorted_df = filtered_df.sort_values(by="Date", ascending=True)

    target_series = sorted_df["Close"].rolling(window=days).mean()

    if os.environ.get("DEBUG"):
        print(target_series.tail(-days))
    return target_series.tail(-days)


# Indicates potentially oversold or overbought stocks
def compute_rsi(ticker: str, df=None):
    if df is None:
        print("No df supplied or df is None")
        return

    filtered_df = df.loc[df["ticker"] == ticker]
    sorted_df = filtered_df.sort_values(by="Date", ascending=True)

    delta = sorted_df["Close"].diff()

    gain = delta.clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()

    loss = -delta.clip(upper=0)
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    if os.environ.get("DEBUG"):
        print(f"avg_gain\n{'-' * 50}\n{avg_gain}\navg_loss\n{'-' * 50}\n{avg_loss}")

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if os.environ.get("DEBUG"):
        print(f"RSI\n{rsi}")

    return rsi


# Volatility compute
def compute_boll_bands(ticker: str, df=None):
    if df is None:
        print("No df supplied or df is None")
        return None, None

    filtered_df = df.loc[df["ticker"] == ticker]
    sorted_df = filtered_df.sort_values(by="Date", ascending=True)

    avg_df = sorted_df["Close"].rolling(window=20).mean()

    std_dev = sorted_df["Close"].rolling(window=20).std()

    upper_band = avg_df + 2 * std_dev
    lower_band = avg_df - 2 * std_dev

    if os.environ.get("DEBUG"):
        print(f"STD_DEV is {std_dev}")
        print(
            f"upper_band: {upper_band.tail(1)} - middle_band: {avg_df.tail(1)} - lower_band: {lower_band.tail(1)}"
        )

    return (avg_df, std_dev)


# Bandwith - calculated based on boll_bands
def compute_bandwith(avg_df, std_dev):
    bandwith = 4 * std_dev / avg_df
    return bandwith


# Moving average convergence/divergence
def compute_MACD(ticker: str, df=None):
    if df is None:
        print("No df supplied or df is None")
        return None, None

    filtered_df = df.loc[df["ticker"] == ticker]
    sorted_df = filtered_df.sort_values(by="Date", ascending=True)

    fastEMA = sorted_df["Close"].ewm(span=12, adjust=False).mean()
    slowEMA = sorted_df["Close"].ewm(span=26, adjust=False).mean()

    MACD_line = fastEMA - slowEMA

    Signal_line = MACD_line.ewm(span=9, adjust=False).mean()
    return MACD_line, Signal_line


# MACD but percentage-wise
def compute_PPO(ticker: str, df=None):
    if df is None:
        print("No df supplied or df is None")
        return None, None

    filtered_df = df.loc[df["ticker"] == ticker]
    sorted_df = filtered_df.sort_values(by="Date", ascending=True)

    fastEMA = sorted_df["Close"].ewm(span=12, adjust=False).mean()
    slowEMA = sorted_df["Close"].ewm(span=26, adjust=False).mean()

    PPOline = (fastEMA - slowEMA) / slowEMA * 100
    SignalLine = PPOline.ewm(span=9, adjust=False).mean()

    return PPOline, SignalLine


# An initial list of signals
def is_golden_cross(short_sma, long_sma):
    if short_sma[1] > long_sma[1] and short_sma[0] < long_sma[0]:
        print("SMA crossover indicates a bullish trend")
        return True
    return False


def is_death_cross(short_sma, long_sma):
    if short_sma[1] < long_sma[1] and short_sma[0] > long_sma[0]:
        print("SMA crossover indicates a bearish trend")
        return True
    return False


def rsi_oversold(rsi):
    if rsi < 30:
        print("RSI indicates a potentially oversold stock")
        return True
    return False


def rsi_overbought(rsi):
    if rsi > 70:
        print("RSI indicated a potentially overbought stock")
        return True
    return False


def macd_bullish(macd_line, signal_line):
    if macd_line - signal_line > 0:
        print("MACD indicates bullish crossover")
        return True
    return False


def macd_bearish(macd_line, signal_line):
    if macd_line - signal_line < 0:
        print("MACD indicates a bearish crossover")
        return True
    return False


def macd_crossover_bull(macd_line_last, macd_line_latest):
    if macd_line_last < 0 and macd_line_latest > 0:
        print("MACD indicates trend shift to bullish")
        return True
    return False


def macd_crossover_bear(macd_line_last, macd_line_latest):
    if macd_line_last > 0 and macd_line_latest < 0:
        print("MACD indicates trend shift to bearish")
        return True
    return False


def boll_lower_band(last_val, avg_val, std_dev):
    if last_val < avg_val - 2 * std_dev:
        print("Unusually low price")
        return True
    return False


def boll_upper_band(last_val, avg_val, std_dev):
    if last_val > avg_val + 2 * std_dev:
        print("Unusually high price")
        return True
    return False


def is_squeeze(bandwidth):
    threshold = bandwidth.rolling(window=250).quantile(0.1)
    if bandwidth.iloc[-1] < threshold.iloc[-1]:
        print("Possible band squeeze detected, be careful")
        return True
    return False


def price_data_main():
    [df, company_list] = get_history("company", DEF_MAP)
    """
    if os.environ.get("DEBUG") and df is not None:
        print(df.info())

    print("Computing SMA for 20 days...")
    compute_sma(20, "AEU.F", df)

    compute_rsi("AEU.F", df)

    compute_boll_bands("AEU.F", df)

    macd, signal = compute_MACD("AEU.F", df)
    if macd is None or signal is None:
        sys.exit(1)
    print(macd - signal)

    ppo, signal = compute_PPO("AEU.F", df)
    if ppo is None or signal is None:
        sys.exit(1)
    print(ppo - signal)
    """
    if company_list is None:
        return

    for ticker in company_list:
        print("=" * 50)
        print(ticker)
        print("=" * 50)
        sma_20 = compute_sma(20, ticker, df)
        sma_200 = compute_sma(200, ticker, df)
        rsi = compute_rsi(ticker, df)
        avg_df, std_dev = compute_boll_bands(ticker, df)
        macd, signal = compute_MACD(ticker, df)
        last_close = (
            df.loc[df["ticker"] == ticker]
            .sort_values(by="Date", ascending=True)["Close"]
            .iloc[-1]
        )
        bandwidth = compute_bandwith(avg_df, std_dev)

        indicators = [sma_20, sma_200, rsi, avg_df, std_dev, macd, signal, bandwidth]

        if any(x is None for x in indicators):
            print(f"You goofed somewhere\nSkipping {ticker} due to missing data")
            continue

        print(f"SMA - 20 days\n\n{sma_20.iloc[-5:]}\n{'=' * 50}\n")  # type: ignore
        print(f"SMA - 200 days\n\n{sma_200.iloc[-5:]}\n{'=' * 50}\n")  # type: ignore
        print(f"RSI last 5 days\n\n{rsi.iloc[-5:]}\n{'=' * 50}\n")  # type: ignore
        print(
            f"Standard deviation: {std_dev.iloc[-1]}\n{avg_df.iloc[-5:]}\n{'=' * 50}\n"
        )  # type: ignore
        hist = macd - signal  # type: ignore
        print(f"histogram from macd - signal\n{hist.iloc[-5:]}\n{'=' * 50}\n")  # type: ignore

        is_golden_cross(
            (sma_20.iloc[-2], sma_20.iloc[-1]), (sma_200.iloc[-2], sma_200.iloc[-1])
        )  # type: ignore
        is_death_cross(
            (sma_20.iloc[-2], sma_20.iloc[-1]), (sma_200.iloc[-2], sma_200.iloc[-1])
        )  # type: ignore
        rsi_overbought(rsi.iloc[-1])  # type: ignore
        rsi_oversold(rsi.iloc[-1])  # type: ignore
        macd_bullish(macd.iloc[-1], signal.iloc[-1])  # type: ignore
        macd_bearish(macd.iloc[-1], signal.iloc[-1])  # type: ignore
        macd_crossover_bull(macd.iloc[-2], macd.iloc[-1])  # type: ignore
        macd_crossover_bear(macd.iloc[-2], macd.iloc[-1])  # type: ignore
        boll_lower_band(last_close, avg_df.iloc[-1], std_dev.iloc[-1])  # type: ignore
        boll_upper_band(last_close, avg_df.iloc[-1], std_dev.iloc[-1])  # type: ignore
        is_squeeze(bandwidth)  # type: ignore

        print("=" * 50)
        print("=" * 50)
        print("=" * 50)

    signal_df = build_signal_dataframe(df, company_list)
    insert_all_signals(signal_df, "data/hist.db")


# Builds a one-row-per-ticker snapshot of today's indicator values and signal outcomes
def build_signal_dataframe(df, company_list):
    rows = []
    for ticker in company_list:
        sma_20 = compute_sma(20, ticker, df)
        sma_200 = compute_sma(200, ticker, df)
        rsi = compute_rsi(ticker, df)
        avg_df, std_dev = compute_boll_bands(ticker, df)
        macd, signal = compute_MACD(ticker, df)
        bandwidth = compute_bandwith(avg_df, std_dev)
        last_close = (
            df.loc[df["ticker"] == ticker]
            .sort_values(by="Date", ascending=True)["Close"]
            .iloc[-1]
        )

        indicators = [sma_20, sma_200, rsi, avg_df, std_dev, macd, signal, bandwidth]
        if any(x is None for x in indicators):
            print(f"Skipping {ticker}, missing indicator data")
            continue

        rows.append(
            {
                "ticker": ticker,
                "price": last_close,
                "sma_20": sma_20.iloc[-1],
                "sma_200": sma_200.iloc[-1],
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "golden_cross": is_golden_cross(
                    (sma_20.iloc[-2], sma_20.iloc[-1]),
                    (sma_200.iloc[-2], sma_200.iloc[-1]),
                ),
                "death_cross": is_death_cross(
                    (sma_20.iloc[-2], sma_20.iloc[-1]),
                    (sma_200.iloc[-2], sma_200.iloc[-1]),
                ),
                "rsi_oversold": rsi_oversold(rsi.iloc[-1]),
                "rsi_overbought": rsi_overbought(rsi.iloc[-1]),
                "macd_bullish": macd_bullish(macd.iloc[-1], signal.iloc[-1]),
                "macd_bearish": macd_bearish(macd.iloc[-1], signal.iloc[-1]),
                "macd_crossover_bull": macd_crossover_bull(
                    macd.iloc[-2], macd.iloc[-1]
                ),
                "macd_crossover_bear": macd_crossover_bear(
                    macd.iloc[-2], macd.iloc[-1]
                ),
                "boll_lower_band": boll_lower_band(
                    last_close, avg_df.iloc[-1], std_dev.iloc[-1]
                ),
                "boll_upper_band": boll_upper_band(
                    last_close, avg_df.iloc[-1], std_dev.iloc[-1]
                ),
                "bandwidth": bandwidth.iloc[-1],
                "squeeze": is_squeeze(bandwidth),
            }
        )

    return pd.DataFrame(rows)


def insert_signal_record(
    ticker: str, signal_type: str, last_value, is_triggered, last_update: str, db: str
):
    conn = sql.connect(db)
    cur = conn.cursor()

    df_last = pd.read_sql_query(
        "SELECT * FROM signal WHERE ticker = ? AND signal_type = ?",
        conn,
        params=(ticker, signal_type),
    )

    if df_last.empty:
        if os.environ.get("DEBUG"):
            print(f"Adding new record to signal {ticker}, {signal_type}")

        query = """
            INSERT INTO signal(ticker, signal_type, last_value, is_triggered, last_update)
            VALUES(?, ?, ?, ?, ?)
                    """
        cur.execute(query, (ticker, signal_type, last_value, is_triggered, last_update))
    else:
        if os.environ.get("DEBUG"):
            print(
                f"Comparing new record stats with old record stats {ticker}, {signal_type}"
            )

        query = """
            UPDATE signal
            SET last_value = ?, is_triggered = ?, last_update = ?
            WHERE ticker = ? AND signal_type = ?
        """

        cur.execute(query, (last_value, is_triggered, last_update, ticker, signal_type))

    conn.commit()
    conn.close()


def insert_all_signals(df_signals: pd.DataFrame, db: str):
    last_update = datetime.now().isoformat()

    for _, row in df_signals.iterrows():
        sma_spread = row["sma_20"] - row["sma_200"]
        macd_hist = row["macd"] - row["macd_signal"]

        signals = {
            "golden_cross": (sma_spread, row["golden_cross"]),
            "death_cross": (sma_spread, row["death_cross"]),
            "rsi_oversold": (row["rsi"], row["rsi_oversold"]),
            "rsi_overbought": (row["rsi"], row["rsi_overbought"]),
            "macd_bullish": (macd_hist, row["macd_bullish"]),
            "macd_bearish": (macd_hist, row["macd_bearish"]),
            "macd_crossover_bull": (row["macd"], row["macd_crossover_bull"]),
            "macd_crossover_bear": (row["macd"], row["macd_crossover_bear"]),
            "boll_lower_band": (row["price"], row["boll_lower_band"]),
            "boll_upper_band": (row["price"], row["boll_upper_band"]),
            "squeeze": (row["bandwidth"], row["squeeze"]),
        }

        for signal_type, (last_value, is_triggered) in signals.items():
            insert_signal_record(
                row["ticker"],
                signal_type,
                float(last_value),
                bool(is_triggered),
                last_update,
                db,
            )


if __name__ == "__main__":
    price_data_main()
