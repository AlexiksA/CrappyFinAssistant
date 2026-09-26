import sqlite3 as sql
import pandas as pd
import re
from datetime import datetime as dt
import os

DEF_DATABASE = "data/hist.db"


def init():
    conn = sql.connect(DEF_DATABASE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS company (
            ticker TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            shares REAL NOT NULL,
            avg_buy REAL NOT NULL,
            currency TEXT NOT NULL,
            date TEXT NOT NULL
        ); 
        """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS signal (
            ticker TEXT,
            signal_type TEXT,
            last_value REAL,
            is_triggered INTEGER,
            last_update TEXT,
            PRIMARY KEY (ticker, signal_type)
        ); 
        """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            log TEXT
        );
        """)

    conn.commit()
    conn.close()


def get_cur_time():
    return dt.now().strftime("%d.%m.%Y %H:%M:%S")


def del_company():
    conn = sql.connect(DEF_DATABASE)
    conn.execute("DELETE FROM company")
    conn.commit()
    conn.close()


def upsert_market_data(file_name: str):
    df = pd.read_csv(file_name)
    df.drop("ISIN", axis=1)

    currency = file_name.split("_")[0]
    cur_date = get_cur_time()

    conn = sql.connect(DEF_DATABASE)
    cur = conn.cursor()

    for idx, row in df.iterrows():
        ticker = row["Symbol"]
        name = row["Company"]
        shares = row["Quantity"]
        avg_buy = re.sub(r"[^\d.]", "", str(row["Price"]))

        query = """
        INSERT INTO company (ticker, name, shares, avg_buy, currency, date)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker)
        DO UPDATE SET
            shares = excluded.shares,
            avg_buy = excluded.avg_buy,
            date = excluded.date
        """

        try:
            cur.execute(query, (ticker, name, shares, avg_buy, currency, cur_date))
            print(f"Row {idx}: Successfully added {ticker}")
        except sql.Error as e:
            print(f"Row {idx}: Error occured -> {e}")

    conn.commit()
    conn.close()
    print("Done!")


def prune_companies(
    keep_by_currency: dict[str, set[str]], ticker_map: dict | None = None
) -> list[str]:
    """Delete holdings that are no longer in the latest statement.

    keep_by_currency maps a currency to the tickers the statement listed
    for it; only currencies present in that dict are touched, so a table
    that failed to parse can never wipe that currency's holdings. Removed
    tickers also lose their rows in the signal table (keyed by the yfinance
    symbol, hence ticker_map), all in one transaction. Returns what was
    removed.
    """
    ticker_map = ticker_map or {}
    conn = sql.connect(DEF_DATABASE)
    removed = []
    try:
        for currency, keep in keep_by_currency.items():
            # an empty set almost certainly means a parse problem, not that
            # every position in this currency was sold - refuse to wipe it
            if not keep:
                print(f"prune_companies: no {currency} tickers parsed, not pruning {currency}")
                continue
            held = [
                row[0]
                for row in conn.execute(
                    "SELECT ticker FROM company WHERE currency = ?", (currency,)
                )
            ]
            for ticker in held:
                if ticker in keep:
                    continue
                conn.execute("DELETE FROM company WHERE ticker = ?", (ticker,))
                conn.execute(
                    "DELETE FROM signal WHERE ticker = ?",
                    (ticker_map.get(ticker, ticker),),
                )
                removed.append(ticker)
        conn.commit()
    finally:
        conn.close()

    for ticker in removed:
        print(f"prune_companies: removed {ticker} (no longer in the statement)")
    return removed


def sql_to_df(table_name: str, conn=None):
    df = None
    if conn is None:
        conn = sql.connect(DEF_DATABASE)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        if os.environ.get("DEBUG"):
            print("verbose db_man.py")
            if df.empty:
                print("Table is Empty bruh")
            else:
                print(df.to_string(index=False))
    except Exception as e:
        print(f"Dont know bruh -> {e}")
    return df


def print_all_tables():
    conn = sql.connect(DEF_DATABASE)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    print(f"Found {len(tables)} tables")

    for table_tuple in tables:
        table_name = table_tuple[0]

        print(table_name)
        print("-" * 50)
        sql_to_df(table_name=table_name, conn=conn)

    print("_" * 50)


if __name__ == "__main__":
    # init()
    # del_company()
    # upsert_market_data("EUR_output.csv")
    # upsert_market_data("USD_output.csv")
    print_all_tables()
