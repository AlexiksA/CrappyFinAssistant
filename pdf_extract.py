import pdfplumber
import csv
import pandas as pd

DEF_FILE_NAME = "stats.pdf"


def csv_tidy(file_name: str):
    df = pd.read_csv(file_name)
    df = df.dropna()
    df = df.rename(columns={"Value % of": "Value"})
    df = df.drop("Portfolio", axis=1)
    df.to_csv(file_name, index=False)


def table_to_csv(table_name: str, file_name: str = DEF_FILE_NAME):

    with pdfplumber.open(file_name) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            target = table_name.split()
            which = target[0]

            match_start = None

            for i in range(len(words) - len(target) + 1):
                if [words[i + j]["text"] for j in range(len(target))] == target:
                    match_start = words[i]
                    break
            else:
                continue

            # heading = next(w for w in words if w["text"] == "breakdown")

            total_matches = (w for w in words if w["text"] == "Total")
            _ = next(total_matches, None)
            bound_text = next(total_matches, None)
            if bound_text is None:
                print("bound_text is None")
                return
            print(bound_text)
            cropped = page.within_bbox(
                (0, match_start["bottom"], page.width, bound_text["bottom"])
            )

            words = cropped.extract_words()
            positions_match = next(w for w in words if w["text"] == "Positions")

            cropped = cropped.within_bbox(
                (0, cropped.bbox[1], cropped.width, positions_match["top"])
            )
            tables = cropped.find_tables(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                }
            )

            im = cropped.to_image(resolution=150)
            im.debug_tablefinder(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                }
            )
            im.save(f"{which}_debug.png")

            target = tables[0]
            if target is not None:
                data = target.extract()
                with open(f"{which}_output.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
                return


CURRENCIES = ("EUR", "USD")


def import_statement(file_name: str = DEF_FILE_NAME) -> dict:
    """Parse the statement's portfolio tables into the company table, then
    drop holdings that no longer appear in it (sold since the last one).

    Pruning is per currency and only for currencies whose table was parsed
    this time - a table that fails to parse must never wipe its holdings.
    The flip side: selling *every* position in one currency removes that
    table from the statement entirely, so those rows stay until deleted by
    hand (or by db_man.prune_companies with an explicit set).
    """
    import os

    import db_man
    from price_data import DEF_MAP

    parsed: dict[str, str] = {}
    for currency in CURRENCIES:
        csv_path = f"{currency}_output.csv"
        before = os.path.getmtime(csv_path) if os.path.exists(csv_path) else None
        table_to_csv(f"{currency} Portfolio breakdown", file_name)
        # table_to_csv only writes when it finds the table; an old csv left
        # over from a previous statement must not count as this one's
        if not os.path.exists(csv_path) or os.path.getmtime(csv_path) == before:
            print(f"no {currency} table in {file_name} - leaving {currency} holdings as they are")
            continue
        csv_tidy(csv_path)
        parsed[currency] = csv_path

    if not parsed:
        raise RuntimeError(f"no portfolio table could be parsed from {file_name}")

    keep_by_currency = {}
    for currency, csv_path in parsed.items():
        db_man.upsert_market_data(csv_path)
        symbols = pd.read_csv(csv_path)["Symbol"].dropna().astype(str).str.strip()
        keep_by_currency[currency] = set(symbols)

    removed = db_man.prune_companies(keep_by_currency, ticker_map=DEF_MAP)
    return {"imported": list(parsed), "removed": removed}


if __name__ == "__main__":
    print(import_statement())
