import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("warehouse/raw_data")
OUT_SPEND = RAW_DIR / "marketing_spend.csv"
OUT_FUNNEL = RAW_DIR / "funnel.csv"

ORDERS_FILE = RAW_DIR / "olist_orders_dataset.csv"

CHANNELS = ["Paid Search", "Paid Social", "Email", "Affiliate", "Organic"]
np.random.seed(42)

def main():
    if not ORDERS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {ORDERS_FILE}. Put Olist CSVs in warehouse/raw_data/ first."
        )

    orders = pd.read_csv(ORDERS_FILE)

    if "order_purchase_timestamp" not in orders.columns:
        raise ValueError(
            "Expected column 'order_purchase_timestamp' not found in olist_orders_dataset.csv"
        )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    orders = orders.dropna(subset=["order_purchase_timestamp"])

    min_date = orders["order_purchase_timestamp"].dt.date.min()
    max_date = orders["order_purchase_timestamp"].dt.date.max()
    all_dates = pd.date_range(min_date, max_date, freq="D")

    orders["order_date"] = orders["order_purchase_timestamp"].dt.floor("D")
    purchases_daily = (
        orders.groupby("order_date")["order_id"]
        .nunique()
        .reindex(all_dates, fill_value=0)
        .reset_index()
    )
    purchases_daily.columns = ["date", "total_purchases"]

    channel_weights = {
        "Paid Search": 0.28,
        "Paid Social": 0.22,
        "Email": 0.15,
        "Affiliate": 0.10,
        "Organic": 0.25,
    }

    funnel_rows = []
    spend_rows = []

    for _, row in purchases_daily.iterrows():
        d = row["date"]
        total_purchases = int(row["total_purchases"])

        weekday = pd.Timestamp(d).weekday()
        weekend_factor = 0.85 if weekday >= 5 else 1.0

        allocations = {}
        remaining = total_purchases
        channels = list(channel_weights.keys())

        for i, ch in enumerate(channels):
            if i < len(channels) - 1:
                val = int(round(total_purchases * channel_weights[ch]))
                allocations[ch] = val
                remaining -= val
            else:
                allocations[ch] = max(0, remaining)

        for ch in CHANNELS:
            purchases = allocations.get(ch, 0)

            lead_rate = np.random.uniform(0.02, 0.08)
            close_rate = np.random.uniform(0.05, 0.18)

            leads = int(
                np.ceil(
                    (purchases / max(close_rate, 0.01))
                    * np.random.uniform(0.95, 1.10)
                )
            )
            sessions = int(
                np.ceil(
                    (leads / max(lead_rate, 0.01))
                    * np.random.uniform(0.95, 1.15)
                )
            )

            sessions = int(sessions * weekend_factor)
            leads = int(leads * weekend_factor)

            funnel_rows.append(
                {
                    "funnel_date": pd.Timestamp(d).date().isoformat(),
                    "channel": ch,
                    "sessions": max(sessions, 0),
                    "leads": max(leads, 0),
                    "purchases": max(purchases, 0),
                }
            )

            if ch in ["Paid Search", "Paid Social", "Affiliate"]:
                cpc = np.random.uniform(0.5, 2.5)
                spend = sessions * cpc * np.random.uniform(0.6, 1.2)
            elif ch == "Email":
                spend = np.random.uniform(20, 120)
            else:
                spend = 0.0

            spend = spend * weekend_factor

            spend_rows.append(
                {
                    "spend_date": pd.Timestamp(d).date().isoformat(),
                    "channel": ch,
                    "spend": round(float(max(spend, 0.0)), 2),
                }
            )

    funnel_df = pd.DataFrame(funnel_rows)
    spend_df = pd.DataFrame(spend_rows)

    funnel_df.to_csv(OUT_FUNNEL, index=False)
    spend_df.to_csv(OUT_SPEND, index=False)

    print("Generated:")
    print(f" - {OUT_FUNNEL} rows: {len(funnel_df):,}")
    print(f" - {OUT_SPEND} rows: {len(spend_df):,}")
    print(f"Date range: {min_date} to {max_date}")
    print("Channels:", ", ".join(CHANNELS))

if __name__ == "__main__":
    main()
