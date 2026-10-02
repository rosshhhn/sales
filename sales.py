"""
Retail Sales Analytics & Customer Insights
Tools: Python, Pandas, NumPy, SQL (sqlite3)

Run:  python retail_sales_analysis.py
Install once:  pip install pandas numpy matplotlib
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_FILE = Path("retail_sales_raw.csv")
CLEAN_FILE = Path("retail_sales_clean.csv")


# ----------------------------------------------------------------------
# STEP 1: Create a raw (messy) dataset if you don't already have one.
# If you have your own CSV, name it retail_sales_raw.csv and this step is skipped.
# ----------------------------------------------------------------------
def create_raw_dataset():
    rng = np.random.default_rng(42)

    products = pd.DataFrame({
        "product_id": range(1, 13),
        "product_name": ["Laptop", "Smartphone", "Headphones", "Smart Watch",
                         "T-Shirt", "Jeans", "Sneakers", "Jacket",
                         "Rice 5kg", "Cooking Oil", "Coffee Pack", "Snacks Box"],
        "category": ["Electronics"] * 4 + ["Clothing"] * 4 + ["Grocery"] * 4,
        "unit_price": [55000, 22000, 2500, 6000, 600, 1800, 3200, 4500,
                       450, 180, 320, 150],
    })

    cities = ["Calicut", "Kochi", "Trivandrum", "Bangalore", "Chennai"]
    customers = pd.DataFrame({
        "customer_id": range(1, 51),
        "customer_name": [f"Customer {i}" for i in range(1, 51)],
        "city": rng.choice(cities, 50),
    })

    n = 500
    orders = pd.DataFrame({
        "order_id": range(1001, 1001 + n),
        "customer_id": rng.integers(1, 51, n),
        "product_id": rng.integers(1, 13, n),
        "quantity": rng.integers(1, 11, n).astype(float),
        "order_date": pd.to_datetime("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    })

    df = orders.merge(customers, on="customer_id").merge(products, on="product_id")

    # Make it messy on purpose
    df.loc[rng.choice(df.index, 15, replace=False), "quantity"] = np.nan
    df.loc[rng.choice(df.index, 10, replace=False), "city"] = None
    upper_idx = rng.choice(df.index, 40, replace=False)
    df.loc[upper_idx, "city"] = df.loc[upper_idx, "city"].str.upper()
    df.loc[rng.choice(df.index, 20, replace=False), "category"] = " electronics "

    # Mixed date formats
    fmt_mask = rng.random(len(df)) < 0.5
    df["order_date"] = np.where(
        fmt_mask,
        df["order_date"].dt.strftime("%d/%m/%Y"),
        df["order_date"].dt.strftime("%Y-%m-%d"),
    )

    # Duplicate rows
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)
    df.to_csv(RAW_FILE, index=False)
    print(f"Created sample raw dataset: {RAW_FILE}")


# ----------------------------------------------------------------------
# STEP 2: Data cleaning (Pandas) - same tasks as the Excel stage
# ----------------------------------------------------------------------
def clean_data(df):
    print("\n=== DATA CLEANING ===")
    print("Raw shape:", df.shape)
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("Duplicate rows:", df.duplicated().sum())

    # Remove duplicates
    df = df.drop_duplicates().copy()

    # Standardise text
    for col in ["city", "category", "product_name", "customer_name"]:
        df[col] = df[col].astype("string").str.strip().str.title()

    # Standardise dates (handles both 2025-03-14 and 14/03/2025)
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=True)

    # Missing quantity -> median of that product
    df["quantity"] = df["quantity"].fillna(
        df.groupby("product_id")["quantity"].transform("median")
    )
    df["quantity"] = df["quantity"].round().astype(int)

    # Missing city -> use same customer's other rows, else "Unknown"
    df["city"] = df.groupby("customer_id")["city"].transform(lambda s: s.ffill().bfill())
    df["city"] = df["city"].fillna("Unknown")

    # Calculated columns (like Excel formulas)
    df["sales_amount"] = df["quantity"] * df["unit_price"]
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    # NumPy: IF-style order size label
    df["order_size"] = np.select(
        [df["sales_amount"] < 1000, df["sales_amount"] < 10000],
        ["Small", "Medium"],
        default="Large",
    )

    df = df.sort_values("order_date").reset_index(drop=True)
    df.to_csv(CLEAN_FILE, index=False)
    print("Clean shape:", df.shape)
    print(f"Saved cleaned data to {CLEAN_FILE}")
    return df


# ----------------------------------------------------------------------
# STEP 3: Analysis with Pandas + NumPy
# ----------------------------------------------------------------------
def analyse_data(df):
    print("\n=== PANDAS / NUMPY ANALYSIS ===")

    total = df["sales_amount"].sum()
    print(f"Total revenue      : {total:,.0f}")
    print(f"Total orders       : {df['order_id'].nunique()}")
    print(f"Average order value: {df['sales_amount'].mean():,.2f}")
    print(f"Median order value : {np.median(df['sales_amount']):,.2f}")
    print(f"Std deviation      : {np.std(df['sales_amount']):,.2f}")
    print(f"90th percentile    : {np.percentile(df['sales_amount'], 90):,.2f}")

    by_category = df.groupby("category")["sales_amount"].sum().sort_values(ascending=False)
    print("\nRevenue by category:\n", by_category.round(0))

    top_products = (df.groupby("product_name")["sales_amount"].sum()
                    .sort_values(ascending=False).head(5))
    print("\nTop 5 products:\n", top_products.round(0))

    by_city = df.groupby("city")["sales_amount"].sum().sort_values(ascending=False)
    print("\nRevenue by city:\n", by_city.round(0))

    monthly = df.groupby("month")["sales_amount"].sum()
    print("\nMonthly revenue:\n", monthly.round(0))

    top_customers = (df.groupby(["customer_id", "customer_name"])["sales_amount"]
                     .sum().sort_values(ascending=False).head(5))
    print("\nTop 5 customers:\n", top_customers.round(0))

    # Pivot table (like Excel PivotTable)
    pivot = pd.pivot_table(df, values="sales_amount", index="city",
                           columns="category", aggfunc="sum", fill_value=0)
    print("\nPivot: City x Category\n", pivot.round(0))

    return by_category, monthly, by_city


# ----------------------------------------------------------------------
# STEP 4: SQL queries (sqlite3) - same data in related tables
# ----------------------------------------------------------------------
def run_sql(df):
    print("\n=== SQL ANALYSIS ===")
    conn = sqlite3.connect(":memory:")

    customers = df[["customer_id", "customer_name", "city"]].drop_duplicates("customer_id")
    products = df[["product_id", "product_name", "category", "unit_price"]].drop_duplicates("product_id")
    orders = df[["order_id", "customer_id", "product_id", "quantity", "order_date"]].copy()
    orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")

    customers.to_sql("customers", conn, index=False)
    products.to_sql("products", conn, index=False)
    orders.to_sql("orders", conn, index=False)

    queries = {
        "1. Orders with quantity above 8 (WHERE + ORDER BY)": """
            SELECT order_id, customer_id, product_id, quantity
            FROM orders
            WHERE quantity > 8
            ORDER BY quantity DESC
            LIMIT 5;
        """,
        "2. Revenue by category (JOIN + GROUP BY + SUM)": """
            SELECT p.category,
                   COUNT(*)                    AS total_orders,
                   SUM(o.quantity * p.unit_price) AS revenue
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            GROUP BY p.category
            ORDER BY revenue DESC;
        """,
        "3. Average, min, max order value per city": """
            SELECT c.city,
                   ROUND(AVG(o.quantity * p.unit_price), 2) AS avg_order,
                   MIN(o.quantity * p.unit_price)           AS min_order,
                   MAX(o.quantity * p.unit_price)           AS max_order
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN products  p ON o.product_id  = p.product_id
            GROUP BY c.city;
        """,
        "4. Customers spending over 150,000 (HAVING)": """
            SELECT c.customer_name,
                   SUM(o.quantity * p.unit_price) AS total_spent
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN products  p ON o.product_id  = p.product_id
            GROUP BY c.customer_id
            HAVING total_spent > 150000
            ORDER BY total_spent DESC
            LIMIT 10;
        """,
    }

    for title, sql in queries.items():
        print(f"\n{title}")
        print(pd.read_sql_query(sql, conn).to_string(index=False))

    conn.close()


# ----------------------------------------------------------------------
# STEP 5: Charts
# ----------------------------------------------------------------------
def make_charts(by_category, monthly, by_city):
    by_category.plot(kind="bar", title="Revenue by Category", color="steelblue")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig("chart_category.png")
    plt.close()

    monthly.plot(kind="line", marker="o", title="Monthly Revenue Trend")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("chart_monthly.png")
    plt.close()

    by_city.plot(kind="bar", title="Revenue by City", color="seagreen")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig("chart_city.png")
    plt.close()

    print("\nSaved charts: chart_category.png, chart_monthly.png, chart_city.png")


def main():
    if not RAW_FILE.exists():
        create_raw_dataset()

    raw = pd.read_csv(RAW_FILE)
    clean = clean_data(raw)
    by_category, monthly, by_city = analyse_data(clean)
    run_sql(clean)
    make_charts(by_category, monthly, by_city)
    print("\nDone.")


if __name__ == "__main__":
    main()
