import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# NASSAU CANDY - FINAL PYTHON ANALYSIS
# =========================================================

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("Nassau Candy Distributor.csv")

# Convert dates
df["Order Date"] = pd.to_datetime(
    df["Order Date"], format="%d-%m-%Y"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], format="%d-%m-%Y"
)

# Calculated fields
df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

df["Gross Margin"] = (
    df["Gross Profit"] / df["Sales"]
)

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month
df["Order Month Name"] = df["Order Date"].dt.strftime("%B")


# =========================================================
# 2. BASIC DATA QUALITY
# =========================================================

print("\n" + "=" * 70)
print("NASSAU CANDY - PYTHON DATA ANALYSIS")
print("=" * 70)

print("\n===== DATASET INFORMATION =====")
print("Total Records:", len(df))
print("Original Columns:", 18)
print("Analysis Columns:", len(df.columns))

print("\n===== DATA QUALITY =====")
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())


# =========================================================
# 3. KEY BUSINESS METRICS
# =========================================================

total_sales = df["Sales"].sum()
total_units = df["Units"].sum()
total_profit = df["Gross Profit"].sum()
total_cost = df["Cost"].sum()
avg_lead_time = df["Lead Time"].mean()
avg_gross_margin = df["Gross Margin"].mean() * 100

print("\n===== KEY BUSINESS METRICS =====")
print("Total Sales:", round(total_sales, 2))
print("Total Units:", int(total_units))
print("Total Gross Profit:", round(total_profit, 2))
print("Total Cost:", round(total_cost, 2))
print("Average Lead Time:", round(avg_lead_time, 2), "days")
print("Average Gross Margin:", round(avg_gross_margin, 2), "%")


# =========================================================
# 4. YEAR-WISE ANALYSIS
# =========================================================

year_analysis = (
    df.groupby("Order Year")
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Cost=("Cost", "sum"),
        Avg_Lead_Time=("Lead Time", "mean")
    )
    .reset_index()
)

print("\n===== YEAR-WISE PERFORMANCE =====")
print(year_analysis.round(2).to_string(index=False))


# =========================================================
# 5. MONTHLY ANALYSIS
# =========================================================

monthly_analysis = (
    df.groupby(["Order Year", "Order Month"])
    .agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Units=("Units", "sum")
    )
    .reset_index()
)

print("\n===== MONTHLY PERFORMANCE =====")
print(monthly_analysis.round(2).to_string(index=False))


# =========================================================
# 6. REGION ANALYSIS
# =========================================================

region_analysis = (
    df.groupby("Region")
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time", "mean")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

print("\n===== REGION PERFORMANCE =====")
print(region_analysis.round(2).to_string(index=False))


# =========================================================
# 7. DIVISION ANALYSIS
# =========================================================

division_analysis = (
    df.groupby("Division")
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time", "mean")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

print("\n===== DIVISION PERFORMANCE =====")
print(division_analysis.round(2).to_string(index=False))


# =========================================================
# 8. SHIP MODE ANALYSIS
# =========================================================

ship_mode_analysis = (
    df.groupby("Ship Mode")
    .agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time", "mean"),
        Orders=("Order ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

print("\n===== SHIP MODE PERFORMANCE =====")
print(ship_mode_analysis.round(2).to_string(index=False))


# =========================================================
# 9. PRODUCT ANALYSIS
# =========================================================

product_analysis = (
    df.groupby("Product Name")
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time", "mean"),
        Gross_Margin=("Gross Margin", "mean")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

print("\n===== TOP 10 PRODUCTS BY SALES =====")
print(
    product_analysis.head(10)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 10. CUSTOMER ANALYSIS
# =========================================================

customer_analysis = (
    df.groupby("Customer ID")
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

print("\n===== TOP 10 CUSTOMERS BY SALES =====")
print(
    customer_analysis.head(10)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 11. STATE ANALYSIS
# =========================================================

state_analysis = (
    df.groupby("State/Province")
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

print("\n===== TOP 10 STATES BY SALES =====")
print(
    state_analysis.head(10)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 12. LEAD TIME ANALYSIS
# =========================================================

lead_time_analysis = (
    df.groupby("Product Name")
    .agg(
        Avg_Lead_Time=("Lead Time", "mean"),
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum")
    )
    .sort_values("Avg_Lead_Time", ascending=False)
    .reset_index()
)

print("\n===== TOP 10 PRODUCTS BY AVERAGE LEAD TIME =====")
print(
    lead_time_analysis.head(10)
    .round(2)
    .to_string(index=False)
)

print("\n===== LEAD TIME SUMMARY =====")
print("Minimum:", df["Lead Time"].min(), "days")
print("Average:", round(df["Lead Time"].mean(), 2), "days")
print("Maximum:", df["Lead Time"].max(), "days")
print(
    "Records above 1000 days:",
    (df["Lead Time"] > 1000).sum()
)


# =========================================================
# 13. GROSS MARGIN ANALYSIS
# =========================================================

margin_analysis = (
    df.groupby("Product Name")
    .agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Gross_Margin=("Gross Margin", "mean")
    )
    .sort_values("Gross_Margin", ascending=False)
    .reset_index()
)

print("\n===== TOP 10 PRODUCTS BY GROSS MARGIN =====")
print(
    margin_analysis.head(10)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 14. PYTHON CHARTS
# =========================================================

print("\n===== CREATING PYTHON CHARTS =====")

# Year-wise Sales
plt.figure(figsize=(8, 5))
plt.bar(
    year_analysis["Order Year"].astype(str),
    year_analysis["Sales"]
)
plt.title("Year-wise Sales")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("python_yearly_sales.png", dpi=300)
plt.close()


# Regional Sales
plt.figure(figsize=(8, 5))
plt.bar(
    region_analysis["Region"],
    region_analysis["Sales"]
)
plt.title("Regional Sales Performance")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("python_region_sales.png", dpi=300)
plt.close()


# Top 10 Products
top10_products = (
    product_analysis
    .head(10)
    .sort_values("Sales")
)

plt.figure(figsize=(10, 6))
plt.barh(
    top10_products["Product Name"],
    top10_products["Sales"]
)
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig("python_top_products.png", dpi=300)
plt.close()


# Ship Mode Sales
plt.figure(figsize=(8, 5))
plt.bar(
    ship_mode_analysis["Ship Mode"],
    ship_mode_analysis["Sales"]
)
plt.title("Sales by Ship Mode")
plt.xlabel("Ship Mode")
plt.ylabel("Sales")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("python_ship_mode_sales.png", dpi=300)
plt.close()


# =========================================================
# 15. SAVE ALL ANALYSIS TO EXCEL
# =========================================================

print("\n===== CREATING PYTHON ANALYSIS EXCEL FILE =====")

with pd.ExcelWriter(
    "Nassau_Python_Analysis.xlsx",
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Processed Data",
        index=False
    )

    year_analysis.to_excel(
        writer,
        sheet_name="Year Analysis",
        index=False
    )

    monthly_analysis.to_excel(
        writer,
        sheet_name="Monthly Analysis",
        index=False
    )

    region_analysis.to_excel(
        writer,
        sheet_name="Region Analysis",
        index=False
    )

    division_analysis.to_excel(
        writer,
        sheet_name="Division Analysis",
        index=False
    )

    ship_mode_analysis.to_excel(
        writer,
        sheet_name="Ship Mode Analysis",
        index=False
    )

    product_analysis.to_excel(
        writer,
        sheet_name="Product Analysis",
        index=False
    )

    customer_analysis.to_excel(
        writer,
        sheet_name="Customer Analysis",
        index=False
    )

    state_analysis.to_excel(
        writer,
        sheet_name="State Analysis",
        index=False
    )

    lead_time_analysis.to_excel(
        writer,
        sheet_name="Lead Time Analysis",
        index=False
    )

    margin_analysis.to_excel(
        writer,
        sheet_name="Gross Margin Analysis",
        index=False
    )


# =========================================================
# 16. FINAL SUMMARY
# =========================================================

top_product = product_analysis.iloc[0]
top_region = region_analysis.iloc[0]
top_state = state_analysis.iloc[0]

print("\n===== FINAL PYTHON INSIGHTS =====")

print(
    "Top Product by Sales:",
    top_product["Product Name"],
    "| Sales:",
    round(top_product["Sales"], 2)
)

print(
    "Top Region by Sales:",
    top_region["Region"],
    "| Sales:",
    round(top_region["Sales"], 2)
)

print(
    "Top State by Sales:",
    top_state["State/Province"],
    "| Sales:",
    round(top_state["Sales"], 2)
)

print(
    "Overall Average Lead Time:",
    round(avg_lead_time, 2),
    "days"
)

print(
    "Overall Average Gross Margin:",
    round(avg_gross_margin, 2),
    "%"
)

print("\nCharts created:")
print("- python_yearly_sales.png")
print("- python_region_sales.png")
print("- python_top_products.png")
print("- python_ship_mode_sales.png")

print("\nExcel analysis created:")
print("- Nassau_Python_Analysis.xlsx")

# =========================================================
# 17. SHIPPING OPTIMIZATION CANDIDATES
# =========================================================

shipping_analysis = (
    df.groupby(
        ["Product Name", "State/Province", "Region"]
    )
    .agg(
        Sales=("Sales", "sum"),
        Units=("Units", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time", "mean")
    )
    .reset_index()
)

# Focus on high-value product/location combinations
shipping_candidates = shipping_analysis[
    (shipping_analysis["Sales"] >= 1000) &
    (shipping_analysis["Avg_Lead_Time"] >= df["Lead Time"].mean())
].copy()

shipping_candidates = shipping_candidates.sort_values(
    ["Sales", "Avg_Lead_Time"],
    ascending=[False, False]
)

print("\n===== SHIPPING OPTIMIZATION CANDIDATES =====")
print(
    shipping_candidates.head(20)
    .round(2)
    .to_string(index=False)
)

# Save candidates separately
shipping_candidates.to_excel(
    "Nassau_Shipping_Optimization_Candidates.xlsx",
    index=False
)

print("\nShipping optimization candidate file created:")
print("- Nassau_Shipping_Optimization_Candidates.xlsx")

print("\n" + "=" * 70)
print("PYTHON ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)