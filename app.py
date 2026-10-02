import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy Analysis",
    page_icon="🍫",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Nassau Candy Distributor.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d-%m-%Y"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d-%m-%Y"
    )

    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.strftime("%b")
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Gross Margin"] = (
        df["Gross Profit"] / df["Sales"] * 100
    )

    return df


df = load_data()

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("Nassau Candy Distributor - Data Analysis")
st.write(
    "Interactive analysis of sales, products, customers, "
    "regions, shipping modes and shipping lead time."
)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

years = sorted(df["Order Year"].unique())

selected_years = st.sidebar.multiselect(
    "Order Year",
    years,
    default=years
)

regions = sorted(df["Region"].dropna().unique())

selected_regions = st.sidebar.multiselect(
    "Region",
    regions,
    default=regions
)

filtered_df = df[
    df["Order Year"].isin(selected_years)
    & df["Region"].isin(selected_regions)
]

# ---------------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------------
total_sales = filtered_df["Sales"].sum()
total_units = filtered_df["Units"].sum()
total_profit = filtered_df["Gross Profit"].sum()
gross_margin = (
    total_profit / total_sales * 100
    if total_sales != 0 else 0
)
avg_lead_time = filtered_df["Lead Time"].mean()

top_state = (
    filtered_df.groupby("State/Province")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

top_state_name = top_state.index[0] if len(top_state) else "N/A"

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Total Sales",
    f"${total_sales:,.2f}"
)

col2.metric(
    "Total Units",
    f"{total_units:,.0f}"
)

col3.metric(
    "Gross Profit",
    f"${total_profit:,.2f}"
)

col4.metric(
    "Gross Margin",
    f"{gross_margin:.2f}%"
)

col5.metric(
    "Avg Lead Time",
    f"{avg_lead_time:,.2f} days"
)

col6.metric(
    "Top State",
    top_state_name
)

st.divider()

# ---------------------------------------------------------
# YEARLY SALES
# ---------------------------------------------------------
st.subheader("Year-wise Sales")

yearly_sales = (
    filtered_df.groupby("Order Year", as_index=False)["Sales"]
    .sum()
)

fig_year = px.bar(
    yearly_sales,
    x="Order Year",
    y="Sales",
    title="Sales by Order Year",
    text_auto=".2s"
)

st.plotly_chart(
    fig_year,
    use_container_width=True
)

# ---------------------------------------------------------
# REGION ANALYSIS
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Region")

    region_sales = (
        filtered_df.groupby("Region", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )

    fig_region = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        title="Regional Sales"
    )

    st.plotly_chart(
        fig_region,
        use_container_width=True
    )

with col2:
    st.subheader("Sales by Ship Mode")

    ship_sales = (
        filtered_df.groupby("Ship Mode", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )

    fig_ship = px.bar(
        ship_sales,
        x="Ship Mode",
        y="Sales",
        title="Sales by Shipping Mode"
    )

    st.plotly_chart(
        fig_ship,
        use_container_width=True
    )

# ---------------------------------------------------------
# TOP PRODUCTS
# ---------------------------------------------------------
st.subheader("Top Products by Sales")

product_sales = (
    filtered_df.groupby("Product Name", as_index=False)["Sales"]
    .sum()
    .sort_values("Sales", ascending=False)
    .head(10)
)

fig_products = px.bar(
    product_sales.sort_values("Sales"),
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 10 Products"
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)

# ---------------------------------------------------------
# MONTHLY SALES
# ---------------------------------------------------------
st.subheader("Monthly Sales")

month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

monthly_sales = (
    filtered_df.groupby(
        ["Order Year", "Order Month"],
        as_index=False
    )["Sales"]
    .sum()
)

monthly_sales["Month Number"] = (
    monthly_sales["Order Month"]
    .map({month: i for i, month in enumerate(month_order, 1)})
)

monthly_sales = monthly_sales.sort_values(
    ["Order Year", "Month Number"]
)

fig_month = px.line(
    monthly_sales,
    x="Order Month",
    y="Sales",
    color="Order Year",
    markers=True,
    category_orders={"Order Month": month_order},
    title="Monthly Sales Trend"
)

st.plotly_chart(
    fig_month,
    use_container_width=True
)

# ---------------------------------------------------------
# LEAD TIME ANALYSIS
# ---------------------------------------------------------
st.subheader("Shipping Lead Time Analysis")

lead_summary = filtered_df["Lead Time"].describe()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Minimum",
    f"{lead_summary['min']:,.0f} days"
)

col2.metric(
    "Median",
    f"{lead_summary['50%']:,.0f} days"
)

col3.metric(
    "Average",
    f"{lead_summary['mean']:,.2f} days"
)

col4.metric(
    "Maximum",
    f"{lead_summary['max']:,.0f} days"
)

# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------
st.subheader("Filtered Data Preview")

st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Nassau Candy Data Analysis | Excel + Python + Streamlit"
)