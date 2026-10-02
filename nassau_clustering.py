import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("=" * 70)
print("NASSAU CANDY - PRODUCT & SHIPPING CLUSTERING")
print("=" * 70)

# Load dataset
df = pd.read_csv("Nassau Candy Distributor.csv")

# Convert dates
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d-%m-%Y",
    errors="coerce"
)
df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Calculate Lead Time
df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

# Remove invalid records
df = df.dropna(subset=["Sales", "Units", "Cost", "Lead Time"])

print(f"\nRecords used for clustering: {len(df)}")

# Aggregate product-level shipping and business metrics
product_cluster = (
    df.groupby("Product Name")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Units=("Units", "sum"),
        Total_Cost=("Cost", "sum"),
        Average_Lead_Time=("Lead Time", "mean"),
        Average_Sales=("Sales", "mean")
    )
    .reset_index()
)

print(f"Products analyzed: {len(product_cluster)}")

# Features used for clustering
features = [
    "Total_Sales",
    "Total_Units",
    "Total_Cost",
    "Average_Lead_Time",
    "Average_Sales"
]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(product_cluster[features])

# K-Means clustering
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

product_cluster["Cluster"] = kmeans.fit_predict(X) + 1

# Cluster summary
cluster_summary = (
    product_cluster
    .groupby("Cluster")
    .agg(
        Products=("Product Name", "count"),
        Total_Sales=("Total_Sales", "sum"),
        Total_Units=("Total_Units", "sum"),
        Average_Lead_Time=("Average_Lead_Time", "mean"),
        Average_Sales=("Average_Sales", "mean")
    )
    .reset_index()
)

print("\n===== CLUSTER SUMMARY =====")
print(cluster_summary.to_string(index=False))

# Sort products by cluster and sales
product_cluster = product_cluster.sort_values(
    ["Cluster", "Total_Sales"],
    ascending=[True, False]
)

# Save results
output_file = "Nassau_Clustering_Results.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    product_cluster.to_excel(
        writer,
        sheet_name="Product Clusters",
        index=False
    )

    cluster_summary.to_excel(
        writer,
        sheet_name="Cluster Summary",
        index=False
    )

print("\n===== CLUSTERING COMPLETED =====")
print(f"Results saved to: {output_file}")