import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# NASSAU CANDY - LEAD TIME PREDICTIVE MODELING
# =========================================================

print("\n" + "=" * 70)
print("NASSAU CANDY - LEAD TIME PREDICTIVE MODELING")
print("=" * 70)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("Nassau Candy Distributor.csv")

print("\nDataset loaded successfully.")
print("Records:", len(df))


# =========================================================
# 2. DATE PROCESSING
# =========================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d-%m-%Y"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    format="%d-%m-%Y"
)

df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month


# =========================================================
# 3. FEATURES
# =========================================================

features = [
    "Ship Mode",
    "Division",
    "Region",
    "State/Province",
    "Sales",
    "Units",
    "Cost",
    "Order Year",
    "Order Month"
]

target = "Lead Time"

model_df = df[features + [target]].copy()

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

model_df = model_df.dropna()

print("\nRecords used for modeling:", len(model_df))


# =========================================================
# 4. X AND Y
# =========================================================

X = model_df[features]
y = model_df[target]

categorical_features = [
    "Ship Mode",
    "Division",
    "Region",
    "State/Province"
]

numeric_features = [
    "Sales",
    "Units",
    "Cost",
    "Order Year",
    "Order Month"
]


# =========================================================
# 5. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# 6. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 7. MODELS
# =========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# =========================================================
# 8. TRAIN AND EVALUATE
# =========================================================

results = []
predictions = {}

print("\n===== MODEL PERFORMANCE =====")

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("\nTraining:", model_name)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    predictions[model_name] = y_pred

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE",
    ascending=True
).reset_index(drop=True)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))


# =========================================================
# 10. BEST MODEL
# =========================================================

best_model_name = results_df.iloc[0]["Model"]

print("\n===== BEST MODEL BY LOWEST RMSE =====")
print(best_model_name)


# =========================================================
# 11. PREDICTIONS
# =========================================================

best_predictions = predictions[best_model_name]

prediction_output = X_test.copy()

prediction_output["Actual Lead Time"] = y_test.values
prediction_output["Predicted Lead Time"] = best_predictions

prediction_output["Prediction Error"] = (
    prediction_output["Actual Lead Time"]
    - prediction_output["Predicted Lead Time"]
)

prediction_output = prediction_output.reset_index(drop=True)


# =========================================================
# 12. SAVE RESULTS
# =========================================================

print("\n===== SAVING MODEL RESULTS =====")

with pd.ExcelWriter(
    "Nassau_Predictive_Model_Results.xlsx",
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Model Comparison",
        index=False
    )

    prediction_output.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    model_df.to_excel(
        writer,
        sheet_name="Model Dataset",
        index=False
    )


# =========================================================
# 13. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("PREDICTIVE MODELING COMPLETED")
print("=" * 70)

print("\nModels evaluated:")
print("- Linear Regression")
print("- Random Forest")
print("- Gradient Boosting")

print("\nBest model by lowest RMSE:")
print(best_model_name)

print("\nResults saved to:")
print("- Nassau_Predictive_Model_Results.xlsx")

print("\n" + "=" * 70)