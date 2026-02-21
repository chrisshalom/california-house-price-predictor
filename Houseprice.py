import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# 1️⃣ LOAD DATA
# ===============================

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Price")

df = pd.concat([X, y], axis=1)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# ===============================
# 2️⃣ TRAIN TEST SPLIT
# ===============================

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Test size:", X_test.shape)


# ===============================
# 3️⃣ TRAIN MODELS
# ===============================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

print("\nModel Performance:")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{name} MSE: {mse:.4f}")
    print(f"{name} R2: {r2:.4f}")
    print("-" * 30)


# ===============================
# 4️⃣ TUNED XGBOOST (BEST MODEL)
# ===============================

xgb_tuned = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb_tuned.fit(X_train, y_train)

pred = xgb_tuned.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nTuned XGBoost Results:")
print("MSE:", round(mse, 4))
print("R2:", round(r2, 4))


# ===============================
# 5️⃣ FEATURE IMPORTANCE
# ===============================

importance = xgb_tuned.feature_importances_

plt.figure()
plt.bar(X.columns, importance)
plt.xticks(rotation=90)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()


# ===============================
# 6️⃣ SAVE MODEL (VERY IMPORTANT)
# ===============================
print("About to save model...")

joblib.dump(xgb_tuned, "house_price_model.pkl")

print("Model saved successfully as house_price_model.pkl")
