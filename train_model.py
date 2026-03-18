import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


df = pd.read_csv("routine_dataset.csv")
print(df.head())
print(df.shape)

df = df.drop(columns = ["user_id", "day", "health_label"])

df["sex"] = df["sex"].map({"Male": 0, "Female": 1})
df["fitness_level"] = df["fitness_level"].map({"Beginner": 0, "Intermediate": 1, "Advanced": 2})

df["exercise_ratio"] = df["exercise_minutes"] / (df["steps"] + 1)
df["water_per_meal"] = df["water_intake_liters"] / df["meals"]
df["sleep_efficiency"] = df["sleep_hours"] / (df["stress_level"] + 1)

x = df.drop("routine_score", axis = 1)
y = df["routine_score"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "mean")),
    ("model", RandomForestRegressor(
        n_estimators = 100, 
        max_depth = 20, 
        min_samples_leaf = 2,
        max_features = 'sqrt',
        random_state = 42,
        n_jobs = -1
    ))
]
)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# Save column order so recommender can build inputs in the correct order
joblib.dump(list(x.columns), "feature_columns.pkl")

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE: ", round(mae, 2))
print("RMSE: ", round(rmse, 2))
print("R2 Score: ", round(r2, 3))

cv_scores = cross_val_score(pipeline, x, y, cv = 5, scoring = "r2")
print("Mean CV R^2: ", cv_scores.mean())

joblib.dump(pipeline, "health_model.pkl", compress = 3)

importances = pd.Series(pipeline["model"].feature_importances_, index=x.columns)
importances = importances.sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()