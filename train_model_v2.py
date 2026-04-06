import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available — skipping")

np.random.seed(42)


def engineer_features(df):
    """
    Apply exact same feature engineering as the original train_model.py.

    train_model.py:
      1. Maps sex strings:          Male→0, Female→1
      2. Maps fitness_level strings: Beginner→0, Intermediate→1, Advanced→2
      3. Computes exercise_ratio    = exercise_minutes / (steps + 1)
      4. Computes water_per_meal    = water_intake_liters / meals
         (recommender_ai.py uses meals+0.01 to avoid div/0; we match recommender_ai.py)
      5. Computes sleep_efficiency  = sleep_hours / (stress_level + 1)

    The CSV also has pre-computed activity_score, diet_quality, recovery_score,
    sedentary_score, work_life_balance — we recompute all of them here so that
    Fitbit rows (which may be missing them) are consistent with synthetic rows.

    Formulas match build_feature_array in recommender_ai.py exactly:
      activity_score    = steps/1000 + exercise_minutes/30
      diet_quality      = 10 - junk_food_meals - abs(caloric_intake-2200)/400
      recovery_score    = sleep_hours - stress_level/2
      sedentary_score   = screen_time_hours - steps/5000
      work_life_balance = work_hours + stress_level
      exercise_ratio    = exercise_minutes / (steps+1)
      water_per_meal    = water_intake_liters / (meals+0.01)
      sleep_efficiency  = sleep_hours / (stress_level+1)
    """
    df = df.copy()

    # --- Encode categoricals ---
    # sex: 'Male'→0, 'Female'→1  (Fitbit rows already have 0/1 as int/float)
    sex_mapped = df['sex'].map({'Male': 0, 'Female': 1})
    # For rows where map returns NaN (already numeric), keep the original value
    df['sex'] = sex_mapped.where(sex_mapped.notna(), other=pd.to_numeric(df['sex'], errors='coerce'))
    df['sex'] = df['sex'].astype(float)

    # fitness_level: 'Beginner'→0, 'Intermediate'→1, 'Advanced'→2
    fl_mapped = df['fitness_level'].map({'Beginner': 0, 'Intermediate': 1, 'Advanced': 2})
    df['fitness_level'] = fl_mapped.where(fl_mapped.notna(), other=pd.to_numeric(df['fitness_level'], errors='coerce'))
    df['fitness_level'] = df['fitness_level'].astype(float)

    # --- Derived features (must match recommender_ai.py build_feature_array) ---
    df['activity_score']    = df['steps'] / 1000 + df['exercise_minutes'] / 30
    df['diet_quality']      = 10 - df['junk_food_meals'] - (df['caloric_intake'] - 2200).abs() / 400
    df['recovery_score']    = df['sleep_hours'] - df['stress_level'] / 2
    df['sedentary_score']   = df['screen_time_hours'] - df['steps'] / 5000
    df['work_life_balance'] = df['work_hours'] + df['stress_level']
    df['exercise_ratio']    = df['exercise_minutes'] / (df['steps'] + 1)
    df['water_per_meal']    = df['water_intake_liters'] / (df['meals'] + 0.01)
    df['sleep_efficiency']  = df['sleep_hours'] / (df['stress_level'] + 1)

    return df


def load_and_prepare_data():
    """Load enhanced dataset and prepare features."""
    print("Loading routine_dataset_enhanced.csv...")
    df = pd.read_csv('routine_dataset_enhanced.csv')
    print(f"Total records: {len(df)}")

    # Engineer features (encode categoricals + compute derived cols)
    df = engineer_features(df)

    # Load exact feature column order from existing pkl — must not change this
    FEATURE_COLUMNS = joblib.load('feature_columns.pkl')
    print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # Verify all feature columns are present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLUMNS].copy()

    # Determine target column
    target_col = 'routine_score'
    if target_col not in df.columns:
        for alt in ['score', 'health_score', 'target']:
            if alt in df.columns:
                target_col = alt
                break

    y = df[target_col]

    print(f"Target: {target_col}")
    print(f"Score range: {y.min():.1f} – {y.max():.1f}")
    print(f"Score mean:  {y.mean():.1f}")

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))

    return X, y, FEATURE_COLUMNS


def evaluate_model(model, X, y, model_name):
    """Evaluate model with 5-fold cross validation."""
    print(f"\n--- Evaluating {model_name} ---")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mae_scores  = []
    rmse_scores = []
    r2_scores   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, 100)

        mae  = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2   = r2_score(y_val, y_pred)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"  Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.4f}")

    avg_mae  = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_r2   = np.mean(r2_scores)

    print(f"  AVERAGE: MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, R²={avg_r2:.4f}")

    return avg_mae, avg_rmse, avg_r2


def main():
    print("=== Health Model Training v2 ===\n")

    X, y, FEATURE_COLUMNS = load_and_prepare_data()

    # Define models to compare
    models = {
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ))
        ]),
        'MLP_Neural_Network': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            ))
        ]),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

    # Evaluate all models
    results = {}
    for name, model in models.items():
        mae, rmse, r2 = evaluate_model(model, X, y, name)
        results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'model': model}

    # Print comparison table
    print("\n=== MODEL COMPARISON ===")
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<25} {res['mae']:>8.3f} {res['rmse']:>8.3f} {res['r2']:>8.4f}")

    # Pick best by MAE
    best_name   = min(results, key=lambda k: results[k]['mae'])
    best_result = results[best_name]

    print(f"\nBest model: {best_name}")
    print(f"MAE:  {best_result['mae']:.3f}")
    print(f"RMSE: {best_result['rmse']:.3f}")
    print(f"R²:   {best_result['r2']:.4f}")

    # Retrain best model on full dataset
    print(f"\nRetraining {best_name} on full dataset...")
    best_model = best_result['model']
    best_model.fit(X, y)

    # Verify predictions are sane
    sample_preds = best_model.predict(X[:10])
    sample_preds = np.clip(sample_preds, 0, 100)
    print(f"Sample predictions: {sample_preds.round(1)}")
    print(f"All in range [0,100]: {all(0 <= p <= 100 for p in sample_preds)}")

    # Save model — overwrites health_model.pkl so recommender_ai.py uses the new model
    joblib.dump(best_model, 'health_model.pkl')
    # feature_columns.pkl is unchanged — same 23 columns in same order
    joblib.dump(FEATURE_COLUMNS, 'feature_columns.pkl')
    print(f"\nSaved health_model.pkl ({best_name})")
    print("Saved feature_columns.pkl (unchanged column order)")

    print("\n=== Training Complete ===")
    print(f"Winner:        {best_name}")
    print(f"Dataset size:  {len(X)} records")
    print(f"Features:      {len(FEATURE_COLUMNS)}")


if __name__ == '__main__':
    main()
