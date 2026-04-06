import pandas as pd
import numpy as np
import requests
import os
import joblib

np.random.seed(42)

# Load the existing trained model once (used by generate_score)
_model = None
_feature_columns = None

def _load_model():
    global _model, _feature_columns
    if _model is None:
        _model = joblib.load('health_model.pkl')
        _feature_columns = joblib.load('feature_columns.pkl')


def download_fitbit_data():
    """Try multiple sources for real Fitbit data."""
    urls = [
        "https://raw.githubusercontent.com/enesonus/fitbit-data/main/Fitbit%20Data/dailyActivity_merged.csv",
        "https://raw.githubusercontent.com/datasets/fitbit/master/data/dailyActivity_merged.csv",
    ]
    for url in urls:
        try:
            print(f"Trying: {url}")
            df = pd.read_csv(url)
            print(f"Downloaded {len(df)} rows from Fitbit dataset")
            return df
        except Exception as e:
            print(f"Failed: {e}")
    print("All Fitbit URLs failed — using synthetic data only")
    return None


def generate_score(row):
    """
    Generate routine_score using the existing trained health_model.pkl.

    The original train_model.py does NOT define an explicit scoring formula —
    routine_score was pre-computed in routine_dataset.csv. To keep Fitbit rows
    on the exact same scale, we use the existing model to predict the score from
    the same 23 feature columns it was trained on.

    Feature engineering matches train_model.py exactly:
      - sex: 0=Male, 1=Female  (already numeric in Fitbit rows)
      - fitness_level: 0=Beginner, 1=Intermediate, 2=Advanced  (already numeric)
      - activity_score  = steps/1000 + exercise_minutes/30
      - diet_quality    = 10 - junk_food_meals - abs(caloric_intake-2200)/400
      - recovery_score  = sleep_hours - stress_level/2
      - sedentary_score = screen_time_hours - steps/5000
      - work_life_balance = work_hours + stress_level
      - exercise_ratio  = exercise_minutes / (steps+1)
      - water_per_meal  = water_intake_liters / (meals+0.01)   [+0.01 avoids div/0, matches recommender_ai.py]
      - sleep_efficiency = sleep_hours / (stress_level+1)
    """
    _load_model()

    steps    = float(row['steps'])
    exercise = float(row['exercise_minutes'])
    junk     = float(row['junk_food_meals'])
    calories = float(row['caloric_intake'])
    sleep    = float(row['sleep_hours'])
    stress   = float(row['stress_level'])
    screen   = float(row['screen_time_hours'])
    work     = float(row['work_hours'])
    water    = float(row['water_intake_liters'])
    meals    = float(row['meals'])

    feature_dict = {
        'age':               float(row['age']),
        'sex':               float(row['sex']),            # already 0/1
        'bmi':               float(row['bmi']),
        'fitness_level':     float(row['fitness_level']),  # already 0/1/2
        'sleep_consistency': float(row['sleep_consistency']),
        'sleep_hours':       sleep,
        'steps':             steps,
        'exercise_minutes':  exercise,
        'meals':             meals,
        'junk_food_meals':   junk,
        'water_intake_liters': water,
        'caloric_intake':    calories,
        'screen_time_hours': screen,
        'work_hours':        work,
        'stress_level':      stress,
        # Derived features — must match train_model.py + recommender_ai.py exactly
        'activity_score':    steps / 1000 + exercise / 30,
        'diet_quality':      10 - junk - abs(calories - 2200) / 400,
        'recovery_score':    sleep - stress / 2,
        'sedentary_score':   screen - steps / 5000,
        'work_life_balance': work + stress,
        'exercise_ratio':    exercise / (steps + 1),
        'water_per_meal':    water / (meals + 0.01),
        'sleep_efficiency':  sleep / (stress + 1),
    }

    X = pd.DataFrame([feature_dict])[_feature_columns]
    score = float(_model.predict(X)[0])
    return round(max(0.0, min(100.0, score)), 1)


def map_fitbit_to_features(fitbit_df):
    """Map Fitbit columns to our feature schema."""
    records = []

    for _, row in fitbit_df.iterrows():
        try:
            steps = float(row.get('TotalSteps', 0))
            if steps < 100:  # Skip clearly invalid rows
                continue

            very_active   = float(row.get('VeryActiveMinutes', 0))
            fairly_active = float(row.get('FairlyActiveMinutes', 0))
            exercise_minutes = min(very_active + fairly_active, 120)

            calories = float(row.get('Calories', 2000))
            calories = max(1000, min(5000, calories))

            sedentary_minutes = float(row.get('SedentaryMinutes', 360))
            screen_time = min(sedentary_minutes / 60, 16)

            # For missing fields, sample from realistic distributions
            # Fitbit users are predominantly adults 18-49
            age = np.random.normal(35, 10)
            age = max(18, min(65, age))

            sex = np.random.choice([0, 1])

            bmi = np.random.normal(26.5, 4.5)
            bmi = max(16, min(45, bmi))

            sleep_hours = np.random.normal(6.8, 1.2)
            sleep_hours = max(3, min(11, sleep_hours))

            stress_level = np.random.normal(5, 2)
            stress_level = max(1, min(10, stress_level))

            work_hours = np.random.normal(8, 2)
            work_hours = max(0, min(14, work_hours))

            junk_food = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])

            water = np.random.normal(2.0, 0.7)
            water = max(0.5, min(5, water))

            meals = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])

            fitness_level    = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            sleep_consistency = np.random.uniform(0.4, 1.0)

            record = {
                'age':               round(age, 1),
                'sex':               sex,
                'bmi':               round(bmi, 1),
                'fitness_level':     fitness_level,
                'sleep_consistency': round(sleep_consistency, 2),
                'sleep_hours':       round(sleep_hours, 1),
                'steps':             int(steps),
                'exercise_minutes':  int(exercise_minutes),
                'meals':             meals,
                'junk_food_meals':   junk_food,
                'water_intake_liters': round(water, 1),
                'caloric_intake':    int(calories),
                'screen_time_hours': round(screen_time, 1),
                'work_hours':        round(work_hours, 1),
                'stress_level':      round(stress_level, 1),
            }
            records.append(record)
        except Exception:
            continue

    return pd.DataFrame(records)


def main():
    print("=== Building Enhanced Dataset ===\n")

    # Load existing synthetic data
    print("Loading synthetic dataset...")
    synthetic_df = pd.read_csv('routine_dataset.csv')
    print(f"Synthetic records: {len(synthetic_df)}")
    print(f"Synthetic columns: {list(synthetic_df.columns)}")

    # Try to get real Fitbit data
    fitbit_raw = download_fitbit_data()

    if fitbit_raw is not None:
        print("\nMapping Fitbit data to feature schema...")
        fitbit_df = map_fitbit_to_features(fitbit_raw)
        print(f"Mapped Fitbit records: {len(fitbit_df)}")

        # Score each Fitbit row using the existing trained model
        print("Scoring Fitbit rows with existing health_model.pkl...")
        fitbit_df['routine_score'] = fitbit_df.apply(generate_score, axis=1)
        print(f"Fitbit score range: {fitbit_df['routine_score'].min():.1f} – {fitbit_df['routine_score'].max():.1f}")
        print(f"Fitbit score mean:  {fitbit_df['routine_score'].mean():.1f}")

        # Combine datasets (keep only columns that exist in both)
        # synthetic_df has extra columns (user_id, day, health_label, activity_score etc.) — keep them
        combined_df = pd.concat([synthetic_df, fitbit_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\nCombined dataset: {len(combined_df)} records")
    else:
        combined_df = synthetic_df
        print("\nUsing synthetic dataset only")

    # Validate no nulls in required columns
    required_cols = [
        'age', 'sex', 'sleep_hours', 'steps', 'exercise_minutes',
        'stress_level', 'screen_time_hours', 'work_hours',
        'junk_food_meals', 'water_intake_liters'
    ]

    for col in required_cols:
        if col in combined_df.columns:
            null_count = combined_df[col].isnull().sum()
            if null_count > 0:
                print(f"WARNING: {col} has {null_count} nulls — filling with median")
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    combined_df.to_csv('routine_dataset_enhanced.csv', index=False)
    print(f"\nSaved routine_dataset_enhanced.csv ({len(combined_df)} records)")
    print("Dataset build complete.")


if __name__ == '__main__':
    main()
