import pandas as pd
import numpy as np
import zipfile
import os
import joblib

np.random.seed(42)

ZIP_PATH = 'mturkfitbit_export_3.12.16-4.11.16.zip'
ACTIVITY_CSV  = 'Fitabase Data 3.12.16-4.11.16/dailyActivity_merged.csv'
SLEEP_CSV     = 'Fitabase Data 3.12.16-4.11.16/minuteSleep_merged.csv'
WEIGHT_CSV    = 'Fitabase Data 3.12.16-4.11.16/weightLogInfo_merged.csv'

# Load the existing trained model once (used by generate_score)
_model = None
_feature_columns = None

def _load_model():
    global _model, _feature_columns
    if _model is None:
        _model = joblib.load('health_model.pkl')
        _feature_columns = joblib.load('feature_columns.pkl')


def load_fitbit_local():
    """
    Load Fitbit data from the local zip file.
    Returns (activity_df, daily_sleep_df, bmi_per_user) where:
      - activity_df   : one row per (user, day) with steps, active minutes, calories, sedentary minutes
      - daily_sleep_df: one row per (user, day) with sleep_hours and sleep_consistency
      - bmi_per_user  : dict {Id: bmi} for users who have weight log entries
    """
    if not os.path.exists(ZIP_PATH):
        print(f"Zip file not found: {ZIP_PATH}")
        return None, None, {}

    print(f"Reading from {ZIP_PATH} ...")

    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(ACTIVITY_CSV) as f:
            activity_df = pd.read_csv(f)
        with z.open(SLEEP_CSV) as f:
            sleep_raw = pd.read_csv(f)
        with z.open(WEIGHT_CSV) as f:
            weight_df = pd.read_csv(f)

    print(f"  Activity rows: {len(activity_df)}")
    print(f"  Minute-sleep rows: {len(sleep_raw)}")
    print(f"  Weight log rows: {len(weight_df)}")

    # --- Aggregate minute-level sleep to daily ---
    # value: 1=asleep, 2=restless (still counted as sleep), 3=awake
    sleep_raw['dt'] = pd.to_datetime(sleep_raw['date'], format='%m/%d/%Y %I:%M:%S %p')
    sleep_raw['date_only'] = sleep_raw['dt'].dt.date
    sleep_raw['is_sleep'] = (sleep_raw['value'].isin([1, 2])).astype(int)

    daily_sleep = (
        sleep_raw
        .groupby(['Id', 'date_only'])['is_sleep']
        .sum()
        .reset_index(name='sleep_minutes')
    )
    daily_sleep['sleep_hours'] = (daily_sleep['sleep_minutes'] / 60).clip(3, 11).round(1)

    # Per-user sleep consistency = 1 - (std of daily sleep hours / 2), clipped to [0.2, 1.0]
    # std of 0h → consistency 1.0; std of 2h+ → consistency 0.0
    user_std = daily_sleep.groupby('Id')['sleep_hours'].std().fillna(0)
    user_consistency = (1 - (user_std / 2)).clip(0.2, 1.0).round(2)
    daily_sleep['sleep_consistency'] = daily_sleep['Id'].map(user_consistency)

    # Normalise the date column to string "M/D/YYYY" so we can merge with activity
    daily_sleep['ActivityDate'] = daily_sleep['date_only'].apply(
        lambda d: f"{d.month}/{d.day}/{d.year}"
    )

    # --- BMI per user from weight log ---
    bmi_per_user = (
        weight_df.dropna(subset=['BMI'])
        .groupby('Id')['BMI']
        .mean()
        .round(1)
        .to_dict()
    )
    print(f"  Users with BMI data: {len(bmi_per_user)}")

    return activity_df, daily_sleep, bmi_per_user


def generate_score(row):
    """
    Predict routine_score using the existing trained health_model.pkl.

    The original train_model.py does NOT define an explicit scoring formula —
    routine_score was pre-computed in routine_dataset.csv.  To keep Fitbit rows
    on the exact same scale we use the trained model with the same 23 features.

    Derived feature formulas match recommender_ai.py build_feature_array exactly.
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
        'sex':               float(row['sex']),
        'bmi':               float(row['bmi']),
        'fitness_level':     float(row['fitness_level']),
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


def map_fitbit_to_features(activity_df, daily_sleep, bmi_per_user):
    """
    Merge activity + sleep, then map real Fitbit columns to our feature schema.

    Real values used directly from Fitbit:
      steps, exercise_minutes, calories, screen_time_hours, sleep_hours, sleep_consistency

    Sampled from realistic distributions (not in Fitbit data):
      age, sex, bmi (real if available), fitness_level, stress_level,
      work_hours, junk_food_meals, water_intake_liters, meals
    """
    # Merge activity with daily sleep on (Id, ActivityDate)
    merged = activity_df.merge(
        daily_sleep[['Id', 'ActivityDate', 'sleep_hours', 'sleep_consistency']],
        on=['Id', 'ActivityDate'],
        how='left'
    )
    print(f"  Activity rows after sleep merge: {len(merged)}")
    print(f"  Rows with real sleep data: {merged['sleep_hours'].notna().sum()}")

    records = []
    for _, row in merged.iterrows():
        try:
            steps = float(row['TotalSteps'])
            if steps < 100:
                continue

            very_active   = float(row['VeryActiveMinutes'])
            fairly_active = float(row['FairlyActiveMinutes'])
            exercise_minutes = min(very_active + fairly_active, 120)

            calories = float(row['Calories'])
            calories = max(1000, min(5000, calories))

            sedentary_minutes = float(row['SedentaryMinutes'])
            screen_time = min(sedentary_minutes / 60, 16)

            # Real sleep if available, otherwise sample
            if pd.notna(row.get('sleep_hours')):
                sleep_hours = float(row['sleep_hours'])
                sleep_consistency = float(row['sleep_consistency'])
            else:
                sleep_hours = np.random.normal(6.8, 1.2)
                sleep_hours = max(3, min(11, sleep_hours))
                sleep_consistency = round(np.random.uniform(0.4, 1.0), 2)

            # Real BMI if user has weight log entry, otherwise sample
            uid = row['Id']
            if uid in bmi_per_user:
                bmi = bmi_per_user[uid]
            else:
                bmi = np.random.normal(26.5, 4.5)
                bmi = max(16, min(45, bmi))

            # Sampled fields (not captured by Fitbit activity tracker)
            age = np.random.normal(35, 10)
            age = max(18, min(65, age))

            sex = np.random.choice([0, 1])

            stress_level = np.random.normal(5, 2)
            stress_level = max(1, min(10, stress_level))

            work_hours = np.random.normal(8, 2)
            work_hours = max(0, min(14, work_hours))

            junk_food = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])

            water = np.random.normal(2.0, 0.7)
            water = max(0.5, min(5, water))

            meals = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])

            fitness_level = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])

            records.append({
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
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def main():
    print("=== Building Enhanced Dataset ===\n")

    # Load existing synthetic data
    print("Loading synthetic dataset...")
    synthetic_df = pd.read_csv('routine_dataset.csv')
    print(f"Synthetic records: {len(synthetic_df)}")

    # Load real Fitbit data from local zip
    print("\nLoading Fitbit data from local zip...")
    activity_df, daily_sleep, bmi_per_user = load_fitbit_local()

    if activity_df is not None:
        print("\nMapping Fitbit data to feature schema...")
        fitbit_df = map_fitbit_to_features(activity_df, daily_sleep, bmi_per_user)
        print(f"Mapped Fitbit records: {len(fitbit_df)}")

        # Score each Fitbit row using the existing trained model
        print("Scoring Fitbit rows with existing health_model.pkl...")
        fitbit_df['routine_score'] = fitbit_df.apply(generate_score, axis=1)
        print(f"Fitbit score range: {fitbit_df['routine_score'].min():.1f} – {fitbit_df['routine_score'].max():.1f}")
        print(f"Fitbit score mean:  {fitbit_df['routine_score'].mean():.1f}")

        # Combine
        combined_df = pd.concat([synthetic_df, fitbit_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\nCombined dataset: {len(combined_df)} records")
        print(f"  Synthetic: {len(synthetic_df)}")
        print(f"  Fitbit:    {len(fitbit_df)}")
    else:
        combined_df = synthetic_df
        print("\nZip not found — using synthetic dataset only")

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
