import os
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the ML model and feature column order saved during training
model = joblib.load("health_model.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

# Fields every user can reasonably answer — missing these is an error
REQUIRED_FIELDS = [
    "age", "sex", "sleep_hours", "steps", "exercise_minutes",
    "stress_level", "screen_time_hours", "work_hours", "junk_food_meals"
]

# Fields that are optional — imputer fills them if missing
# bmi, fitness_level, sleep_consistency, caloric_intake, meals, water_intake_liters

# Valid ranges for each field
BOUNDS = {
    "sleep_hours":         (3, 11),
    "steps":               (500, 18000),
    "exercise_minutes":    (0, 120),
    "stress_level":        (1, 10),
    "screen_time_hours":   (0, 18),
    "work_hours":          (0, 14),
    "junk_food_meals":     (0, 3),
    "water_intake_liters": (0.5, 5),
    "caloric_intake":      (1000, 5000),
    "sleep_consistency":   (0.2, 1.0)
}


def validate_inputs(user_data):
    """Raise an error if any required field is missing or any value is out of range."""
    for field in REQUIRED_FIELDS:
        if user_data.get(field) is None:
            raise ValueError(f"Missing required field: {field}")

    for field, (low, high) in BOUNDS.items():
        val = user_data.get(field)
        if val is not None and not (low <= val <= high):
            raise ValueError(f"{field} must be between {low} and {high}, got {val}")


def build_feature_array(user_data):
    """Build the feature array in the exact order the model was trained on."""
    sleep    = user_data["sleep_hours"]
    steps    = user_data["steps"]
    exercise = user_data["exercise_minutes"]
    stress   = user_data["stress_level"]
    screen   = user_data["screen_time_hours"]
    work     = user_data["work_hours"]
    junk     = user_data["junk_food_meals"]
    water    = user_data.get("water_intake_liters") or 2.0
    calories = user_data.get("caloric_intake") or 2200
    meals    = user_data.get("meals") or 3

    feature_dict = {
        "age":               user_data["age"],
        "sex":               user_data["sex"],
        "bmi":               user_data.get("bmi"),               # optional
        "fitness_level":     user_data.get("fitness_level"),     # optional
        "sleep_consistency": user_data.get("sleep_consistency"), # optional
        "sleep_hours":       sleep,
        "steps":             steps,
        "exercise_minutes":  exercise,
        "meals":             meals,
        "junk_food_meals":   junk,
        "water_intake_liters": water,
        "caloric_intake":    calories,
        "screen_time_hours": screen,
        "work_hours":        work,
        "stress_level":      stress,
        "activity_score":    steps / 1000 + exercise / 30,
        "diet_quality":      10 - junk - abs(calories - 2200) / 400,
        "recovery_score":    sleep - stress / 2,
        "sedentary_score":   screen - steps / 5000,
        "work_life_balance": work + stress,
        "exercise_ratio":    exercise / (steps + 1),
        "water_per_meal":    water / (meals + 0.01),
        "sleep_efficiency":  sleep / (stress + 1)
    }

    return pd.DataFrame([feature_dict])[FEATURE_COLUMNS]


def predict_score(user_data):
    """Predict routine score from user data."""
    return round(float(model.predict(build_feature_array(user_data))[0]), 1)


def run_counterfactual(user_data, override):
    """Predict score with one or more inputs changed."""
    modified = user_data.copy()
    modified.update(override)
    return predict_score(modified)


def calculate_sub_scores(user_data):
    """Calculate 0-100 sub-scores from raw inputs.
    Note: These use manually designed formulas, not the ML model.
    They are directionally aligned with the training formula but approximate.
    Future improvement: train separate models per sub-score.
    """
    sleep       = user_data["sleep_hours"]
    steps       = user_data["steps"]
    exercise    = user_data["exercise_minutes"]
    stress      = user_data["stress_level"]
    screen      = user_data["screen_time_hours"]
    work        = user_data["work_hours"]
    junk        = user_data["junk_food_meals"]
    water       = user_data.get("water_intake_liters") or 2.0
    calories    = user_data.get("caloric_intake") or 2200
    consistency = user_data.get("sleep_consistency") or 0.7

    return {
        "Sleep Quality":     round(min(100, max(0, (sleep / 9) * 100 * consistency))),
        "Physical Activity": round(min(100, max(0, (steps / 12000) * 50 + (exercise / 90) * 50))),
        "Diet & Nutrition":  round(min(100, max(0, 100 - junk * 15 - abs(calories - 2200) / 50 + water * 5))),
        "Recovery & Stress": round(min(100, max(0, 100 - stress * 10 + (sleep - 5) * 5))),
        "Work-Life Balance": round(min(100, max(0, 100 - max(0, work - 8) * 8 - stress * 4 + (8 - screen) * 2)))
    }


def calculate_counterfactuals(user_data, current_score):
    """Run counterfactual simulations, rank by impact, build combined from actual top 3."""
    sleep    = user_data["sleep_hours"]
    screen   = user_data["screen_time_hours"]
    stress   = user_data["stress_level"]
    exercise = user_data["exercise_minutes"]

    individual_scenarios = {
        f"Increase sleep to {min(sleep + 1.5, 9)}h":
            run_counterfactual(user_data, {"sleep_hours": min(sleep + 1.5, 9)}),
        f"Reduce screen time to {max(screen - 3, 0)}h":
            run_counterfactual(user_data, {"screen_time_hours": max(screen - 3, 0)}),
        f"Reduce stress to {max(stress - 2, 1)}/10":
            run_counterfactual(user_data, {"stress_level": max(stress - 2, 1)}),
        f"Add 20 min exercise ({min(exercise + 20, 120)} min total)":
            run_counterfactual(user_data, {"exercise_minutes": min(exercise + 20, 120)}),
    }

    # Rank individual changes by impact
    ranked = sorted(individual_scenarios.items(), key=lambda x: x[1] - current_score, reverse=True)

    # Build combined scenario from actual top 3 ranked changes
    top_3_overrides = {}
    for label, _ in ranked[:3]:
        if "sleep" in label:
            top_3_overrides["sleep_hours"] = min(sleep + 1.5, 9)
        elif "screen" in label:
            top_3_overrides["screen_time_hours"] = max(screen - 3, 0)
        elif "stress" in label:
            top_3_overrides["stress_level"] = max(stress - 2, 1)
        elif "exercise" in label:
            top_3_overrides["exercise_minutes"] = min(exercise + 20, 120)

    combined_score = run_counterfactual(user_data, top_3_overrides)
    results = ranked + [("Combine all top 3 changes", combined_score)]

    return [(label, score, round(score - current_score, 1)) for label, score in results]


def generate_report(score, sub_scores, cf_results, user_data):
    """Call the LLM to write the final report using pre-calculated data."""
    cf_text  = "\n".join([
        f"- {label}: predicted score {pred} ({'+' if delta > 0 else ''}{delta} points)"
        for label, pred, delta in cf_results
    ])
    sub_text = "\n".join([f"- {k}: {v}/100" for k, v in sub_scores.items()])

    prompt = f"""
You are a health analytics platform generating a personalized daily report. Be analytical, direct, and specific. Do not ask questions. Do not end with "how do you feel." Lead with findings, not warmth.

USER DATA:
- Overall routine score: {score}/100
- Sleep: {user_data['sleep_hours']}h | Steps: {user_data['steps']} | Exercise: {user_data['exercise_minutes']} min
- Stress: {user_data['stress_level']}/10 | Screen time: {user_data['screen_time_hours']}h | Work: {user_data['work_hours']}h
- Junk food meals: {user_data['junk_food_meals']} | Water: {user_data['water_intake_liters']}L

SUB-SCORES (report these exactly — do not change the numbers):
{sub_text}

COUNTERFACTUAL PREDICTIONS (use these exact numbers — do not improvise):
{cf_text}

FORMAT YOUR REPORT EXACTLY LIKE THIS:

**Overall Score: {score}/100**
[One sentence: below 40 is poor, 40-60 is poor to moderate, 60-80 is good, above 80 is excellent. Be honest and specific — reference which sub-scores are driving the result, not just the number.]

**Sub-Score Breakdown**
[List all 5 sub-scores with a one-line interpretation of each]

**Root Cause Analysis**
[Identify the 2 lowest sub-scores. Do not just list the inputs — explain the causal chain between them. Show how one bad habit is feeding into another. Be specific about which exact input values are responsible and why they interact with each other.]

**Predicted Impact of Changes**
List individual changes first ranked by impact highest to lowest, then the combined scenario last as a summary. Never rank the combined scenario above individual changes. Use the exact numbers provided above.

**Priority Action Plan**
Do NOT repeat the counterfactual list. For each priority, explain HOW to make the change — give a specific behavioral instruction the user can act on today. Each action must reference the user's actual input values — do not give advice that could apply to anyone, connect the recommendation directly to what this specific user logged. Do not soften the target values — if the counterfactual says 1.5 hours more sleep, the recommendation must reflect that exactly, not a reduced version. For stress, do not give generic advice like "take a walk" — identify the specific inputs causing the stress (work hours, screen time) and recommend reducing those directly. List only the top 3 individual changes (do NOT include "Combine all top 3 changes"):
1. [Specific behavioral instruction tied to this user's actual numbers + predicted improvement]
2. [Specific behavioral instruction tied to this user's actual numbers + predicted improvement]
3. [Specific behavioral instruction tied to this user's actual numbers + predicted improvement]

**Today's Focus**
[One sentence. Name the single most important action for tonight specifically — not a general strategy, one concrete thing. Example: "Set your phone to Do Not Disturb at 10:30pm tonight and be in bed by 11:00pm."]
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a precise health analytics engine. Follow the report format exactly. Use only the numbers provided. Do not improvise or round differently."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"Report generation failed: {str(e)}\n\n"
            f"Raw scores — Overall: {score}/100\n"
            f"Sub-scores: {sub_scores}"
        )


def generate_recommendation(user_data):
    """Main entry point — validates input, runs all calculations, returns all results."""
    validate_inputs(user_data)
    score      = predict_score(user_data)
    sub_scores = calculate_sub_scores(user_data)
    cf_results = calculate_counterfactuals(user_data, score)
    report     = generate_report(score, sub_scores, cf_results, user_data)
    return score, sub_scores, cf_results, report


# Test it
if __name__ == "__main__":
    test_data = {
        "age": 25,
        "sex": 0,
        "bmi": 24,
        "fitness_level": 1,
        "sleep_consistency": 0.6,
        "sleep_hours": 5.5,
        "steps": 8000,
        "exercise_minutes": 90,
        "stress_level": 8,
        "screen_time_hours": 10,
        "work_hours": 10,
        "junk_food_meals": 2,
        "water_intake_liters": 2,
        "caloric_intake": 2600,
        "meals": 3
    }

    score, sub_scores, cf_results, report = generate_recommendation(test_data)

    print(f"Predicted Score: {score}/100\n")

    print("Sub-scores:")
    for k, v in sub_scores.items():
        print(f"  {k}: {v}/100")

    print("\nCounterfactuals:")
    for label, pred, delta in cf_results:
        print(f"  {label}: {pred} ({'+' if delta > 0 else ''}{delta})")

    print("\n--- REPORT ---")
    print(report)
