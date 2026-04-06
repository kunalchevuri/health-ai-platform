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
    "stress_level", "screen_time_hours", "work_hours", "junk_food_meals",
    "water_intake_liters"
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

PERSONAS = ["athlete", "student", "office_worker", "parent", "manual_laborer", "healthcare_worker", "retired", "general"]

PERSONA_BASELINES = {
    "athlete": {
        "sleep_hours": 8.5,
        "steps": 12000,
        "exercise_minutes": 60,
        "stress_level": 4,
        "screen_time_hours": 4,
        "work_hours": 6,
        "junk_food_meals": 0,
        "water_intake_liters": 3.5,
    },
    "student": {
        "sleep_hours": 7.5,
        "steps": 7000,
        "exercise_minutes": 30,
        "stress_level": 6,
        "screen_time_hours": 8,
        "work_hours": 6,
        "junk_food_meals": 1,
        "water_intake_liters": 2.0,
    },
    "office_worker": {
        "sleep_hours": 7,
        "steps": 5000,
        "exercise_minutes": 20,
        "stress_level": 5,
        "screen_time_hours": 9,
        "work_hours": 9,
        "junk_food_meals": 1,
        "water_intake_liters": 1.8,
    },
    "parent": {
        "sleep_hours": 6.5,
        "steps": 7000,
        "exercise_minutes": 20,
        "stress_level": 6,
        "screen_time_hours": 5,
        "work_hours": 10,
        "junk_food_meals": 1,
        "water_intake_liters": 2.0,
    },
    "manual_laborer": {
        "sleep_hours": 7.5,
        "steps": 15000,
        "exercise_minutes": 45,
        "stress_level": 5,
        "screen_time_hours": 3,
        "work_hours": 10,
        "junk_food_meals": 1,
        "water_intake_liters": 3.0,
    },
    "healthcare_worker": {
        "sleep_hours": 7,
        "steps": 10000,
        "exercise_minutes": 25,
        "stress_level": 7,
        "screen_time_hours": 5,
        "work_hours": 12,
        "junk_food_meals": 1,
        "water_intake_liters": 2.0,
    },
    "retired": {
        "sleep_hours": 8,
        "steps": 5000,
        "exercise_minutes": 30,
        "stress_level": 3,
        "screen_time_hours": 5,
        "work_hours": 2,
        "junk_food_meals": 1,
        "water_intake_liters": 2.0,
    },
    "general": {
        "sleep_hours": 7,
        "steps": 7000,
        "exercise_minutes": 30,
        "stress_level": 5,
        "screen_time_hours": 6,
        "work_hours": 8,
        "junk_food_meals": 1,
        "water_intake_liters": 2.0,
    },
}


def classify_persona(occupation: str) -> str:
    """Use the LLM to classify occupation into a persona category."""
    if not occupation or occupation.strip() == "":
        return "general"

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a classifier. Respond with exactly one word from this list: athlete, student, office_worker, parent, manual_laborer, healthcare_worker, retired, general. No punctuation, no explanation."
                },
                {
                    "role": "user",
                    "content": f"Classify this occupation into one category: {occupation}"
                }
            ],
            max_tokens=10
        )
        persona = response.choices[0].message.content.strip().lower()
        return persona if persona in PERSONAS else "general"
    except:
        return "general"


def calculate_bmi(user_data: dict) -> float | None:
    """Calculate BMI from height and weight if provided, otherwise use provided BMI."""
    # If explicit BMI provided, use it
    if user_data.get("bmi"):
        return user_data["bmi"]

    # Calculate from height + weight
    height_cm = user_data.get("height_cm")
    weight_kg = user_data.get("weight_kg")

    if height_cm and weight_kg and height_cm > 0:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        return round(bmi, 1)

    # Imperial: height_ft, height_in, weight_lbs
    height_ft = user_data.get("height_ft")
    height_in_extra = user_data.get("height_in", 0)
    weight_lbs = user_data.get("weight_lbs")

    if height_ft and weight_lbs:
        total_inches = (height_ft * 12) + (height_in_extra or 0)
        bmi = (weight_lbs / (total_inches ** 2)) * 703
        return round(bmi, 1)

    return None


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


def calculate_sub_scores(user_data, persona="general"):
    """
    Calculate sub-scores using the RF model with persona-adjusted baselines
    so scores reflect what's realistic for this person's life situation.
    """
    persona_base = PERSONA_BASELINES.get(persona, PERSONA_BASELINES["general"])

    baseline = {
        "age": user_data["age"],
        "sex": user_data["sex"],
        "bmi": calculate_bmi(user_data) or 24,
        "fitness_level": user_data.get("fitness_level") or 1,
        "sleep_consistency": 0.7,
        "sleep_hours": persona_base["sleep_hours"],
        "steps": persona_base["steps"],
        "exercise_minutes": persona_base["exercise_minutes"],
        "meals": 3,
        "junk_food_meals": persona_base["junk_food_meals"],
        "water_intake_liters": persona_base["water_intake_liters"],
        "caloric_intake": 2200,
        "screen_time_hours": persona_base["screen_time_hours"],
        "work_hours": persona_base["work_hours"],
        "stress_level": persona_base["stress_level"],
    }
    baseline_score = predict_score(baseline)

    sleep = user_data["sleep_hours"]
    steps = user_data["steps"]
    exercise = user_data["exercise_minutes"]
    stress = user_data["stress_level"]
    screen = user_data["screen_time_hours"]
    work = user_data["work_hours"]
    junk = user_data["junk_food_meals"]
    water = user_data.get("water_intake_liters") or 2.0
    calories = user_data.get("caloric_intake") or 2200
    consistency = user_data.get("sleep_consistency") or 0.7

    def score_dimension(overrides):
        d = baseline.copy()
        d.update(overrides)
        raw = predict_score(d) - baseline_score + 50
        return round(min(100, max(0, raw)))

    return {
        "Sleep Quality": score_dimension({
            "sleep_hours": sleep,
            "sleep_consistency": consistency,
        }),
        "Physical Activity": score_dimension({
            "steps": steps,
            "exercise_minutes": exercise,
        }),
        "Diet & Nutrition": score_dimension({
            "junk_food_meals": junk,
            "water_intake_liters": water,
            "caloric_intake": calories,
        }),
        "Recovery & Stress": score_dimension({
            "stress_level": stress,
            "sleep_hours": sleep,
        }),
        "Work-Life Balance": score_dimension({
            "work_hours": work,
            "screen_time_hours": screen,
            "stress_level": stress,
        }),
    }


def calculate_counterfactuals(user_data, current_score):
    """Dynamically pick the 4 most improvable inputs, rank by impact, build combined from top 3."""

    sleep = user_data["sleep_hours"]
    screen = user_data["screen_time_hours"]
    stress = user_data["stress_level"]
    exercise = user_data["exercise_minutes"]
    steps = user_data["steps"]
    water = user_data.get("water_intake_liters") or 2.0
    junk = user_data["junk_food_meals"]

    candidate_scenarios = {}

    if sleep < 8.5:
        target_sleep = min(sleep + 1.5, 9)
        candidate_scenarios[f"Increase sleep to {target_sleep}h"] = \
            run_counterfactual(user_data, {"sleep_hours": target_sleep})

    if screen > 2:
        target_screen = max(screen - 3, 0)
        candidate_scenarios[f"Reduce screen time to {target_screen}h"] = \
            run_counterfactual(user_data, {"screen_time_hours": target_screen})

    if stress > 2:
        target_stress = max(stress - 2, 1)
        candidate_scenarios[f"Reduce stress to {target_stress}/10"] = \
            run_counterfactual(user_data, {"stress_level": target_stress})

    if exercise < 100:
        target_exercise = min(exercise + 20, 120)
        candidate_scenarios[f"Add 20 min exercise ({target_exercise} min total)"] = \
            run_counterfactual(user_data, {"exercise_minutes": target_exercise})

    if steps < 10000 and exercise < 60:
        target_steps = min(steps + 3000, 12000)
        candidate_scenarios[f"Increase steps to {target_steps}"] = \
            run_counterfactual(user_data, {"steps": target_steps})

    if water < 2.5:
        target_water = min(water + 1.0, 4.0)
        candidate_scenarios[f"Increase water to {target_water}L"] = \
            run_counterfactual(user_data, {"water_intake_liters": target_water})

    if junk > 0:
        target_junk = max(junk - 1, 0)
        candidate_scenarios[f"Reduce junk food to {target_junk} meals"] = \
            run_counterfactual(user_data, {"junk_food_meals": target_junk})

    if not candidate_scenarios:
        candidate_scenarios["Increase sleep to 9h"] = \
            run_counterfactual(user_data, {"sleep_hours": 9})
        candidate_scenarios["Reduce stress to 1/10"] = \
            run_counterfactual(user_data, {"stress_level": 1})

    ranked = sorted(candidate_scenarios.items(), key=lambda x: x[1] - current_score, reverse=True)[:4]

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
        elif "steps" in label:
            top_3_overrides["steps"] = min(steps + 3000, 12000)
        elif "water" in label:
            top_3_overrides["water_intake_liters"] = min(water + 1.0, 4.0)
        elif "junk" in label:
            top_3_overrides["junk_food_meals"] = max(junk - 1, 0)

    combined_score = run_counterfactual(user_data, top_3_overrides)
    results = ranked + [("Combine all top 3 changes", combined_score)]

    return [(label, score, round(score - current_score, 1)) for label, score in results]


def generate_report(score, sub_scores, cf_results, user_data, occupation="", persona="general", user_context="", physical_stats=None):
    """Call the LLM to write the final report using pre-calculated data."""
    cf_text  = "\n".join([
        f"- {label}: predicted score {pred} ({'+' if delta > 0 else ''}{delta} points)"
        for label, pred, delta in cf_results
    ])
    sub_text = "\n".join([f"- {k}: {v}/100" for k, v in sub_scores.items()])

    occupation_line = f"Occupation: {occupation}" if occupation else ""
    persona_line = f"Lifestyle category: {persona.replace('_', ' ')}"
    context_line = f"User context: {user_context}" if user_context else ""
    personal_context = "\n".join(filter(None, [occupation_line, persona_line, context_line]))

    prompt = f"""
You are a health analytics platform generating a personalized daily report. Be analytical, direct, and specific. Do not ask questions. Do not end with "how do you feel." Lead with findings, not warmth.

CRITICAL PERSONALIZATION RULES — VIOLATIONS WILL MAKE THIS REPORT USELESS:
1. You MUST reference the user's occupation and context in EVERY section — not just once at the start.
2. If the user is an athlete, NEVER suggest "take a walk" or "add steps" — they already have extreme physical output. Focus on recovery, sleep, and nutrition instead.
3. If the user mentions exams, competitions, or high-stress events, acknowledge them by name and adjust recommendations around them.
4. Every single recommendation must be impossible to give to a different person — it must reference their exact occupation, context, and numbers.
5. If occupation contains a sport, address training load, recovery, and sport-specific nutrition — not generic fitness advice.
6. NEVER give advice that contradicts the user's persona. An athlete getting "add 20 min exercise" is a failure. A parent getting "reduce work hours" without acknowledging childcare is a failure.
7. If height, weight, or body fat % are provided, reference them when giving nutrition and recovery advice — a 155lb swimmer needs completely different caloric guidance than a 220lb manual laborer.
8. If BMI is provided or calculated, use it to contextualize nutrition recommendations — but never shame the user about their body composition.

{personal_context}

USER DATA:
- Overall routine score: {score}/100
- Sleep: {user_data['sleep_hours']}h | Steps: {user_data['steps']} | Exercise: {user_data['exercise_minutes']} min
- Stress: {user_data['stress_level']}/10 | Screen time: {user_data['screen_time_hours']}h | Work: {user_data['work_hours']}h
- Junk food meals: {user_data['junk_food_meals']} | Water: {user_data['water_intake_liters']}L
- BMI: {user_data.get('bmi', 'not provided')} | Fitness level: {user_data.get('fitness_level', 'not provided')}
- Height: {str(int(physical_stats.get('height_ft', 0))) + 'ft ' + str(int(physical_stats.get('height_in', 0))) + 'in' if physical_stats and physical_stats.get('height_ft') else ('not provided')} | Weight: {str(physical_stats.get('weight_lbs', '')) + 'lbs' if physical_stats and physical_stats.get('weight_lbs') else 'not provided'} | Body fat: {str(physical_stats.get('body_fat_pct', '')) + '%' if physical_stats and physical_stats.get('body_fat_pct') else 'not provided'}

PERSONA CONTEXT (use this to make every recommendation specific):
- Occupation: {occupation if occupation else 'not provided'}
- Lifestyle category: {persona.replace('_', ' ')}
- User's own words about their situation: "{user_context if user_context else 'not provided'}"

PERSONA-SPECIFIC RULES FOR THIS USER:
{"- This is an ATHLETE. Do NOT suggest adding steps or basic exercise. Focus exclusively on sleep quality, recovery, nutrition timing, and stress management as they relate to training load. Reference their specific sport if mentioned." if persona == "athlete" else ""}{"- This is a STUDENT. Acknowledge academic pressure explicitly. Frame all recommendations around study schedules and academic calendar. If exams are mentioned, factor that into stress recommendations." if persona == "student" else ""}{"- This is a PARENT. Acknowledge that reduced work-life balance may be structural, not a choice. Focus on high-leverage small changes that fit around family responsibilities." if persona == "parent" else ""}{"- This is a HEALTHCARE WORKER. Acknowledge irregular hours and shift work as a root cause of poor sleep and stress, not a personal failure. Recommendations must be realistic for someone with unpredictable schedules." if persona == "healthcare_worker" else ""}{"- This is a MANUAL LABORER. Their physical activity at work is already extremely high. Do not suggest more exercise — focus entirely on recovery, hydration, nutrition, and sleep quality." if persona == "manual_laborer" else ""}{"- This is a RETIRED person. They have more time flexibility than most. Recommendations should leverage this — longer sleep windows, midday exercise, meal prep time." if persona == "retired" else ""}{"- This is an OFFICE WORKER. Sedentary work is a core problem. Address the specific health risks of sitting 8+ hours — back health, eye strain from screens, metabolic slowdown." if persona == "office_worker" else ""}

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
    """Main entry point — validates input, classifies persona, runs all calculations."""
    # Extract string fields before validation
    occupation = user_data.pop("occupation", "") or ""
    user_context = user_data.pop("user_context", "") or ""

    # Calculate BMI from height/weight if not provided
    computed_bmi = calculate_bmi(user_data)
    if computed_bmi:
        user_data["bmi"] = computed_bmi

    # Store physical stats before popping for use in report
    physical_stats = {
        "height_ft": user_data.get("height_ft"),
        "height_in": user_data.get("height_in", 0),
        "weight_lbs": user_data.get("weight_lbs"),
        "height_cm": user_data.get("height_cm"),
        "weight_kg": user_data.get("weight_kg"),
        "body_fat_pct": user_data.get("body_fat_pct"),
    }

    # Remove height/weight fields before model inference
    for field in ["height_cm", "weight_kg", "height_ft", "height_in", "weight_lbs"]:
        user_data.pop(field, None)

    validate_inputs(user_data)

    # Classify persona from occupation
    persona = classify_persona(occupation) if occupation else "general"

    score = predict_score(user_data)
    sub_scores = calculate_sub_scores(user_data, persona)
    cf_results = calculate_counterfactuals(user_data, score)
    report = generate_report(score, sub_scores, cf_results, user_data, occupation, persona, user_context, physical_stats)

    return score, sub_scores, cf_results, report, persona


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
        "meals": 3,
        "occupation": "high school student and competitive swimmer",
        "user_context": "I train twice a day and have AP exams coming up",
        "height_ft": 5,
        "height_in": 10,
        "weight_lbs": 155,
    }

    score, sub_scores, cf_results, report, persona = generate_recommendation(test_data)

    print(f"Predicted Score: {score}/100 | Persona: {persona}\n")

    print("Sub-scores:")
    for k, v in sub_scores.items():
        print(f"  {k}: {v}/100")

    print("\nCounterfactuals:")
    for label, pred, delta in cf_results:
        print(f"  {label}: {pred} ({'+' if delta > 0 else ''}{delta})")

    print("\n--- REPORT ---")
    print(report)
