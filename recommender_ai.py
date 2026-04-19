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


# The 5 fields a daily check-in updates (everything else blends from last full analysis)
CHECKIN_FIELDS = ["sleep_hours", "stress_level", "exercise_minutes", "steps", "screen_time_hours", "water_intake_liters"]


def apply_stressor_adjustments(persona_base: dict, stressors: list) -> dict:
    """Tighten persona baselines based on active stressors so 'good' means more for this user."""
    if not stressors:
        return persona_base
    adjusted = persona_base.copy()
    sl = [s.lower() for s in stressors]
    if any(k in sl for k in ["exam", "exams", "finals", "test", "midterms"]):
        # Exams make sleep and stress management more critical
        adjusted["sleep_hours"] = max(adjusted["sleep_hours"], 8.0)
        adjusted["stress_level"] = min(adjusted["stress_level"], 4)
    if any(k in sl for k in ["training", "competition", "game", "match", "tournament", "season"]):
        # Athletic competition → recovery window is tighter
        adjusted["sleep_hours"] = max(adjusted["sleep_hours"], 8.5)
        adjusted["water_intake_liters"] = max(adjusted["water_intake_liters"], 3.0)
    if any(k in sl for k in ["work", "deadline", "project", "overtime"]):
        adjusted["stress_level"] = min(adjusted["stress_level"], 5)
        adjusted["screen_time_hours"] = min(adjusted["screen_time_hours"], 6)
    return adjusted


def detect_trend_boosts(checkin_history: list) -> dict:
    """
    If a field has improved for 3+ consecutive check-ins, return a small score
    boost (+3 per field) to reflect momentum. Returns dict of {sub_score_name: boost}.
    """
    boosts = {}
    if not checkin_history or len(checkin_history) < 3:
        return boosts
    recent = checkin_history[-3:]
    trend_fields = {
        "sleep_hours":       ("Sleep Quality",    True),   # True = higher is better
        "exercise_minutes":  ("Physical Activity", True),
        "steps":             ("Physical Activity", True),
        "stress_level":      ("Recovery & Stress", False), # False = lower is better
        "screen_time_hours": ("Work-Life Balance", False),
        "water_intake_liters": ("Diet & Nutrition", True),
    }
    for field, (sub_score, higher_is_better) in trend_fields.items():
        vals = [d.get(field) for d in recent if d.get(field) is not None]
        if len(vals) < 3:
            continue
        improving = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)) if higher_is_better \
                    else all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        if improving:
            boosts[sub_score] = boosts.get(sub_score, 0) + 3
    return boosts


def calculate_sub_scores(user_data, persona="general", stressors=None, checkin_history=None):
    """
    Calculate sub-scores using the RF model with persona-adjusted baselines
    so scores reflect what's realistic for this person's life situation.
    Stressor weights tighten what 'baseline good' means. Trend boosts reward momentum.
    """
    persona_base = apply_stressor_adjustments(
        PERSONA_BASELINES.get(persona, PERSONA_BASELINES["general"]),
        stressors,
    )

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

    raw_scores = {
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

    # Apply trend boosts: reward 3+ consecutive days of improvement
    trend_boosts = detect_trend_boosts(checkin_history)
    return {
        k: min(100, v + trend_boosts.get(k, 0))
        for k, v in raw_scores.items()
    }


def calculate_counterfactuals(user_data, current_score, persona="general", stressors=None, goals=None):
    """
    Dynamically pick the 4 most improvable inputs, rank by impact, build combined from top 3.
    Filters suggestions to match what's relevant for this user's persona and stressors.
    """
    sleep = user_data["sleep_hours"]
    screen = user_data["screen_time_hours"]
    stress = user_data["stress_level"]
    exercise = user_data["exercise_minutes"]
    steps = user_data["steps"]
    water = user_data.get("water_intake_liters") or 2.0
    junk = user_data["junk_food_meals"]

    sl = [s.lower() for s in (stressors or [])]
    exam_stress = any(k in sl for k in ["exam", "exams", "finals", "test", "midterms"])
    athletic_load = persona == "athlete" or any(k in sl for k in ["training", "competition", "game", "match"])
    high_physical = persona in ("athlete", "manual_laborer")

    candidate_scenarios = {}

    # Sleep — always relevant; higher priority when exam stress or athletic recovery is active
    sleep_ceiling = 9.0 if (exam_stress or athletic_load) else 8.5
    if sleep < sleep_ceiling:
        target_sleep = min(sleep + 1.5, sleep_ceiling)
        label = (
            f"Extend sleep to {target_sleep}h (critical for exam recovery)"
            if exam_stress else
            f"Extend sleep to {target_sleep}h (recovery window for training)"
            if athletic_load else
            f"Increase sleep to {target_sleep}h"
        )
        candidate_scenarios[label] = run_counterfactual(user_data, {"sleep_hours": target_sleep})

    if screen > 2:
        target_screen = max(screen - 3, 0)
        label = (
            f"Reduce screen time to {target_screen}h (blue light blocks recovery sleep)"
            if athletic_load else
            f"Reduce screen time to {target_screen}h"
        )
        candidate_scenarios[label] = run_counterfactual(user_data, {"screen_time_hours": target_screen})

    if stress > 2:
        target_stress = max(stress - 2, 1)
        label = (
            f"Reduce stress to {target_stress}/10 via 10min breathing (specific to {sl[0]} stressor)"
            if sl else
            f"Reduce stress to {target_stress}/10"
        )
        candidate_scenarios[label] = run_counterfactual(user_data, {"stress_level": target_stress})

    # Exercise — skip for high-physical personas; they already have massive output
    if not high_physical and exercise < 100:
        target_exercise = min(exercise + 20, 120)
        candidate_scenarios[f"Add 20 min exercise ({target_exercise} min total)"] = \
            run_counterfactual(user_data, {"exercise_minutes": target_exercise})

    # Steps — skip for athletes/manual laborers
    if not high_physical and steps < 10000 and exercise < 60:
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


def _get_top_priorities(sub_scores: dict, persona: str, stressors: list, goals: list) -> list[str]:
    """
    Return the top-2 sub-score names that need the most attention for THIS user.
    Starts from lowest raw scores, then context-boosts based on stressors/goals/persona.
    Doing this in Python (not in the LLM) ensures reliable, testable prioritization.
    """
    sl = [s.lower() for s in (stressors or [])]
    gl = [g.lower() for g in (goals or [])]

    # Baseline: sort by lowest score (most room for improvement)
    order = [name for name, _ in sorted(sub_scores.items(), key=lambda x: x[1])]

    def boost(name):
        if name in order:
            order.remove(name)
            order.insert(0, name)

    # Stressor-based boosts
    if any(k in sl for k in ["exam", "exams", "finals", "test", "midterms"]):
        boost("Recovery & Stress")
        boost("Sleep Quality")        # sleep is #1 for exam performance
    if any(k in sl for k in ["training", "competition", "game", "match", "tournament", "season"]):
        boost("Sleep Quality")
        boost("Recovery & Stress")   # recovery is #1 for athletic load
    if any(k in sl for k in ["work", "deadline", "project", "overtime"]):
        boost("Work-Life Balance")

    # Persona-based boosts (applied after stressors so stressors take precedence)
    if persona == "athlete":
        boost("Recovery & Stress")
    elif persona == "office_worker":
        boost("Work-Life Balance")
    elif persona == "student":
        boost("Sleep Quality")
    elif persona == "parent":
        boost("Work-Life Balance")

    # Goal-based boosts (user's explicit intent wins over everything)
    if any("sleep" in g for g in gl):
        boost("Sleep Quality")
    if any("stress" in g for g in gl):
        boost("Recovery & Stress")
    if any(g in ["lose weight", "weight", "nutrition", "diet", "eat"] for g in gl):
        boost("Diet & Nutrition")
    if any(g in ["fitness", "active", "exercise", "steps"] for g in gl):
        boost("Physical Activity")

    return order[:2]


def _score_status(s: int) -> str:
    if s >= 80: return "Strong"
    if s >= 60: return "Decent"
    if s >= 40: return "Weak"
    return "Critical"


def generate_report(score, sub_scores, cf_results, user_data, occupation="", persona="general",
                    user_context="", physical_stats=None, stressors=None, goals=None,
                    grade_year=None, life_context=None, mode="full_analysis"):
    """Call the LLM to write the final structured report (max 300 words, bullets only)."""

    # Pre-compute priorities in Python so the LLM never has to guess
    top2 = _get_top_priorities(sub_scores, persona, stressors or [], goals or [])
    p1_name, p1_score = top2[0], sub_scores[top2[0]]
    p2_name, p2_score = top2[1], sub_scores[top2[1]]

    # Best single counterfactual (highest delta, excluding the combined scenario)
    individual_cfs = [(l, p, d) for l, p, d in cf_results if "Combine" not in l]
    best_cf = max(individual_cfs, key=lambda x: x[2]) if individual_cfs else cf_results[0]

    # Build compact data block — only what the LLM needs
    sub_lines   = "\n".join(f"• {k}: {v}/100 ({_score_status(v)})" for k, v in sub_scores.items())
    cf_lines    = "\n".join(
        f"• {l}: {p}/100 ({'+' if d > 0 else ''}{d} pts)"
        for l, p, d in cf_results if "Combine" not in l
    )
    physical_line = ""
    if physical_stats:
        parts = []
        if physical_stats.get("height_ft"):
            parts.append(f"{int(physical_stats['height_ft'])}ft {int(physical_stats.get('height_in', 0))}in")
        if physical_stats.get("weight_lbs"):
            parts.append(f"{physical_stats['weight_lbs']}lbs")
        if physical_stats.get("body_fat_pct"):
            parts.append(f"{physical_stats['body_fat_pct']}% body fat")
        if parts:
            physical_line = "• Physical: " + ", ".join(parts)

    # Overall score status label
    if score >= 80:   overall_status = "Excellent"
    elif score >= 60: overall_status = "Good"
    elif score >= 40: overall_status = "Moderate"
    else:             overall_status = "Poor"

    # Context description for the system prompt
    who = f"{occupation} ({grade_year})" if grade_year else (occupation or persona.replace("_", " "))
    stressor_str = ", ".join(stressors) if stressors else "none reported"
    goal_str     = ", ".join(goals)     if goals     else "none reported"

    system_prompt = f"""You are a concise health coach for a {who}.
Situation: {life_context or user_context or 'not provided'}
Active stressors: {stressor_str}
Goals: {goal_str}

RULES (violations will make the report useless):
- NO paragraphs. Bullets and short sentences only.
- Max 300 words total. Stop after the last section.
- Every recommendation must name a specific number from their data (e.g. "your {user_data['sleep_hours']}h of sleep").
- {"NEVER suggest adding exercise or steps — this is an athlete with extreme physical output already." if persona == "athlete" else ""}
- {"Frame sleep as a performance enhancer for cognitive retention during exams — not just health advice." if stressors and any(k in [s.lower() for s in stressors] for k in ["exam","exams","finals","test","midterms"]) else ""}
- {"Address recovery and nutrition timing around training load specifically." if stressors and any(k in [s.lower() for s in stressors] for k in ["training","competition","game","match","tournament"]) else ""}
- Use exact numbers from the data. Do not round or change any score."""

    user_prompt = f"""Generate this user's health report now.

DATA:
• Overall: {score}/100 — {overall_status}
• Sleep: {user_data['sleep_hours']}h | Steps: {user_data['steps']} | Exercise: {user_data['exercise_minutes']} min
• Stress: {user_data['stress_level']}/10 | Screen: {user_data['screen_time_hours']}h | Work: {user_data['work_hours']}h
• Junk food: {user_data['junk_food_meals']} meals | Water: {user_data['water_intake_liters']}L
{physical_line}

SUB-SCORES (use exactly):
{sub_lines}

TOP COUNTERFACTUALS (use exactly):
{cf_lines}

PRE-DETERMINED PRIORITIES (use these — do not reorder):
Priority 1: {p1_name} ({p1_score}/100)
Priority 2: {p2_name} ({p2_score}/100)
Best single move: {best_cf[0]} → {best_cf[1]}/100 (+{best_cf[2]} pts)

OUTPUT FORMAT (copy these headers exactly, fill in content):

📊 OVERALL SCORE: {score}/100 — "{overall_status}"
[One sentence on what's driving the score, referencing the two lowest sub-scores by name and number]

🎯 YOUR TOP 2 PRIORITIES
1. {p1_name} ({p1_score}/100): [One sentence explaining WHY this is the bottleneck, referencing their specific numbers and context — mention {stressor_str if stressors else 'their situation'} explicitly]
→ [Specific action with exact target number, tied to their life]
2. {p2_name} ({p2_score}/100): [One sentence explanation referencing their context]
→ [Specific action with exact target number]

💪 WHAT'S WORKING
[List the 2 highest sub-scores as bullets with score and one short reason]

⚠️ WHAT'S DRAGGING YOU DOWN
[List the 2 lowest sub-scores as bullets with score and one short cause]

🚀 IF YOU DID THIS TODAY
{best_cf[0]} → could bring your score to {best_cf[1]}/100 (+{best_cf[2]} pts)
[One sentence on exactly how to do this action today, tied to their specific situation]"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=500,   # ~375 words hard ceiling; keeps us well under 300 useful words
            temperature=0.4,  # low temp = consistent structure, less hallucination
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"Report generation failed: {str(e)}\n\n"
            f"📊 OVERALL SCORE: {score}/100 — {overall_status}\n"
            f"Top priorities: {p1_name} ({p1_score}/100), {p2_name} ({p2_score}/100)"
        )


def generate_recommendation(user_data, mode="full_analysis", last_analysis=None, checkin_history=None):
    """
    Main entry point — validates input, classifies persona, runs all calculations.

    mode="full_analysis"  → uses all submitted fields (7-day re-analysis)
    mode="daily_checkin"  → blends last_analysis with today's 5 slider fields, skips full re-analysis
    """
    # Extract string/list context fields before validation (model never sees these)
    occupation   = user_data.pop("occupation",   "") or ""
    user_context = user_data.pop("user_context", "") or ""
    grade_year   = user_data.pop("grade_year",   None)
    stressors    = user_data.pop("stressors",    None)
    goals        = user_data.pop("goals",        None)
    life_context = user_data.pop("life_context", None)

    # --- DAILY CHECK-IN: blend last full-analysis data with today's slider values ---
    if mode == "daily_checkin" and last_analysis:
        # Start from the previous full analysis snapshot
        blended = {k: v for k, v in last_analysis.items() if v is not None}
        # Override only the fields the user updated today
        for field in CHECKIN_FIELDS:
            if field in user_data and user_data[field] is not None:
                blended[field] = user_data[field]
        user_data = blended

    # Calculate BMI from height/weight if not provided
    computed_bmi = calculate_bmi(user_data)
    if computed_bmi:
        user_data["bmi"] = computed_bmi

    # Store physical stats for report display before stripping them from model input
    physical_stats = {
        "height_ft":    user_data.get("height_ft"),
        "height_in":    user_data.get("height_in", 0),
        "weight_lbs":   user_data.get("weight_lbs"),
        "height_cm":    user_data.get("height_cm"),
        "weight_kg":    user_data.get("weight_kg"),
        "body_fat_pct": user_data.get("body_fat_pct"),
    }
    for field in ["height_cm", "weight_kg", "height_ft", "height_in", "weight_lbs"]:
        user_data.pop(field, None)

    validate_inputs(user_data)

    # Classify persona from occupation string
    persona = classify_persona(occupation) if occupation else "general"

    score     = predict_score(user_data)
    sub_scores = calculate_sub_scores(user_data, persona, stressors=stressors, checkin_history=checkin_history)
    cf_results = calculate_counterfactuals(user_data, score, persona=persona, stressors=stressors, goals=goals)
    report    = generate_report(
        score, sub_scores, cf_results, user_data,
        occupation=occupation, persona=persona, user_context=user_context,
        physical_stats=physical_stats, stressors=stressors, goals=goals,
        grade_year=grade_year, life_context=life_context, mode=mode,
    )

    return score, sub_scores, cf_results, report, persona


# Test it
if __name__ == "__main__":
    base_data = {
        "age": 17,
        "sex": 0,
        "bmi": 22,
        "fitness_level": 3,
        "sleep_consistency": 0.6,
        "sleep_hours": 5.5,
        "steps": 12000,
        "exercise_minutes": 90,
        "stress_level": 8,
        "screen_time_hours": 6,
        "work_hours": 7,
        "junk_food_meals": 1,
        "water_intake_liters": 2.5,
        "caloric_intake": 3200,
        "meals": 4,
        "occupation": "high school student and competitive swimmer",
        "grade_year": "junior",
        "stressors": ["exams", "soccer training"],
        "goals": ["improve sleep", "reduce stress"],
        "life_context": "I train twice a day and have AP exams coming up",
        "height_ft": 5,
        "height_in": 10,
        "weight_lbs": 155,
    }

    print("=== FULL ANALYSIS ===")
    score, sub_scores, cf_results, report, persona = generate_recommendation(
        base_data.copy(), mode="full_analysis"
    )
    print(f"Score: {score}/100 | Persona: {persona}")
    for k, v in sub_scores.items():
        print(f"  {k}: {v}/100")
    for label, pred, delta in cf_results:
        print(f"  {label}: {pred} ({'+' if delta > 0 else ''}{delta})")

    # Simulate a check-in the next day: user slept 1h more and reduced stress
    print("\n=== DAILY CHECK-IN (sleep +1h, stress -1) ===")
    last_analysis_snapshot = {
        "age": 17, "sex": 0, "bmi": 22, "fitness_level": 3, "sleep_consistency": 0.6,
        "sleep_hours": 5.5, "steps": 12000, "exercise_minutes": 90, "stress_level": 8,
        "screen_time_hours": 6, "work_hours": 7, "junk_food_meals": 1,
        "water_intake_liters": 2.5, "caloric_intake": 3200, "meals": 4,
    }
    checkin_data = {
        "sleep_hours": 6.5,
        "stress_level": 7,
        "steps": 11000,
        "exercise_minutes": 90,
        "screen_time_hours": 5,
        "occupation": "high school student and competitive swimmer",
        "stressors": ["exams", "soccer training"],
        "goals": ["improve sleep", "reduce stress"],
    }
    score2, sub2, cf2, _, persona2 = generate_recommendation(
        checkin_data, mode="daily_checkin", last_analysis=last_analysis_snapshot
    )
    print(f"Check-in Score: {score2}/100 (was {score}/100)")
    for k, v in sub2.items():
        print(f"  {k}: {v}/100")
