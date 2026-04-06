# Health Routine Analyzer

**An ML-powered health scoring platform that predicts your daily routine score, explains the causal chain behind it, and simulates exactly what happens to your score if you make specific changes.**

**Frontend:** [health-ai-frontend-eight.vercel.app](https://health-ai-frontend-eight.vercel.app) &nbsp;|&nbsp; **API Docs:** [health-ai-platform-n7a0.onrender.com/docs](https://health-ai-platform-n7a0.onrender.com/docs)

---

## What makes it different

Most health apps track habits and show you charts. This one does three things most apps don't:

**1. Counterfactual engine**
Instead of generic advice, it re-runs the ML model with modified inputs and shows the exact predicted score change. "Sleep 1.5 more hours → your score goes from 47 to 55." Every recommendation comes with a number, not a suggestion. The engine dynamically selects only the changes where there's meaningful room to improve — it won't tell you to add steps if you're already doing 15,000 a day.

**2. Persona system**
You enter your occupation as free text. A Groq LLM classifies it into one of 8 personas (athlete, student, office worker, parent, manual laborer, healthcare worker, retired, general) and the scoring baseline adjusts to what's realistic for your lifestyle. An athlete is benchmarked against athlete norms — not against a generic desk worker. A competitive swimmer won't get a "take a walk" recommendation. A student during finals week gets recommendations built around their academic calendar.

**3. Context-aware reports**
You describe your situation in plain text ("I train twice a day, AP exams next week"). Every section of the AI-generated report references your specific occupation, context, and exact numbers. The report is structured to be impossible to give to a different person — it connects every recommendation directly to the values you logged.

---

## How it works

```
User fills out form
  └── sleep, steps, exercise, stress, screen time, work hours,
      junk food, water, caloric intake, occupation, context,
      height, weight (or direct BMI)
         │
         ▼
Backend (FastAPI)
  ├── 1. Classify occupation → persona via Groq LLM
  ├── 2. Calculate BMI from height/weight (metric or imperial)
  ├── 3. Run MLP Neural Network → routine score (0–100)
  ├── 4. Calculate 5 sub-scores using persona-adjusted baselines
  │       Sleep Quality · Physical Activity · Diet & Nutrition
  │       Recovery & Stress · Work-Life Balance
  ├── 5. Run dynamic counterfactual simulations
  │       (only suggests changes with room to improve,
  │        persona-aware: athletes skip steps recommendations)
  └── 6. Call Groq (Llama 4 Scout) → personalized report
          with exact numbers, causal chain analysis,
          and persona-specific action plan
```

---

## Model

Three architectures were trained and compared with 5-fold cross-validation on 60,385 records:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 2.185 | 2.765 | 0.9512 |
| XGBoost | 1.636 | 2.057 | 0.9730 |
| **MLP Neural Network** | **1.583** | **1.986** | **0.9748** |

**Winner: MLP Neural Network** — 3 hidden layers (128 → 64 → 32), ReLU activation, Adam optimizer, early stopping (20 rounds patience, 10% validation split).

**Dataset: 60,385 records total**
- 60,000 synthetic records generated with realistic behavioral correlations across sleep, activity, diet, stress, and recovery variables
- 385 real Fitbit records from the Fitabase dataset — actual step counts, active minutes, and sedentary minutes from device logs; sleep hours aggregated from 198,559 minute-level sleep entries (Fitbit sleep staging: asleep/restless/awake); real BMI from weight logs for 11 users

**23 engineered features:**

| Feature | Description |
|---|---|
| `activity_score` | `steps/1000 + exercise_minutes/30` |
| `diet_quality` | `10 - junk_food - abs(calories-2200)/400` |
| `recovery_score` | `sleep_hours - stress_level/2` |
| `sedentary_score` | `screen_time_hours - steps/5000` |
| `work_life_balance` | `work_hours + stress_level` |
| `exercise_ratio` | `exercise_minutes / (steps+1)` |
| `water_per_meal` | `water_liters / (meals+0.01)` |
| `sleep_efficiency` | `sleep_hours / (stress_level+1)` |
| + 15 raw inputs | age, sex, BMI, fitness level, sleep consistency, sleep hours, steps, exercise minutes, meals, junk food meals, water intake, caloric intake, screen time, work hours, stress level |

---

## Persona system

| Persona | Classified from occupations like... | Scoring adjusts for... |
|---|---|---|
| Athlete | competitive swimmer, marathon runner, pro soccer player | High exercise baseline, recovery-focused benchmarks |
| Student | high school student, college sophomore, grad student | Academic stress, irregular sleep, low income diet |
| Office Worker | software engineer, accountant, data analyst | Sedentary work, screen time, 9-to-5 structure |
| Parent | stay-at-home mom, father of three, single parent | Reduced sleep baseline, high work hours structural |
| Manual Laborer | construction worker, warehouse associate, nurse aide | High physical activity already counted in work |
| Healthcare Worker | ER nurse, physician, physical therapist | Shift work, irregular hours, high stress baseline |
| Retired | retired teacher, former engineer, senior | Time flexibility, lower work hours, moderate activity |
| General | anything else | Population average baselines |

---

## API

**`POST /predict`** — Full docs at [health-ai-platform-n7a0.onrender.com/docs](https://health-ai-platform-n7a0.onrender.com/docs)

**Required fields:**
```json
{
  "age": 25,
  "sex": 0,
  "sleep_hours": 6.5,
  "steps": 8000,
  "exercise_minutes": 45,
  "stress_level": 6,
  "screen_time_hours": 7,
  "work_hours": 9,
  "junk_food_meals": 1,
  "water_intake_liters": 2.0
}
```

**Optional fields:** `bmi`, `fitness_level` (0/1/2), `sleep_consistency` (0–1), `caloric_intake`, `meals`, `occupation` (free text), `user_context` (free text), `height_ft`, `height_in`, `weight_lbs`, `height_cm`, `weight_kg`, `body_fat_pct`

**Response:**
```json
{
  "routine_score": 54.2,
  "persona": "office_worker",
  "sub_scores": {
    "Sleep Quality": 48,
    "Physical Activity": 62,
    "Diet & Nutrition": 55,
    "Recovery & Stress": 44,
    "Work-Life Balance": 41
  },
  "counterfactuals": [
    { "label": "Increase sleep to 8.0h", "predicted_score": 61.7, "delta": 7.5 },
    { "label": "Reduce screen time to 4h", "predicted_score": 57.3, "delta": 3.1 }
  ],
  "report": "..."
}
```

---

## Stack

| Layer | Technology |
|---|---|
| ML model | scikit-learn `MLPRegressor` |
| Model comparison | Random Forest, XGBoost, MLP (5-fold CV) |
| Backend | FastAPI, deployed on Render |
| LLM | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Frontend | Next.js, deployed on Vercel |
| Language | Python 3.11 (backend), TypeScript (frontend) |

---

## Project structure

```
health-ai-platform/
├── api.py                      # FastAPI app — /predict endpoint, input validation, response schema
├── recommender_ai.py           # Core engine: persona classification, BMI calc, scoring,
│                               #   sub-scores, counterfactual simulations, LLM report generation
├── train_model.py              # Original training script (Random Forest baseline)
├── train_model_v2.py           # Multi-architecture comparison: RF vs XGBoost vs MLP, 5-fold CV
├── build_real_dataset.py       # Builds enhanced dataset: synthetic + real Fitbit data from zip,
│                               #   aggregates minute-level sleep, merges activity + sleep + weight
├── health_model.pkl            # Trained MLP pipeline (best model, deployed)
├── feature_columns.pkl         # Exact 23-column order used at training and inference
├── routine_dataset.csv         # 60,000 synthetic training records
├── routine_dataset_enhanced.csv # 60,385 records (synthetic + real Fitbit)
└── requirements.txt            # Python dependencies
```

---

## Running locally

```bash
git clone https://github.com/kunalchevuri/health-ai-platform.git
cd health-ai-platform
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
uvicorn api:app --reload
```

API docs available at `http://localhost:8000/docs`

To retrain the model:
```bash
python build_real_dataset.py   # rebuild enhanced dataset
python train_model_v2.py       # compare architectures and deploy best
```

---

## What's next

- **Supabase auth + daily score history** — persistent accounts, trend charts over time
- **7-day routine plan generator** — AI builds a week-long plan toward a target score
- **Streaks and gamification** — score improvement tracking, personal bests
- **NHANES dataset integration** — replace synthetic data with real population health survey data
- **Wearable sync** — direct Fitbit / Apple Health / Garmin integration via OAuth

---

Built by [Kunal Chevuri](https://github.com/kunalchevuri)
