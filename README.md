# Health Routine Analyzer

I built this because every health app I've used just tracks things without telling you what to actually do about them. This one scores your daily routine using a machine learning model and then simulates what happens to your score if you make specific changes — "sleep 1.5 more hours and your score goes from 47 to 55." No vague advice, no streaks, just numbers.

**Live:** [v0-health-routine-analyzer.vercel.app](https://v0-health-routine-analyzer.vercel.app)

---

## The idea

Most habit trackers are glorified checklists. The thing that makes this different is the counterfactual engine — instead of telling you to "sleep more," it runs the ML model with your modified inputs and shows you the exact predicted score change. It also breaks your overall score into 5 sub-scores and explains the causal chain between them (why high stress leads to poor sleep leads to low recovery, using your specific numbers).

The AI report at the end isn't generic. It's generated with your actual values plugged into the prompt, so every recommendation references the exact numbers you logged.

---

## How it works

You fill out a form with 10-15 inputs about your day — sleep, steps, stress, screen time, etc. That gets sent to a FastAPI backend which:

1. Validates the inputs and fills in missing optional fields via mean imputation
2. Runs a trained Random Forest model to predict your routine score (0-100)
3. Calculates 5 sub-scores using domain formulas
4. Runs 4 counterfactual simulations + 1 combined scenario
5. Calls Groq (Llama 4 Scout) to write a personalized report with a specific action plan

The whole thing returns in a few seconds.

---

## Stack

- **Model** — scikit-learn RandomForestRegressor, trained on 60k synthetic records, R² = 0.95
- **Backend** — FastAPI on Render
- **LLM** — Groq API (meta-llama/llama-4-scout-17b-16e-instruct)
- **Frontend** — Next.js on Vercel

---

## Project structure

```
health-ai-platform/
├── api.py                  # FastAPI — /predict endpoint
├── recommender_ai.py       # scoring, counterfactuals, LLM report
├── train_model.py          # training script
├── health_model.pkl        # trained pipeline (compressed)
├── feature_columns.pkl     # column order from training
├── requirements.txt
└── runtime.txt             # pins Python 3.11 for Render
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

Docs at `http://localhost:8000/docs`

---

## API

`POST /predict` — takes a JSON body, returns score + sub-scores + counterfactuals + report.

Required fields: `age`, `sex`, `sleep_hours`, `steps`, `exercise_minutes`, `stress_level`, `screen_time_hours`, `work_hours`, `junk_food_meals`, `water_intake_liters`

Optional: `bmi`, `fitness_level`, `sleep_consistency`, `caloric_intake`, `meals`

---

## Model

Trained on a synthetic dataset of 60,000 daily routine records generated in R with realistic behavioral correlations. Three features are engineered at inference time: `exercise_ratio`, `water_per_meal`, `sleep_efficiency`.

| Metric | Value |
|--------|-------|
| MAE    | 2.21  |
| RMSE   | 2.79  |
| R²     | 0.95  |

---

## What's next

The current version is stateless — no accounts, no history. The next phase adds Supabase for user auth and daily score logging, which unlocks trend tracking, goal setting, and a social leaderboard. After that, context mode (adjusting expectations for finals week, travel, illness) and a weekly AI summary email.

---

Built by [Kunal Chevuri](https://github.com/kunalchevuri)
