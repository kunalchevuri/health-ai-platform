from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from recommender_ai import generate_recommendation

app = FastAPI()

# Allow frontend apps (Base44, Vercel, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthInput(BaseModel):
    age: Optional[float] = None
    sex: Optional[float] = None                    # 0 = Male, 1 = Female
    bmi: Optional[float] = None
    fitness_level: Optional[float] = None          # 0 = Beginner, 1 = Intermediate, 2 = Advanced
    sleep_consistency: Optional[float] = None
    sleep_hours: Optional[float] = None
    steps: Optional[float] = None
    exercise_minutes: Optional[float] = None
    meals: Optional[float] = None
    junk_food_meals: Optional[float] = None
    water_intake_liters: Optional[float] = None
    caloric_intake: Optional[float] = None
    screen_time_hours: Optional[float] = None
    work_hours: Optional[float] = None
    stress_level: Optional[float] = None


@app.get("/")
def home():
    return {"message": "Health AI API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Convert to dict and strip None values
        user_data = {k: v for k, v in data.model_dump().items() if v is not None}

        # Run the full pipeline — validation, scoring, counterfactuals, AI report
        score, sub_scores, cf_results, report = generate_recommendation(user_data)

        # Format counterfactuals for JSON
        counterfactuals = [
            {"label": label, "predicted_score": pred, "delta": delta}
            for label, pred, delta in cf_results
        ]

        return {
            "routine_score": score,
            "sub_scores": sub_scores,
            "counterfactuals": counterfactuals,
            "report": report
        }

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})
