
# ============================================================

"""
📘 Description:
This FastAPI app exposes a REST endpoint /predict that accepts
citizen demographic and income data, passes it through a trained
ML model (from CharityML Project), and returns a JSON response
with the predicted donation probability (%).

⚙ Features:
- Automatic interactive documentation at /docs (Swagger UI)
- Validation via Pydantic models
- Detailed responses with confidence levels
- Logging & error handling
- CORS enabled for front-end integration
"""

# =========================
# 1️⃣ Import Dependencies
# =========================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import logging
import uvicorn
from typing import Optional

# =========================
# 2️⃣ Initialize Application
# =========================
app = FastAPI(
    title="CharityML Donation Prediction API",
    description="Predicts the probability of a person donating to charity based on demographic and income data.",
    version="2.0.0",
    contact={
        "name": "Shahinda",
        "url": "https://github.com/Shahinda383",
        "email": "shahindaibrahim52@gmail.com",
    },
)

# =========================
# 3️⃣ Configure Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================
# 4️⃣ CORS Configuration
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 5️⃣ Load Trained Model
# =========================
try:
    model = joblib.load("final_automl_best_model.joblib")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error("❌ Failed to load model: %s", e)
    raise RuntimeError("Model file not found or corrupted.")

# =========================
# 6️⃣ Define Input Schema
# =========================
class InputData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the individual")
    workclass: str = Field(..., description="Type of employment")
    education_num: int = Field(..., ge=1, le=16, description="Years of education")
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(..., ge=0)
    capital_loss: float = Field(..., ge=0)
    hours_per_week: float = Field(..., ge=0, le=100)
    native_country: str

# =========================
# 7️⃣ Define Output Schema
# =========================
class PredictionResponse(BaseModel):
    donation_probability: float
    donation_decision: str
    confidence_level: str
    message: Optional[str] = "Prediction completed successfully."

# =========================
# 8️⃣ Root Endpoint
# =========================
@app.get("/", tags=["Root"])
def read_root():
    """
    🏠 Root endpoint showing basic project information.
    """
    return {
        "message": "Welcome to CharityML Donation Prediction API 🌟",
        "documentation": "Go to /docs for interactive API testing.",
        "author": "Shahinda",
        "version": "2.0.0"
    }

# =========================
# 9️⃣ Prediction Endpoint
# =========================
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_donation(data: InputData):
    """
    🎯 Predict whether an individual will donate to charity.
    """
    try:
        # Convert input to model format
        input_dict = data.dict()
        logger.info(f"📦 Received Input: {input_dict}")

        # Example: encode features (this part should match your preprocessing)
        # For simplicity, assume the model pipeline handles encoding internally.
        features = np.array([list(input_dict.values())]).reshape(1, -1)

        # Predict
        probability = model.predict_proba(features)[0][1]
        donation_decision = "Likely to Donate" if probability > 0.5 else "Unlikely to Donate"

        # Confidence level
        confidence = "High" if probability >= 0.8 else "Moderate" if probability >= 0.6 else "Low"

        logger.info(f"✅ Prediction: {donation_decision} ({probability:.2f})")

        # Return response
        return PredictionResponse(
            donation_probability=round(probability * 100, 2),
            donation_decision=donation_decision,
            confidence_level=confidence
        )

    except Exception as e:
        logger.error("Prediction Error: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# =========================
# 🔟 Run API
# =========================
# To run locally: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)