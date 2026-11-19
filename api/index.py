
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS AND TRAINED MODEL
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "kmeans_model.joblib"
PREPROCESSOR_PATH = ROOT_DIR / "ml" / "preprocessor.joblib"

FRONTEND_DIR = ROOT_DIR / "frontend"

kmeans = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# 2. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="MicroSegment API",
    description="Unsupervised clustering of Indian Kiva loan profiles",
    version="1.0.0"
)


# --------------------------------------------------
# 3. CORS FOR LOCAL DEVELOPMENT
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# 4. ALLOWED CATEGORIES
# --------------------------------------------------

VALID_SECTORS = {
    "Agriculture",
    "Arts",
    "Clothing",
    "Construction",
    "Food",
    "Health",
    "Manufacturing",
    "Other",
    "Retail",
    "Services",
    "Transportation"
}

VALID_REPAYMENT_INTERVALS = {
    "bullet",
    "irregular",
    "monthly"
}


# --------------------------------------------------
# 5. CLUSTER PROFILES
# --------------------------------------------------

CLUSTER_PROFILES = {
    0: {
        "profile": "Smaller-loan, agriculture-focused",
        "average_loan_inr": 296.85,
        "average_term_months": 43.00,
        "dominant_sectors": [
            "Agriculture",
            "Food",
            "Services"
        ],
        "repayment_pattern": "100% bullet in the training cluster"
    },

    1: {
        "profile": "Shorter-term, diversified-sector",
        "average_loan_inr": 410.06,
        "average_term_months": 15.70,
        "dominant_sectors": [
            "Food",
            "Agriculture",
            "Services"
        ],
        "repayment_pattern": (
            "Approximately 99.85% monthly in the training cluster"
        )
    },

    2: {
        "profile": "Larger-loan, agriculture and food-focused",
        "average_loan_inr": 2271.41,
        "average_term_months": 42.76,
        "dominant_sectors": [
            "Agriculture",
            "Food",
            "Retail"
        ],
        "repayment_pattern": (
            "Approximately 99.03% bullet in the training cluster"
        )
    }
}


# --------------------------------------------------
# 6. INPUT MODEL
# --------------------------------------------------

class LoanDetails(BaseModel):
    loan_amount: float = Field(gt=0)
    term_in_months: float = Field(gt=0)
    sector: str = Field(min_length=1)
    repayment_interval: str = Field(min_length=1)


# --------------------------------------------------
# 7. API HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "message": "Welcome to MicroSegment API",
        "status": "running",
        "model": "K-Means",
        "clusters": 3
    }


# --------------------------------------------------
# 8. CLUSTER PREDICTION
# --------------------------------------------------

@app.post("/api/predict")
@app.post("/predict")
def predict_cluster(details: LoanDetails):

    repayment = details.repayment_interval.strip().lower()

    if repayment not in VALID_REPAYMENT_INTERVALS:
        raise HTTPException(
            status_code=400,
            detail="Repayment interval must be monthly, bullet, or irregular."
        )

    sector = details.sector.strip().title()

    if sector not in VALID_SECTORS:
        sector = "Other"

    input_data = pd.DataFrame([{
        "loan_amount_log": np.log1p(details.loan_amount),
        "term_in_months": details.term_in_months,
        "sector_clean": sector,
        "repayment_interval": repayment
    }])

    processed_array = preprocessor.transform(input_data)

    processed_data = pd.DataFrame(
        processed_array,
        columns=preprocessor.get_feature_names_out()
    )

    cluster = int(kmeans.predict(processed_data)[0])

    profile = CLUSTER_PROFILES[cluster]

    return {
        "success": True,
        "cluster": cluster,
        "cluster_profile": profile["profile"],

        "input": {
            "loan_amount_inr": details.loan_amount,
            "term_in_months": details.term_in_months,
            "sector": sector,
            "repayment_interval": repayment
        },

        "training_cluster_profile": profile,

        "notice": (
            "This is an unsupervised cluster assignment based on "
            "historical loan patterns. It is not a credit score, "
            "repayment prediction, or credit approval."
        )
    }


# --------------------------------------------------
# 9. SERVE FRONTEND ON VERCEL
# --------------------------------------------------

app.frontend(
    "/",
    directory=str(FRONTEND_DIR)
)
