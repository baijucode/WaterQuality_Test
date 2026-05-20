# app/main.py
# FastAPI server — serves the form, receives user input, returns prediction.

import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ── Initialize app ────────────────────────────────────────────────────────────
app = FastAPI(title="Water Quality Classifier")
templates = Jinja2Templates(directory="templates")

# ── Load the saved model and imputer at startup ───────────────────────────────
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    imputer = saved["imputer"]

# Safe ranges for each feature (used to flag risky values)
SAFE_RANGES = {
    "ph":              (6.5, 8.5),
    "hardness":        (0,   300),
    "solids":          (0,   500),
    "chloramines":     (0,   4),
    "sulfate":         (0,   250),
    "conductivity":    (0,   400),
    "organic_carbon":  (0,   2),
    "trihalomethanes": (0,   80),
    "turbidity":       (0,   4),
}

FEATURE_LABELS = {
    "ph":              "pH level",
    "hardness":        "Hardness (mg/L)",
    "solids":          "Total Dissolved Solids (mg/L)",
    "chloramines":     "Chloramines (mg/L)",
    "sulfate":         "Sulfate (mg/L)",
    "conductivity":    "Conductivity (µS/cm)",
    "organic_carbon":  "Organic Carbon (mg/L)",
    "trihalomethanes": "Trihalomethanes (µg/L)",
    "turbidity":       "Turbidity (NTU)",
}

def get_risky_factors(inputs: dict) -> list:
    """Return list of features that are outside the safe range."""
    risky = []
    for key, value in inputs.items():
        low, high = SAFE_RANGES[key]
        if value < low or value > high:
            risky.append({
                "name": FEATURE_LABELS[key],
                "value": value,
                "safe_range": f"{low} – {high}"
            })
    return risky

# ── Route 1: Home page (show form) ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ── Route 2: Predict (receive form, return result) ────────────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:          Request,
    ph:               float = Form(...),
    hardness:         float = Form(...),
    solids:           float = Form(...),
    chloramines:      float = Form(...),
    sulfate:          float = Form(...),
    conductivity:     float = Form(...),
    organic_carbon:   float = Form(...),
    trihalomethanes:  float = Form(...),
    turbidity:        float = Form(...),
):
    # Package inputs into a dict for easy processing
    inputs = {
        "ph": ph, "hardness": hardness, "solids": solids,
        "chloramines": chloramines, "sulfate": sulfate,
        "conductivity": conductivity, "organic_carbon": organic_carbon,
        "trihalomethanes": trihalomethanes, "turbidity": turbidity,
    }

    # Prepare input array — shape (1, 9) for the model
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])

    # Apply same imputer (handles any unexpected edge cases)
    features = imputer.transform(features)

    # Get prediction and confidence
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(max(probabilities) * 100, 2)

    # Get risky factors
    risky = get_risky_factors(inputs)

    return templates.TemplateResponse(
        "result.html",
        {
            "request":    request,
            "safe":       bool(prediction == 1),
            "confidence": confidence,
            "risky":      risky,
            "inputs":     inputs,
            "labels":     FEATURE_LABELS,
        }
    )