import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI(title="Water Quality Classifier")
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

with open(str(BASE_DIR / "model.pkl"), "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    imputer = saved["imputer"]

# WHO safe drinking water ranges
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
    "ph":              "pH Level",
    "hardness":        "Hardness (mg/L)",
    "solids":          "Total Dissolved Solids (mg/L)",
    "chloramines":     "Chloramines (mg/L)",
    "sulfate":         "Sulfate (mg/L)",
    "conductivity":    "Conductivity (uS/cm)",
    "organic_carbon":  "Organic Carbon (mg/L)",
    "trihalomethanes": "Trihalomethanes (ug/L)",
    "turbidity":       "Turbidity (NTU)",
}

def get_risky_factors(inputs):
    risky = []
    for key, value in inputs.items():
        low, high = SAFE_RANGES[key]
        if value < low or value > high:
            risky.append({
                "name": FEATURE_LABELS[key],
                "value": value,
                "safe_range": f"{low} - {high}"
            })
    return risky

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:         Request,
    ph:              float = Form(...),
    hardness:        float = Form(...),
    solids:          float = Form(...),
    chloramines:     float = Form(...),
    sulfate:         float = Form(...),
    conductivity:    float = Form(...),
    organic_carbon:  float = Form(...),
    trihalomethanes: float = Form(...),
    turbidity:       float = Form(...),
):
    inputs = {
        "ph": ph, "hardness": hardness, "solids": solids,
        "chloramines": chloramines, "sulfate": sulfate,
        "conductivity": conductivity, "organic_carbon": organic_carbon,
        "trihalomethanes": trihalomethanes, "turbidity": turbidity,
    }

    # Step 1 — pH is scientifically impossible beyond 0-14
    # Instantly mark as UNSAFE without asking the model
    if ph < 0 or ph > 14:
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "safe": False,
                "confidence": 100.0,
                "risky": [{
                    "name": "pH Level",
                    "value": ph,
                    "safe_range": "0 - 14 (physically impossible beyond this)"
                }],
                "inputs": inputs,
                "labels": FEATURE_LABELS,
                "impossible": True,
            }
        )

    # Step 2 — ML model decides for all other cases
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    features = imputer.transform(features)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(max(probabilities) * 100, 2)

    # Step 3 — Show which individual values are outside WHO safe ranges
    risky = get_risky_factors(inputs)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "safe": bool(prediction == 1),
            "confidence": confidence,
            "risky": risky,
            "inputs": inputs,
            "labels": FEATURE_LABELS,
            "impossible": False,
        }
    )