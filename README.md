\# Water Quality Classifier



A Machine Learning web application that predicts whether water is safe to drink based on chemical measurements.



\## Real World Use

Users enter 9 chemical measurements from a water sample and the app instantly predicts if the water is Safe or Unsafe to drink, shows confidence percentage, and highlights which factors are dangerous.



\## Tech Stack

\- Machine Learning: Random Forest + XGBoost Ensemble

\- Web Framework: FastAPI

\- Frontend: HTML + CSS + Jinja2

\- Dataset: Water Potability Dataset 3276 samples

\- Accuracy: 70.25%



\## How to Run

1\. Install dependencies: pip install -r requirements.txt

2\. Train the model: python model/train\_model.py

3\. Start the server: uvicorn app.main:app --reload

4\. Open browser at: http://127.0.0.1:8000



\## Developer

GitHub: baijucode

