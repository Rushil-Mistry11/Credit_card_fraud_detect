# Real-Time Credit Card Fraud Risk Scoring & Analytics System

SMOTE + SVC / Decision Tree / Naive Bayes / MLP  ->  FastAPI + SQLite  ->  Streamlit dashboard.

## Structure
```
fraud-risk-system/
├── data/creditcard.csv        # download from Kaggle (mlg-ulb/creditcardfraud)
├── ml/        config.py  preprocess.py  train.py  evaluate.py  make_demo_data.py
├── backend/   main.py  database.py  models.py  schemas.py
├── frontend/  app.py
├── models/                    # created by training (*.pkl + metrics.json)
├── reports/figures/           # created by ml.evaluate (for your report)
└── requirements.txt
```

## Setup (VS Code terminal, from the project root)
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> pick the `venv` one.
Put `creditcard.csv` inside `data/`.

## Run order (always from the project root)
```bash
python -m ml.train               # 1. SMOTE, train 4 models, save models/ + metrics.json
python -m ml.evaluate            # 2. save report figures to reports/figures/
uvicorn backend.main:app --reload            # 3. API  (docs: http://127.0.0.1:8000/docs)
streamlit run frontend/app.py                # 4. dashboard (second terminal)
```
Use two terminals for steps 3 and 4.

### Try the pipeline before downloading the dataset
```bash
python -m ml.make_demo_data                  # creates data/creditcard_demo.csv (synthetic)
# PowerShell:  $env:FRAUD_DATA="data/creditcard_demo.csv"
# macOS/Linux: export FRAUD_DATA=data/creditcard_demo.csv
python -m ml.train --svc-samples 6000
```
Unset `FRAUD_DATA` and retrain on the real file for your final results.

## API example
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"algorithm":"mlp","features":{"Time":0,"V1":0,...,"V28":0,"Amount":50}}'
```
Endpoints: `POST /predict`, `GET /logs?limit=&risk=`, `GET /logs/summary`, `GET /metrics`, `GET /health`.

## Notes for the report
- Split -> scale -> SMOTE order avoids leakage; SMOTE touches training data only.
- Headline metrics are Precision, Recall, F1, PR-AUC, ROC-AUC (not accuracy: ~99.8% is achievable by predicting "legit").
- `DecisionTreeClassifier(criterion="entropy")` uses information gain like ID3/C4.5, but scikit-learn implements CART (binary splits, no gain ratio). State this honestly.
- SVC is trained on a 40k subsample of the SMOTE'd set (`--svc-samples`) because kernel SVMs scale poorly; state this tradeoff.
- Bias-variance: train-vs-test F1 gap per model (`bias_variance.png`) and the decision-tree depth sweep (`decision_tree_depth_curve.png`).
- SMOTE shifts class priors, so predicted probabilities are not calibrated; treat the gauge as a relative risk score. Thresholds are in `ml/config.py`.
- Expected training time on the full dataset: roughly 10-25 min total (SVC prediction with probabilities and the MLP dominate).
