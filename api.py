from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Employee Attrition API"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("HR_Employee_Attrition.csv")

# Convert Target

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

# Remove useless columns

X = X.drop([
    "EmployeeCount",
    "StandardHours",
    "Employee_ID"
], axis=1, errors="ignore")

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

le = LabelEncoder()

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = le.fit_transform(X[col])

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = XGBClassifier(
    n_estimators=70,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# =========================
# REQUEST MODEL
# =========================

class EmployeeData(BaseModel):

    Age: int

    MonthlyIncome: int

    DistanceFromHome: int

    YearsAtCompany: int

    JobSatisfaction: int

    WorkLifeBalance: int

    EnvironmentSatisfaction: int

    OverTime: str

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():

    return {
        "message": "Employee Attrition API is running"
    }

# =========================
# PREDICTION ROUTE
# =========================

@app.post("/predict")
def predict(data: EmployeeData):

    # Create Empty Sample

    sample = pd.DataFrame(columns=X.columns)

    sample.loc[0] = 0

    # Input Values

    sample["Age"] = data.Age

    sample["MonthlyIncome"] = data.MonthlyIncome

    sample["DistanceFromHome"] = data.DistanceFromHome

    sample["YearsAtCompany"] = data.YearsAtCompany

    sample["JobSatisfaction"] = data.JobSatisfaction

    sample["WorkLifeBalance"] = data.WorkLifeBalance

    sample["EnvironmentSatisfaction"] = data.EnvironmentSatisfaction

    # OverTime

    overtime_value = 1 if data.OverTime == "Yes" else 0

    if "OverTime" in sample.columns:

        sample["OverTime"] = overtime_value

    # Prediction Probability

    probability = model.predict_proba(sample)[0][1]

    risk_percent = round(probability * 100, 2)

    # Risk Level

    if probability < 0.3:

        risk_level = "Low Risk"

    elif probability < 0.6:

        risk_level = "Medium Risk"

    else:

        risk_level = "High Risk"

    # Return Result

    return {

        "risk_score": risk_percent,

        "risk_level": risk_level
    }