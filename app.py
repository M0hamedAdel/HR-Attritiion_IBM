import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("HR_Employee_Attrition.csv")

df = load_data()

# =========================
# PREPROCESSING
# =========================

# Convert target column

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Labels for charts

df["Attrition_Label"] = df["Attrition"].map({
    1: "Left",
    0: "Stayed"
})

# =========================
# FEATURES & TARGET
# =========================

X = df.drop(["Attrition", "Attrition_Label"], axis=1)

y = df["Attrition"]

# Remove unnecessary columns

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
# TRAIN XGBOOST MODEL
# =========================

@st.cache_resource
def train_model():

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

    return model

model = train_model()

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Filters")

selected_department = st.sidebar.selectbox(
    "Select Department",
    df["Department"].unique()
)

# Filtered Data

dff = df[df["Department"] == selected_department]

# =========================
# TITLE
# =========================

st.title("Employee Attrition Dashboard")

# =========================
# KPI METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Employees", len(dff))

with col2:
    st.metric("Attrition Count", int(dff["Attrition"].sum()))

with col3:
    attrition_rate = round(dff["Attrition"].mean() * 100, 2)
    st.metric("Attrition Rate", f"{attrition_rate}%")

with col4:
    avg_income = round(dff["MonthlyIncome"].mean(), 2)
    st.metric("Avg Income", avg_income)

# =========================
# CHARTS
# =========================

chart1, chart2 = st.columns(2)

with chart1:

    fig_attrition = px.histogram(
        dff,
        x="Attrition_Label",
        color="Attrition_Label",
        title="Attrition Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig_attrition, use_container_width=True)

with chart2:

    fig_overtime = px.histogram(
        dff,
        x="OverTime",
        color="Attrition_Label",
        barmode="group",
        title="Attrition vs OverTime",
        template="plotly_dark"
    )

    st.plotly_chart(fig_overtime, use_container_width=True)

chart3, chart4 = st.columns(2)

with chart3:

    fig_income = px.box(
        dff,
        x="Attrition_Label",
        y="MonthlyIncome",
        title="Monthly Income vs Attrition",
        template="plotly_dark"
    )

    st.plotly_chart(fig_income, use_container_width=True)

with chart4:

    fig_importance = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Important Features",
        template="plotly_dark"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

# =========================
# PREDICTION SECTION
# =========================

st.header("Employee Attrition Prediction")

# INPUTS

age = st.slider(
    "Age",
    18,
    60,
    30
)

department = st.selectbox(
    "Department",
    [
        "Human Resources",
        "Research & Development",
        "Sales"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=50000,
    value=5000
)

distance = st.slider(
    "Distance From Home",
    1,
    30,
    5
)

years_company = st.slider(
    "Years At Company",
    0,
    40,
    5
)

job_satisfaction = st.slider(
    "Job Satisfaction",
    1,
    4,
    2
)

work_life = st.slider(
    "Work Life Balance",
    1,
    4,
    2
)

environment = st.slider(
    "Environment Satisfaction",
    1,
    4,
    2
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

# =========================
# ENCODING
# =========================

overtime_value = 1 if overtime == "Yes" else 0

department_mapping = {

    "Human Resources": 0,

    "Research & Development": 1,

    "Sales": 2
}

department_value = department_mapping[department]

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    # Create Empty Sample

    sample = pd.DataFrame(columns=X.columns)

    sample.loc[0] = 0

    # Input Values

    sample["Age"] = age

    sample["MonthlyIncome"] = monthly_income

    sample["DistanceFromHome"] = distance

    sample["YearsAtCompany"] = years_company

    sample["JobSatisfaction"] = job_satisfaction

    sample["WorkLifeBalance"] = work_life

    sample["EnvironmentSatisfaction"] = environment

    # Department

    if "Department" in sample.columns:

        sample["Department"] = department_value

    # OverTime

    if "OverTime" in sample.columns:

        sample["OverTime"] = overtime_value

    # Prediction Probability

    probability = model.predict_proba(sample)[0][1]

    # Convert to Percentage

    risk_percent = round(probability * 100, 2)

    # Risk Levels

    if probability < 0.3:

        st.success(
            f"Low Attrition Risk\n\n"
            f"Risk Score: {risk_percent}%"
        )

    elif probability < 0.6:

        st.warning(
            f"Medium Attrition Risk\n\n"
            f"Risk Score: {risk_percent}%"
        )

    else:

        st.error(
            f"High Attrition Risk\n\n"
            f"Risk Score: {risk_percent}%"
        )