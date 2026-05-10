# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

# Label for charts
df["Attrition_Label"] = df["Attrition"].map({
    1: "Left",
    0: "Stayed"
})

# =========================
# MACHINE LEARNING
# =========================

X = df.drop(["Attrition", "Attrition_Label"], axis=1)
y = df["Attrition"]

# Remove unnecessary columns
X = X.drop([
    "EmployeeCount",
    "StandardHours",
    "Employee_ID"
], axis=1, errors="ignore")

# Encode categorical columns
le = LabelEncoder()

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Feature Importance
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
# KPI METRICS
# =========================

st.title("Employee Attrition Dashboard")

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
        title="Attrition vs Monthly Income",
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