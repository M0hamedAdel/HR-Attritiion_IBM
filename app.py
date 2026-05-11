```python
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
```
