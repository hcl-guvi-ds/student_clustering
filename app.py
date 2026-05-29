import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Intelligence Dashboard",
    layout="wide"
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("models/student_cluster_predictor.pkl")


# ---------------------------------------------------
# CLUSTER NAME MAPPING
# ---------------------------------------------------

cluster_names = {
    0: "High Performer",
    1: "At Risk",
    2: "Consistent Learner",
    3: "Early Dropper"
}


# ---------------------------------------------------
# HEALTH SCORE FUNCTION
# ---------------------------------------------------

def calculate_health_score(
    attendance,
    task_completion,
    avg_score,
    mentor_rating
):

    score = (
        attendance * 0.35 +
        task_completion * 0.35 +
        avg_score * 0.20 +
        mentor_rating * 20 * 0.10
    )

    return round(score, 2)


# ---------------------------------------------------
# RISK CATEGORY
# ---------------------------------------------------

def get_risk_category(score):

    if score >= 85:
        return "Low Risk"

    elif score >= 65:
        return "Moderate Risk"

    else:
        return "High Risk"


# ---------------------------------------------------
# INTERVENTION ENGINE
# ---------------------------------------------------

def intervention_recommendation(cluster_name):

    recommendations = {

        "High Performer":
            "Assign advanced projects and placement preparation.",

        "At Risk":
            "Immediate mentor intervention and attendance follow-up required.",

        "Passive Learner":
            "Encourage assignment participation and regular engagement.",

        "Consistent Learner":
            "Provide structured mentoring and periodic monitoring."
    }

    return recommendations.get(cluster_name, "No recommendation available")


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("Student Intelligence Dashboard")

st.markdown(
    "Predict future student behavior clusters using early signals"
)


# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.header("Student Inputs")


# Attendance
attendance_percent = st.sidebar.slider(
    "Attendance %",
    0,
    100,
    70
)


# Attending Hours
avg_attending_hrs = st.sidebar.slider(
    "Average Attending Hours",
    0.0,
    5.0,
    2.0
)


# Task Completion
task_completion_percent = st.sidebar.slider(
    "Task Completion %",
    0,
    100,
    50
)


# Task Score
avg_task_score = st.sidebar.slider(
    "Average Task Score",
    0,
    100,
    60
)


# Mini Project Completion
m_project_completion_percent = st.sidebar.slider(
    "Mini Project Completion %",
    0,
    100,
    50
)


# Mini Project Score
avg_m_project_score = st.sidebar.slider(
    "Mini Project Score",
    0,
    100,
    60
)


# Final Project Completion
f_project_completion_percent = st.sidebar.slider(
    "Final Project Completion %",
    0,
    100,
    50
)


# Final Project Score
avg_f_project_score = st.sidebar.slider(
    "Final Project Score",
    0,
    100,
    60
)


# CCT Percentage
cct_percentage = st.sidebar.slider(
    "CCT Percentage",
    0,
    100,
    60
)


# Queries
coordination_query = st.sidebar.number_input(
    "Coordination Queries",
    0,
    100,
    0
)

zenclassdoubt_queries = st.sidebar.number_input(
    "Doubt Queries",
    0,
    100,
    0
)


# Ratings
avg_mentor_rating = st.sidebar.slider(
    "Mentor Rating",
    0.0,
    5.0,
    4.0
)

avg_session_rating = st.sidebar.slider(
    "Session Rating",
    0.0,
    5.0,
    4.0
)


# Work Experience
work_experience_in_years = st.sidebar.slider(
    "Work Experience (Years)",
    0,
    15,
    1
)


# Gender
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)


# UG Marks
ug_marks_in_percentage = st.sidebar.slider(
    "UG Marks %",
    0,
    100,
    70
)


# Degree
highest_degree = st.sidebar.selectbox(
    "Highest Degree",
    [
        "BE/B.Tech",
        "B.Sc",
        "M.Tech",
        "MBA",
        "MCA",
        "Other"
    ]
)


# Department
department = st.sidebar.selectbox(
    "Department",
    [
        "Computer Science",
        "Information Technology",
        "Electronics",
        "Mechanical",
        "Civil",
        "Other"
    ]
)


# ---------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------

input_df = pd.DataFrame({

    "attendance_percent": [attendance_percent],

    "avg_attending_hrs": [avg_attending_hrs],

    "task_completion_percent": [task_completion_percent],

    "avg_task_score": [avg_task_score],

    "m_project_completion_percent": [m_project_completion_percent],

    "avg_m_project_score": [avg_m_project_score],

    "f_project_completion_percent": [f_project_completion_percent],

    "avg_f_project_score": [avg_f_project_score],

    "cct_percentage": [cct_percentage],

    "coordination_query": [coordination_query],

    "zenclassdoubt_queries": [zenclassdoubt_queries],

    "avg_mentor_rating": [avg_mentor_rating],

    "avg_session_rating": [avg_session_rating],

    "work_experience_in_years": [work_experience_in_years],

    "ug_marks_in_percentage": [ug_marks_in_percentage],

    "gender": [gender],

    "what_is_your_highest_degree?": [highest_degree],

    "which_department_have_you_studied?": [department]
})


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict Student Cluster"):

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = round(np.max(probabilities) * 100, 2)

    cluster_name = cluster_names[prediction]


    # Health Score
    health_score = calculate_health_score(
        attendance_percent,
        task_completion_percent,
        avg_task_score,
        avg_mentor_rating
    )


    risk_category = get_risk_category(
        health_score
    )


    recommendation = intervention_recommendation(
        cluster_name
    )


    # ---------------------------------------------------
    # RESULTS
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Cluster",
            cluster_name
        )


    with col2:

        st.metric(
            "Prediction Confidence",
            f"{confidence}%"
        )


    with col3:

        st.metric(
            "Health Score",
            health_score
        )


    st.subheader("Risk Assessment")

    st.write(risk_category)


    st.subheader("Recommended Intervention")

    st.info(recommendation)


    # ---------------------------------------------------
    # CLUSTER PROBABILITY CHART
    # ---------------------------------------------------

    probability_df = pd.DataFrame({

        "Cluster": [
            cluster_names[i]
            for i in range(len(probabilities))
        ],

        "Probability": probabilities
    })


    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=probability_df["Cluster"],
            y=probability_df["Probability"]
        )
    )


    fig.update_layout(
        title="Cluster Probabilities",
        xaxis_title="Cluster",
        yaxis_title="Probability"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ---------------------------------------------------
    # INPUT SUMMARY
    # ---------------------------------------------------

    st.subheader("Input Summary")

    st.dataframe(input_df)