import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
<style>
div.stButton > button {
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("model.pkl")

# Title
st.title("🎓 Student Performance Predictor")
st.caption("Predict student performance using ML based on study habitrs and academic factors")

# Section
st.subheader("📘 Student Inputs")

# Row 1
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Study Hours", 0, 12, 5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    assignments = st.slider("Assignments Completed (0-10) (%)", 0, 100, 70)

with col2:
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    previous_marks = st.slider("Previous Marks", 0, 100, 65)
    distractions = st.slider("Distraction Hours", 0, 10, 3)

# Predict Button
if st.button("Predict"):

    input_data = np.array([[
        study_hours,
        sleep_hours,
        attendance,
        previous_marks,
        assignments,
        distractions
    ]])
    
    prediction = model.predict(input_data)[0]

    st.write('### Input Summary')
    st.write(f"""
    - Study Hours: {study_hours}
    - Sleep Hours: {sleep_hours}
    - Attendance: {attendance}%
    - Previous Marks: {previous_marks}
    - Assignments: {assignments}
    - Distractions: {distractions}
    """)

# Result
    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Final Marks: {prediction:.2f}")
    
    if prediction > 80:
        st.success("Excellent Performance Expected")
    elif prediction > 50:
        st.warning("Average Performance")
    else:
        st.error("Needs Improvement")

# Progress bar
    st.progress(min(int(prediction), 100))


# Graph 1: Feature Values
    st.subheader("📊 Input Feature Overview")

    features = [
        "Study Hours", "Sleep Hours", "Attendance",
        "Previous Marks", "Assignments", "Distractions"
    ]

    values = [
        study_hours, sleep_hours, attendance,
        previous_marks, assignments, distractions
    ]

    fig, ax = plt.subplots()
    ax.bar(features, values)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# Graph 2: Prediction Comparison
    st.subheader("📈 Previous vs Predicted")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Previous Marks", "Predicted Marks"],
            [previous_marks, prediction])
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.info("Model Used: Best Performing Model")