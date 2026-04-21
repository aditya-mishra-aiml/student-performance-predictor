import joblib

model = joblib.load("model.pkl")

print("Enter student details:\n")

study_hours = float(input("Study hours (0-12): "))
sleep_hours = float(input("Sleep hours (0-12): "))
attendance = float(input("Attendance (%) (0-100): "))
previous_marks = float(input("Previous marks (0-100): "))
assignments_completed = float(input("Assignments completed (0-10): "))
distractions = float(input("Distractions (1-10 scale): "))

# Basic validation
if not (0 <= study_hours <= 12):
    print("Invalid study hours")
    exit()

if not (0 <= attendance <= 100):
    print("Invalid attendance")
    exit()

# Prediction
data = [[
    study_hours,
    sleep_hours,
    attendance,
    previous_marks,
    assignments_completed,
    distractions
]]

prediction = model.predict(data)

print(f"\nPredicted Final Marks: {round(prediction[0], 2)} / 100")