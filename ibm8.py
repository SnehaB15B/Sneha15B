import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# For reproducibility
np.random.seed(42)

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=column_names)

# Inspect the dataset
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Define a function to get user input and make predictions
def get_user_input():
    print("Enter the following details to check diabetes risk:")
    try:
        pregnancies = float(input("Pregnancies: "))
        glucose = float(input("Glucose: "))
        blood_pressure = float(input("Blood Pressure: "))
        skin_thickness = float(input("Skin Thickness: "))
        insulin = float(input("Insulin: "))
        bmi = float(input("BMI: "))
        diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
        age = float(input("Age: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    input_data_scaled = scaler.transform(input_data)  # Scale the input data
    prediction = model.predict(input_data_scaled)
   
    print("Prediction (0: Non-Diabetic, 1: Diabetic):", prediction[0])

# Loop to allow multiple user inputs
while True:
    get_user_input()
    cont = input("Do you want to check another person? (yes/no): ").strip().lower()
    if cont != 'yes':
        break