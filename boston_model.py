import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your dataset
data = pd.read_csv("boston.csv")

st.title("Boston Housing 🏠 Price Prediction ")

# Create a sidebar for user inputs
st.sidebar.header("Parameters")

# Remove feature selection and directly set 'age' as the selected feature
selected_features = ['age']

# Train-test split ratio
test_size = st.sidebar.slider("Test Size", 0.1, 0.8, 0.2)

# Skewed feature transformation (excluding 'age')
skewed_features = ['crim', 'zn', 'indus', 'nox', 'dis', 'rad', 'ptratio', 'black', 'lstat']
data[skewed_features] = np.log1p(data[skewed_features])

# For single-value prediction, pick only 'medv' as the target
y = data["medv"]

# Ensure target variable is non-negative
if (y < 0).any():
    st.error("Target variable contains negative values. Please check the dataset.")
    st.stop()

# Split the data into features (X) and target (y)
X = data[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
user_input_age_str = st.text_input("Enter age:", value="10", key="age_input_field")
user_input_age = float(user_input_age_str)

# Convert the text input to float before creating the array
input_data = pd.DataFrame([[user_input_age]], columns=selected_features)
input_data = scaler.transform(input_data)

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(input_data)

# Ensure prediction is non-negative
if prediction < 0:
    st.warning("The prediction is negative, which is not realistic for housing prices. Please check the model and data preprocessing steps.")

st.write("Amount: ", prediction*1000)
