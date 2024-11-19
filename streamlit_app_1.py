
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("nasa.csv")  # Ensure 'nasa.csv' is in the same directory or GitHub repo
    df = df.drop(['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
                  'Epoch Date Close Approach', 'Orbit Determination Date',
                  'Orbiting Body', 'Equinox'], axis=1)
    hazardous_labels = pd.get_dummies(df['Hazardous'])
    df = pd.concat([df, hazardous_labels], axis=1).drop(['Hazardous'], axis=1)
    df = df.drop(['Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
                  'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 
                  'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
                  'Relative Velocity km per hr', 'Miles per hour', 
                  'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)'], axis=1)
    return df

# Load and preprocess data
st.title("Asteroid Hazard Prediction")
df = load_data()

x = df.drop([True], axis=1)
y = df[True].astype(int)

# Ensure column names are strings
x.columns = x.columns.astype(str)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train Decision Tree model
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
preds = dec_tree.predict(x_test)

# Metrics
acc = accuracy_score(y_test, preds)

st.write(f"Model Accuracy: {acc * 100:.2f}%")

# User input
st.sidebar.header("User Input Features")
input_data = {col: st.sidebar.number_input(f"{col}", value=0.0) for col in x.columns}
input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict"):
    prediction = dec_tree.predict(input_df)
    st.write("Prediction (1 = Hazardous, 0 = Non-Hazardous):", prediction[0])
