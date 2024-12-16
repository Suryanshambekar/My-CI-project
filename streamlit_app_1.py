import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    # Load and preprocess the dataset
    df = pd.read_csv("nasa.csv")  # Ensure 'nasa.csv' is in the working directory
    df = df.drop(['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
                  'Epoch Date Close Approach', 'Orbit Determination Date',
                  'Orbiting Body', 'Equinox'], axis=1)  # Drop irrelevant columns
    df['Hazardous'] = df['Hazardous'].astype(int)  # Ensure target is integer
    return df

# Load and preprocess data
st.title("Asteroid Hazard Prediction")
df = load_data()

# Feature and target separation
X = df.drop(['Hazardous'], axis=1)
y = df['Hazardous']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Decision Tree model
dec_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=0)
dec_tree.fit(X_train, y_train)

# Predictions and accuracy
y_pred = dec_tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {acc * 100:.2f}%")

# Visualize the Decision Tree
st.write("### Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(dec_tree, feature_names=X.columns, class_names=['Non-Hazardous', 'Hazardous'],
          filled=True, rounded=True, ax=ax)
st.pyplot(fig)

# User input for predictions
st.sidebar.header("User Input Features")
input_data = {col: st.sidebar.number_input(f"{col}", value=0.0) for col in X.columns}
input_df = pd.DataFrame([input_data])

# Prediction on user input
if st.sidebar.button("Predict"):
    prediction = dec_tree.predict(input_df)
    st.write(f"### Prediction: {'Hazardous' if prediction[0] == 1 else 'Non-Hazardous'}")
