import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("No. of Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("No. of Parents/Children Aboard", 0, 5, 0)
fare = st.slider("Fare Paid", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Convert inputs
sex = 0 if sex == 'male' else 1
embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked]

# Load and train model (basic, fast)
@st.cache_resource
def load_model():

    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = load_model()

# Predict
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"🎉 Survived! (Confidence: {probability*100:.2f}%)")
    else:
        st.error(f"💀 Did Not Survive. (Confidence: {probability*100:.2f}%)")



st.sidebar.title("ℹ️ About")
st.sidebar.info("This app predicts Titanic passenger survival using a Random Forest model trained on real data.")

