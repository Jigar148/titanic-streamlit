import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Title
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
gender = st.selectbox("Gender", ['male', 'female'])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("No. of Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("No. of Parents/Children Aboard", 0, 5, 0)
fare = st.slider("Fare Paid", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Convert inputs
gender_converted = 0 if gender == 'male' else 1
embarked_converted = {'S': 0, 'C': 1, 'Q': 2}[embarked]

# Load and train model
@st.cache_resource
def load_data_and_model():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    df['Gender'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = df[['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']
    model = RandomForestClassifier()
    model.fit(X, y)
    return df, model, X.columns.tolist()

df, model, feature_names = load_data_and_model()

# Predict
if st.button("🎯 Predict Survival"):
    input_data = np.array([[pclass, gender_converted, age, sibsp, parch, fare, embarked_converted]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"🎉 Survived! (Confidence: {probability * 100:.2f}%)")
    else:
        st.error(f"💀 Did Not Survive. (Confidence: {probability * 100:.2f}%)")

# 📊 Feature Importance
st.subheader("📊 Feature Importance (Random Forest)")
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(feature_names, importances, color='skyblue')
ax.set_xlabel("Importance")
ax.set_title("Which features impact survival?")
st.pyplot(fig)

# 🧬 Clustering Section
st.subheader("🧬 KMeans Clustering of Passengers (Visualization)")
df_clean = df[['Age', 'Fare']].dropna()
kmeans = KMeans(n_clusters=3, n_init=10)
clusters = kmeans.fit_predict(df_clean)
df_clean['Cluster'] = clusters

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df_clean['Age'], df_clean['Fare'], c=clusters, cmap='viridis')
ax2.set_xlabel("Age")
ax2.set_ylabel("Fare")
ax2.set_title("KMeans Clustering: Age vs Fare")
st.pyplot(fig2)

# 🔍 ML Insights
with st.expander("🧠 Machine Learning Details"):
    st.markdown("""
    - **Model Used:** Random Forest Classifier  
    - **Target Variable:** `Survived` (0 = No, 1 = Yes)  
    - **Features Used:**  
      - Pclass (Ticket Class)  
      - Gender  
      - Age  
      - SibSp (Siblings/Spouses Aboard)  
      - Parch (Parents/Children Aboard)  
      - Fare  
      - Embarked (Port)
    - **Preprocessing:**  
      - Missing value imputation  
      - Label encoding for categorical features  
    - **Extras:**  
      - Feature Importance Graph  
      - KMeans Clustering (Unsupervised)
    """)

# Sidebar Info
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This app predicts whether a Titanic passenger would survive based on input features,  
trained using a **Random Forest Classifier** on the original Titanic dataset.

🔍 ML Features:  
- Classification  
- Clustering  
- Feature Importance Visualization

Created with ❤️ using **Streamlit**
""")
