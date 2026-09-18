import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Disease Risk Tool", layout="wide")



DATA_PATH = "Heart_disease_cleveland_new.csv"          
MODEL_PATH = "Heart_Disease_RF_model.joblib"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model(df):
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
    return model

df = load_data()
model = load_model(df)
feature_cols = [c for c in df.columns if c != 'target']



st.sidebar.header("Patient Input")

def user_input():
    age = st.sidebar.slider("Age", 20, 90, 50)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80, 220, 130)
    chol = st.sidebar.slider("Cholesterol (chol)", 100, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of ST Segment (slope)", options=[0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels Colored (ca)", options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([data])

input_df = user_input()


tab1, tab2 = st.tabs(["🩺 Prediction", "📊 EDA Dashboard"])

with tab1:
    st.title("Heart Disease Risk Prediction")
    st.write("Enter patient details in the sidebar, then view the predicted risk below.")

    st.subheader("Input Summary")
    st.dataframe(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_df[feature_cols])[0]
        proba = model.predict_proba(input_df[feature_cols])[0]

        if prediction == 1:
            st.error(f"⚠️ Higher risk of heart disease  (confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ Lower risk of heart disease  (confidence: {proba[0]*100:.1f}%)")

        st.caption(
            "This is  ML demo trained on the UCI Heart Disease dataset "
            "(~83% cross-validated accuracy). Not a medical diagnostic tool."
        )

with tab2:
    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=df, ax=ax)
        ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Importance (Tuned Random Forest)")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x=importances.values, y=importances.index, ax=ax)
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.subheader("Numeric Feature Distribution vs Target")
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    selected_col = st.selectbox("Choose a feature", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(x='target', y=selected_col, data=df, ax=ax)
    st.pyplot(fig)
