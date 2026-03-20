import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Train Model
df = pd.read_csv('mammographic_masses.data.txt',
                 na_values=['?'],
                 names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
df.dropna(inplace=True)

X = df[['age', 'shape', 'margin', 'density']].values
y = df['severity'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# UI
st.title("🩺 Mammogram Mass Classifier")
st.subheader("Predict whether a mass is Benign or Malignant")

st.sidebar.header("Enter Patient Data")

age     = st.sidebar.slider("Age", 18, 100, 50)
shape   = st.sidebar.selectbox("Shape",   [1, 2, 3, 4], format_func=lambda x: {1:"Round", 2:"Oval", 3:"Lobular", 4:"Irregular"}[x])
margin  = st.sidebar.selectbox("Margin",  [1, 2, 3, 4, 5], format_func=lambda x: {1:"Circumscribed", 2:"Microlobulated", 3:"Obscured", 4:"Ill-defined", 5:"Spiculated"}[x])
density = st.sidebar.selectbox("Density", [1, 2, 3, 4], format_func=lambda x: {1:"High", 2:"Iso", 3:"Low", 4:"Fat-containing"}[x])

if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[age, shape, margin, density]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 0:
        st.success("✅ Result: BENIGN")
        st.info(f"Confidence: {probability[0]*100:.1f}%")
    else:
        st.error("⚠️ Result: MALIGNANT")
        st.info(f"Confidence: {probability[1]*100:.1f}%")