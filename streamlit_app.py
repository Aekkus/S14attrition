
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os

st.set_page_config(page_title="Attrition Prediction App", layout="centered")
st.title("Employee Attrition Prediction")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURE_INFO = os.path.join(BASE_DIR, "feature_info.json")

if not os.path.exists(MODEL_PATH):
    st.warning("model.pkl not found. Please run the training script (train_model.py) in Colab to create model.pkl and upload it to this folder.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)
model = payload.get('model')
model_columns = payload.get('columns', [])

# Try to load feature_info to build UI guidance
if os.path.exists(FEATURE_INFO):
    with open(FEATURE_INFO, "r") as f:
        feat_meta = json.load(f)
    features = feat_meta.get('features', model_columns if model_columns else [])
else:
    features = model_columns if model_columns else []

st.subheader("Provide employee details")
user_input = {}
for col in features:
    # Simple heuristics to display input types
    if col.lower().find("age")!=-1 or col.lower().find("years")!=-1 or col.lower().find("rate")!=-1 or col.lower().find("income")!=-1 or col.lower().find("distance")!=-1 or col.lower().find("num")!=-1:
        val = st.number_input(col, value=30, step=1)
    elif col.lower() in ['gender','overtime','over18'] or col.lower().find('yes')!=-1:
        val = st.selectbox(col, options=["Yes","No"])
        val = 1 if val=="Yes" or val=="Male" else 0
    else:
        val = st.text_input(col, value="0")
        # try cast to float if possible
        try:
            val = float(val)
        except:
            val = 0.0
    user_input[col] = val

if st.button("Predict attrition probability"):
    X = pd.DataFrame([user_input])
    # align columns
    for c in model_columns:
        if c not in X.columns:
            X[c] = 0
    X = X[model_columns]
    proba = model.predict_proba(X)[:,1][0] if hasattr(model, 'predict_proba') else model.predict(X)[0]
    st.metric("Attrition probability", f"{proba:.3f}")
    st.write("Prediction (0 = No, 1 = Yes):", int(proba>=0.5))
